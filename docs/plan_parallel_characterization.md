# Parallel Characterization — Design Plan

> **Status: COMPLETE.** All steps implemented and verified against the nonlinear
> oscillator example campaign.  Architectural notes graduated to `CLAUDE.md`.
> This file is retained as a record of the design decisions made.

---

## North Star

Allow a characterization campaign to distribute DOE-level work across N Julia server
processes so that a sweep over 20 parameter combinations on a 4-core machine runs in
roughly 20/4 = 5× less wall time, with zero change to results and zero extra boilerplate
in the YAML.

---

## Constraints / non-goals

| Constraint | Rationale |
|---|---|
| Parallelism only for `julia_server` backend | Pool management is Julia-process-level; scipy/JAX/julia backends are single-threaded. Non-Julia paths ignore `n_workers`. |
| DOE-level granularity, not leaf-solve | A single sub-test (FRF sweep at one DC offset, chirp at one amplitude, …) runs on **one server** start-to-finish. This lets the first solve on each server JIT-compile all dynamics; subsequent solves on that server are warm. Splitting solves within a sub-test across servers would lose the JIT benefit and add per-solve overhead. |
| `n_workers=1` default | Single-server path is unchanged. YAML opt-in. |
| Non-sweep tests route through one worker | `continuous_chirp`, `discrete_frequency_sweep`, `amplitude_sweep`, `dc_operating_point_sweep` defined at the top level (not as `sub_test`) are not parallelised. They already run on the single shared server. |
| No result ordering change | Results must appear in the same order as `test.values` / design-point list regardless of which worker finishes first. |

---

## User-facing API

### YAML

```yaml
backend:
  type: julia_server
  julia_file: dynamics.jl
  method: Tsit5
  rtol: 1.0e-8
  atol: 1.0e-9
  n_workers: 4          # NEW — default 1
```

### CLI override (always wins over YAML)

```bash
numen characterize test_plan.yaml --workers 4
numen characterize test_plan.yaml --workers 1   # force sequential even if YAML says 4
```

---

## Architecture

### Current flow (n_workers=1)

```
CharacterizationRunner.run()
  backend = JuliaServerBackend(...)         # 1 process

  for test in config.tests:
      _run_test(test, backend)

      ParameterSweepSpec → run_parameter_sweep(test, exc_spec, sub_runner, ...)
        for val in test.values:             # SEQUENTIAL
            spec_v = _set_model_param(exc_spec, key, val)
            sub_runner(spec_v)              # sub_runner closes over backend
```

### Target flow (n_workers>1)

```
CharacterizationRunner.run()
  pool = JuliaServerPool(n_workers=N, ...)  # N processes, all started eagerly

  for test in config.tests:
      _run_test(test, pool)

      ParameterSweepSpec → run_parameter_sweep_parallel(test, exc_spec, sub_runner_factory, pool)
        specs_v = [_set_model_param(exc_spec, key, val) for val in test.values]
        pool.map(
            lambda server, spec_v: sub_runner_factory(server)(spec_v),
            specs_v,
        )                                   # PARALLEL across N servers
                                            # results returned in input order
```

`pool.map` already exists in `JuliaServerPool` (`server_backend.py:302`) and returns
results in input order by allocating by index.

---

## Key design decision — sub_runner_factory

Currently `_make_sub_runner(sub_spec_obj, backend, context)` closes over `backend` at
construction time (runner.py:237). This makes it impossible to hand the same sub-test to
a different server per design point.

**Change:** split into a factory that captures everything except the server, plus a
call-site that supplies the server:

```python
# New signature — backend NOT captured at construction
def _make_sub_runner_factory(self, sub_spec_obj, context):
    """Return Callable[[backend, spec_v], result]."""
    e_id   = self.config.excitation.entity
    e_port = self.config.excitation.port
    out    = self._output_key

    def _runner(backend, spec_v):
        if isinstance(sub_spec_obj, DiscreteFrequencySweepSpec):
            from numen.characterization.tests.freq_sweep import run_discrete_frequency_sweep
            return run_discrete_frequency_sweep(sub_spec_obj, spec_v, e_id, e_port, out, backend)
        # … other types …
    return _runner

# Sequential path (n_workers=1):
factory = self._make_sub_runner_factory(sub, test.name)
sub_runner = lambda spec_v: factory(backend, spec_v)   # close over the single backend
run_parameter_sweep(test, exc_spec, sub_runner, ...)

# Parallel path (n_workers>1):
factory = self._make_sub_runner_factory(sub, test.name)
run_parameter_sweep_parallel(test, exc_spec, factory, pool, ...)
  # pool.map(lambda server, spec_v: factory(server, spec_v), all_specs_v)
```

The three sweep runners (`param_sweep`, `param_grid`, `doe_sweep`) each get a
`_parallel` variant or a unified `parallel` kwarg. Simplest: add a `parallel=False`
flag and a `pool=None` argument; when `pool` is provided, call `pool.map`.

---

## Files to change

### 1. `src/numen/characterization/schema.py`

Add `n_workers: int = 1` to `BackendSpec`:

```python
class BackendSpec(BaseModel):
    type:       Literal["scipy", "jax", "julia", "julia_server"] = "scipy"
    julia_file: str | None = None
    method:     str | None = None
    rtol:       float = 1e-8
    atol:       float = 1e-9
    n_workers:  int   = 1          # NEW
```

### 2. `src/numen/cli.py`

Add `--workers` option to `characterize` command:

```python
@app.command()
def characterize(
    plan: Path = typer.Argument(...),
    only_characterize: bool = typer.Option(False, "--characterize", "-c"),
    only_plot:         bool = typer.Option(False, "--plot", "-p"),
    workers:           int  = typer.Option(0, "--workers", "-w",
                                           help="Parallel Julia servers (0 = use YAML value)"),
):
    ...
    if workers > 0:
        config = config.model_copy(update={
            "backend": config.backend.model_copy(update={"n_workers": workers})
        })
```

### 3. `src/numen/characterization/runner.py`

**`_backend_context`:** open a `JuliaServerPool` instead of `JuliaServerBackend` when
`n_workers > 1`:

```python
@contextmanager
def _backend_context(spec: BackendSpec) -> Generator[Any, None, None]:
    if spec.type == "julia_server" and spec.n_workers > 1:
        from numen.bridge.server_backend import JuliaServerPool
        pool = JuliaServerPool(
            n_workers  = spec.n_workers,
            julia_file = spec.julia_file,
            method     = spec.method or "Tsit5",
            rtol       = spec.rtol,
            atol       = spec.atol,
        )
        with pool:
            _log.info("Julia pool started (%d workers)", spec.n_workers)
            yield pool
    else:
        # existing path
        backend = _open_backend(spec)
        if spec.type == "julia_server":
            with backend:
                yield backend
        else:
            with nullcontext(backend):
                yield backend
```

**`_run_test`:** pass `pool` through to sweep methods that can exploit it:

```python
def _run_test(self, test: TestSpec, backend_or_pool: Any) -> Any:
    ...
    if isinstance(test, ParameterSweepSpec):
        return self._run_parameter_sweep(test, backend_or_pool)
    if isinstance(test, ParameterGridSpec):
        return self._run_parameter_grid(test, backend_or_pool)
    if isinstance(test, DOESweepSpec):
        return self._run_doe_sweep(test, backend_or_pool)
    # leaf tests — use a single server from pool, or the backend directly
    actual_backend = _one_server(backend_or_pool)
    ...
```

`_one_server(b)`: if `b` is a `JuliaServerPool`, acquire-and-release using
`pool._q.get()` / put (or add a `JuliaServerPool.one()` context manager); otherwise
return `b` unchanged.

**Rename `_make_sub_runner` → `_make_sub_runner_factory`** (returns a 2-arg callable as
described above).

**`_run_parameter_sweep`, `_run_parameter_grid`, `_run_doe_sweep`:** detect pool and
dispatch:

```python
def _run_parameter_sweep(self, test: ParameterSweepSpec, backend_or_pool: Any) -> Any:
    from numen.characterization.tests.param_sweep import run_parameter_sweep
    sub = self._get_sub_spec(test.sub_test, test.name)
    exc = self.config.excitation
    factory = self._make_sub_runner_factory(sub, test.name)

    if isinstance(backend_or_pool, JuliaServerPool):
        return _run_sweep_parallel(test, self._exc_spec, factory, backend_or_pool,
                                   entity_id=exc.entity, port_name=exc.port)
    else:
        sub_runner = lambda spec_v: factory(backend_or_pool, spec_v)
        return run_parameter_sweep(test, self._exc_spec, sub_runner,
                                   entity_id=exc.entity, port_name=exc.port)
```

### 4. `src/numen/characterization/tests/param_sweep.py`

Add `run_parameter_sweep_parallel` (or fold into `run_parameter_sweep` with a
`pool` kwarg):

```python
def run_parameter_sweep_parallel(
    test: ParameterSweepSpec,
    exc_spec: Any,
    factory: Callable[[Any, Any], Any],   # (server, spec_v) → result
    pool: Any,                             # JuliaServerPool
    entity_id: str | None = None,
    port_name: str | None = None,
) -> ParameterFamilyResult:
    all_specs = [
        _set_model_param(exc_spec, test.sweep_param, val, entity_id, port_name)
        for val in test.values
    ]
    sub_results = pool.map(factory, all_specs)
    result_obj = ParameterFamilyResult(
        name=test.name, sweep_param=test.sweep_param, param_values=list(test.values),
    )
    result_obj.sub_results.extend(sub_results)
    return result_obj
```

Same pattern for `param_grid.py` (`run_parameter_grid_parallel`) and
`doe_sweep.py` (`run_doe_sweep_parallel`).

---

## JIT precompilation — replacing dummy warmup solves

### Problem

Without precompilation, the first solve on each pool worker triggers Julia's JIT
compilation. If we DON'T do a nominal warmup solve, the first real solve on each server
carries JIT latency. With N workers the aggregate JIT overhead is N × compile time.

### Solution — `precompile()` in server.jl

Julia's `precompile(fn, (arg_types...))` triggers type-inference and native code
generation **without executing the function**. All Numen dynamics functions use one
concrete signature:

```julia
fn(::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Float64,
   ::CompiledSpec, ::CompiledSystemSpec) → nothing
```

So we can precompile every dynamics function after loading user files by scanning all
top-level modules in `Main`:

```julia
# In server.jl, after: for f in ARGS; include(f); end

const _DYNAMICS_SIG = (
    Vector{Float64}, Vector{Float64}, Vector{Float64},
    Float64, CompiledSpec, CompiledSystemSpec,
)

function _precompile_dynamics_modules()
    n = 0
    for sym in names(Main; all=false)
        mod = getfield(Main, sym)
        mod isa Module || continue
        for fn_sym in names(mod; all=false)
            fn = try getfield(mod, fn_sym) catch; continue end
            fn isa Function || continue
            if precompile(fn, _DYNAMICS_SIG)
                n += 1
            end
        end
    end
    println(stderr, "NUMEN_PRECOMPILE_DONE ($n functions)")
    flush(stderr)
end

_precompile_dynamics_modules()
```

`precompile` returns `true` on success, `false` if the method doesn't exist for those
types (silently ignored). Framework functions in `NumenCharacterization` (loaded by
server.jl itself) are included automatically.

**Result:** each worker exits its startup phase with all dynamics functions compiled.
The first real solve incurs no JIT latency — it runs at full compiled speed immediately.

### Python side — wait for precompile signal

In `JuliaServerBackend._ensure_started`, extend the ready signal check:

```python
_READY_SIGNAL     = "NUMEN_SERVER_READY"
_PRECOMPILE_DONE  = "NUMEN_PRECOMPILE_DONE"

def _read_stderr(self) -> None:
    for raw in proc.stderr:
        decoded = raw.decode(errors="replace").rstrip()
        if decoded == _READY_SIGNAL:
            ready_event.set()
        elif decoded.startswith(_PRECOMPILE_DONE):
            _log.info("[julia] %s", decoded)    # logs how many functions compiled
        elif decoded.strip():
            _log.debug("[julia] %s", decoded)
```

The precompile step runs after the packages are loaded but before `NUMEN_SERVER_READY`,
so the existing ready-wait mechanism covers it automatically — Python waits for
`NUMEN_SERVER_READY`, which only comes after precompilation completes.

Actually slightly cleaner: emit `NUMEN_SERVER_READY` **after** precompilation. That way
the pool's `eager=True` startup already guarantees all workers are compiled before the
first request is dispatched.

---

## Ordering and thread-safety

- `JuliaServerPool.map` already allocates results by index (`results[idx] = fn(...)`)
  — result order matches input order regardless of finish order (server_backend.py:324).
- Each `JuliaServerBackend` has its own `threading.Lock` (`_lock`); the pool ensures at
  most one request per server at a time via the queue.
- `CompiledSpec` is a frozen dataclass — safe to read from multiple threads.
- The sub-test runners (`run_discrete_frequency_sweep`, etc.) do not write shared state;
  they return new result objects.
- The pool `map` uses a `ThreadPoolExecutor` with `max_workers=n_workers` — same as the
  pool size, so threads never contend for server slots.

---

## Unchanged paths

| Scenario | Behaviour |
|---|---|
| `n_workers=1` (default) | Sequential, single server, existing code paths unchanged |
| `backend.type != julia_server` | `n_workers` ignored; sequential execution |
| Top-level non-sweep test | Routes to one server; not parallelized |
| Nested sub-tests that are themselves sweeps | Out of scope (not supported today; treat as error) |

---

## Implementation order

1. **YAML schema** — add `n_workers` to `BackendSpec`. Zero risk.
2. **CLI** — add `--workers` option to `characterize`. Zero risk.
3. **`server.jl` precompile** — add `_precompile_dynamics_modules()` to server.jl.
   Test: start a server, look for `NUMEN_PRECOMPILE_DONE` in logs.
4. **Runner: factory refactor** — rename `_make_sub_runner` → `_make_sub_runner_factory`,
   change signature to `(sub_spec_obj, context) → (backend, spec_v) → result`. Update
   `_run_parameter_sweep`, `_run_parameter_grid`, `_run_doe_sweep` to use new factory.
   `n_workers=1` still sequential; full regression test.
5. **Runner: pool dispatch** — `_backend_context` opens `JuliaServerPool` when
   `n_workers > 1`; sweep methods call `_run_*_parallel` when they receive a pool.
6. **Sweep runners** — add parallel variants to `param_sweep.py`, `param_grid.py`,
   `doe_sweep.py`.
7. **End-to-end test** — run `numen characterize test_plan.yaml --workers 4` on
   nonlinear oscillator; verify results match `--workers 1`, check wall time.

---

## Open questions

- Should `--workers` default to `0` (use YAML) or to the machine's CPU count? Current
  plan: `0` = use YAML. The user can always add `n_workers: $(nproc)` in YAML for
  auto-scale; explicit beats implicit here.
- Pool startup is eager (`eager=True` in `JuliaServerPool.__init__`). Startup time is
  ~8–15 s per server and they all start in parallel (each is its own thread waiting for
  `NUMEN_SERVER_READY`). Should we surface per-worker startup time in the log? Yes — add
  a pool-level `INFO` line after all workers are ready: "Julia pool ready: N workers in
  X.X s".
- `_one_server` helper for routing top-level leaf tests to a pool worker — should this
  use `pool._q.get()` directly (accessing a private attribute) or should `JuliaServerPool`
  expose a `borrow()` context manager? Prefer a public context manager: cleaner, no
  private-attribute coupling.
