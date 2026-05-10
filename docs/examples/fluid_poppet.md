# Fluid Poppet Example

The fluid poppet example is the most complex built-in model: a pneumatic
check valve network with four control volumes and a spring-mass poppet.
It demonstrates isentropic orifice flow, spring-mass mechanics with hard stops,
and multi-backend usage.

Source: `src/numen/examples/fluid_poppet/`

## Physical model

```
[supply CV] --orifice_in--> [inlet CV] --poppet flow--> [outlet CV] --orifice_out--> [vent CV]
                                              |
                                          [poppet mass + spring]
```

Four isothermal ideal-gas control volumes connected by orifices. The poppet
is a 1-DOF mass-spring system; its position determines the flow area between
inlet and outlet.

## Components

### ControlVolumeComponent

```python
class ControlVolumeComponent(Component):
    """Lumped-parameter fluid control volume — isothermal ideal gas."""

    kind:        Literal["control_volume"] = "control_volume"
    pressure:    Annotated[float, IntegratedField()] = 101_325.0   # Pa
    volume:      Annotated[float, ParameterField()]  = 1e-3        # m³
    temperature: Annotated[float, ParameterField()]  = 293.15      # K
    R_specific:  Annotated[float, ParameterField()]  = 287.058     # J/(kg·K)
```

State ODE: `dP/dt = (R·T/V) · Σṁ_net`

### OrificeComponent

```python
class OrificeComponent(Component):
    """Fixed-geometry orifice — carries only parameters."""

    kind:  Literal["orifice"] = "orifice"
    Cd:    Annotated[float, ParameterField()] = 0.7
    area:  Annotated[float, ParameterField()] = 1e-5    # m²
    gamma: Annotated[float, ParameterField()] = 1.4
```

### PoppetComponent

```python
class PoppetComponent(Component):
    """1-DOF poppet valve with spring preload, pressure-area forces, and hard stops."""

    kind:           Literal["poppet"] = "poppet"
    position:       Annotated[float, IntegratedField()] = 0.0    # m (0 = closed)
    velocity:       Annotated[float, IntegratedField()] = 0.0    # m/s
    mass:           Annotated[float, ParameterField()]  = 0.02   # kg
    spring_k:       Annotated[float, ParameterField()]  = 5_000.0
    spring_preload: Annotated[float, ParameterField()]  = 10.0   # N (closing)
    seat_area:      Annotated[float, ParameterField()]  = 5e-5   # m²
    max_travel:     Annotated[float, ParameterField()]  = 3e-3   # m
    stop_stiffness: Annotated[float, ParameterField()]  = 1e7
    stop_damping:   Annotated[float, ParameterField()]  = 100.0
    max_flow_area:  Annotated[float, ParameterField()]  = 5e-5   # m²
    Cd:             Annotated[float, ParameterField()]  = 0.7
    gamma:          Annotated[float, ParameterField()]  = 1.4
```

## Isentropic compressible orifice flow

The orifice flow computation in `dynamics.py` handles both choked and unchoked flow:

```python
def _orifice_mdot(P_up, P_dn, T_up, R, Cd, A, gamma):
    safe_P_up = jnp.maximum(P_up, 1e-300)
    beta      = jnp.maximum(0.0, P_dn) / safe_P_up
    beta_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    choke_exp      = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    mdot_choked    = Cd * A * P_up * jnp.sqrt(gamma / (R * T_up)) * (2.0/(gamma+1.0))**choke_exp

    arg            = beta**(2/gamma) - beta**((gamma+1)/gamma)
    mdot_unchoked  = Cd * A * P_up * jnp.sqrt(jnp.maximum(0.0, 2*gamma/((gamma-1)*R*T_up)*arg))

    mdot = jnp.where(beta <= beta_crit, mdot_choked, mdot_unchoked)
    return jnp.where((P_up <= 0.0) | (A <= 0.0), 0.0, mdot)
```

Note: both branches are always evaluated; `jnp.maximum(0.0, arg)` guards the sqrt.

## Smooth hard stops

The poppet has two hard stops (closed and fully open). A C1-smooth ramp over 1 µm
avoids the ODE solver pathology caused by a sharp `max(0, -pos)` force:

```python
_STOP_DELTA = 1e-6

def _soft_pen(pos_from_stop):
    x = pos_from_stop
    return jnp.where(
        x <= 0.0, 0.0,
        jnp.where(x >= _STOP_DELTA, x - 0.5 * _STOP_DELTA, 0.5 * x * x / _STOP_DELTA)
    )

# Velocity damping blended in over the same 1 µm
alpha  = jnp.clip(-pos / _STOP_DELTA, 0.0, 1.0)
v_damp = jnp.maximum(0.0, -vel) * alpha
F_close = stop_k * _soft_pen(-pos) + stop_c * v_damp
```

## Backend selection

This model is stiff (pressure ≈ 1e5 Pa, poppet dynamics at kHz). The performance
breakdown for a 150 ms simulation:

| Backend | Time | Notes |
|---|---|---|
| `ScipyBackend(RK45)` | ~9 s | Development / debugging |
| `JAXBackend(Dopri5)` | ~6 ms | ~1500× speedup, warm |
| `JuliaBackend(Tsit5)` | ~14 ms | ~600× speedup, warm |

```python
from numen.bridge.scipy_backend import ScipyBackend
from numen.bridge.jax_backend import JAXBackend
from numen.bridge.runtime import JuliaBackend

# Development
result = ScipyBackend().solve(spec, tspan=(0.0, 0.15))

# Production (after JIT warm-up)
result = JAXBackend(solver="Dopri5", max_steps=200_000).solve(spec, tspan=(0.0, 0.15))

# Julia (single long simulation)
result = JuliaBackend("dynamics.jl", method="Tsit5").solve(spec, tspan=(0.0, 0.15))
```

!!! tip "Use Dopri5 for JAX"
    With state values at ~1e5 Pa, Tsit5 causes pathological step rejection with
    tight `atol`. Use `Dopri5` instead.

## Run it

```bash
uv run numen run fluid_poppet
```
