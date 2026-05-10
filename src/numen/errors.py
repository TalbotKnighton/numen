"""Numen framework errors and backend compatibility checks.

``NumenFeatureError`` is raised before any solver work begins when the
selected backend does not support a feature required by the compiled spec.
This gives an actionable error message with a hint about which backend to
use instead.

``NumenMissingFnError`` is raised when a system is missing the required
dynamics function for the selected backend (``python_fn`` for scipy/JAX,
``dynamics_fn`` for Julia).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numen.compiler.flatten import CompiledSpec


class NumenError(RuntimeError):
    """Base class for all Numen framework errors."""


class NumenFeatureError(NumenError):
    """A ``CompiledSpec`` requires features not supported by the selected backend.

    Raised by ``check_backend_features`` before the solve begins, with a hint
    about which backend to use instead.
    """


class NumenMissingFnError(NumenError):
    """A ``System`` is missing the required dynamics function for the selected backend.

    Raised by ``check_python_fns`` (scipy/JAX) or ``check_julia_fns`` (Julia)
    before the solve begins.
    """


_FEATURE_HINTS: dict[str, str] = {
    "control_callbacks": (
        "Use ScipyBackend (segment-solve) or JuliaServerBackend (PeriodicCallback) instead."
    ),
    "dae_constraints": (
        "ContinuousField(algebraic=True) requires an implicit Julia solver.\n"
        "  Use JuliaBackend(method='Rodas5P') or JuliaServerBackend(method='Rodas5P').\n"
        "  Set method='FBDF' for very stiff DAE systems."
    ),
}


def check_backend_features(
    spec: "CompiledSpec",
    backend_name: str,
    supported: frozenset[str],
) -> None:
    """Raise NumenFeatureError if spec requires features not supported by this backend."""
    unsupported = spec.required_features - supported
    if unsupported:
        hints = "\n".join(
            f"  {f}: {_FEATURE_HINTS[f]}" if f in _FEATURE_HINTS else f"  {f}"
            for f in sorted(unsupported)
        )
        raise NumenFeatureError(
            f"{backend_name} does not support: {sorted(unsupported)}\n"
            f"Supported by this backend: {sorted(supported)}\n"
            f"\n{hints}"
        )


def check_python_fns(spec: "CompiledSpec", backend_name: str) -> None:
    """Raise NumenMissingFnError if any system lacks a python_fn."""
    missing = [s.dynamics_fn or repr(s) for s in spec.systems if s.python_fn is None]
    if missing:
        raise NumenMissingFnError(
            f"{backend_name} requires python_fn on all Systems.\n"
            f"Missing for: {missing}\n"
            f"Declare 'python_fn: ClassVar[DynamicsFn] = staticmethod(your_fn)' on the System class."
        )


def check_julia_fns(spec: "CompiledSpec", backend_name: str) -> None:
    """Raise NumenMissingFnError if any system has an empty dynamics_fn."""
    missing = [repr(s) for s in spec.systems if not s.dynamics_fn]
    if missing:
        raise NumenMissingFnError(
            f"{backend_name} requires dynamics_fn on all Systems.\n"
            f"Missing for: {missing}\n"
            f"Set 'dynamics_fn: str = \"Module.function_name!\"' on each System class."
        )
