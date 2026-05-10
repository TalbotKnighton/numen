"""Post-processing utilities for accessing solver results.

``SnapshotCollector`` provides two access patterns:

1. **Field series** — direct array extraction for a single field, without
   reconstructing full world objects.  Fast and suitable for plotting.

2. **World snapshots** — full ``GenericWorld`` deep-copies with all compiled
   fields updated to their values at a given time ``t``.  Suitable for
   inspection, serialisation, and downstream computation.

Example::

    from numen.reconstruction.collector import SnapshotCollector

    collector = SnapshotCollector(world, spec, result)

    # Fast: extract one field as a time series
    t, position = collector.field_series("osc", "oscillator", "position")

    # Slow but rich: reconstruct typed world object at a time point
    snap = collector.at(t=1.5)
    print(snap.components["osc"]["oscillator"].position)
"""
from __future__ import annotations

import numpy as np

from numen.compiler.flatten import CompiledSpec
from numen.bridge.runtime import SolveResult
from numen.reconstruction.snapshot import reconstruct_snapshot


class SnapshotCollector:
    """Reconstructs GenericWorld snapshots from solver output at arbitrary times.

    Args:
        world:  The original ``GenericWorld`` instance used for the solve.
        spec:   The ``CompiledSpec`` produced by ``compile_spec(world)``.
        result: The ``SolveResult`` from a backend ``solve()`` call.
    """

    def __init__(self, world: any, spec: CompiledSpec, result: SolveResult):
        self.world = world
        self.spec = spec
        self.result = result

    def at(self, t: float) -> any:
        """Reconstruct a full world snapshot at simulation time ``t``.

        Uses binary search on ``result.t`` to find the nearest saved time point,
        then rebuilds the world with all compiled state fields updated.

        Args:
            t: Simulation time in seconds.

        Returns:
            A deep copy of the original world with state fields updated to
            their values at time ``t``.
        """
        return reconstruct_snapshot(self.world, self.spec, self.result, t)

    def at_times(self, times: list[float]) -> list[tuple[float, any]]:
        """Reconstruct world snapshots at a list of times.

        Args:
            times: List of simulation times in seconds.

        Returns:
            List of ``(t, snapshot)`` tuples in the same order as ``times``.
        """
        return [(t, self.at(t)) for t in times]

    def uniform(self, n: int = 100) -> list[tuple[float, any]]:
        """Reconstruct world snapshots at ``n`` uniformly-spaced times.

        Args:
            n: Number of snapshots. Default 100.

        Returns:
            List of ``(t, snapshot)`` tuples spanning the full solve interval.
        """
        times = np.linspace(self.result.t[0], self.result.t[-1], n)
        return self.at_times(times.tolist())

    def field_series(
        self,
        entity_id: str,
        component_kind: str,
        field_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a single field's time series directly from the result arrays.

        Args:
            entity_id:      Entity key (e.g. "osc").
            component_kind: Component kind string (e.g. "nl_oscillator").
            field_name:     Field name on the component (e.g. "position").

        Returns:
            (t, values) — faster than reconstructing full snapshots.
        """
        key = f"{entity_id}.{component_kind}.{field_name}"
        if key in self.spec.state_index_map:
            start, end = self.spec.state_index_map[key]
            values = self.result.x[start:end, :]
        elif key in self.spec.param_index_map:
            start, end = self.spec.param_index_map[key]
            p = np.array(self.spec.p)
            values = np.tile(p[start:end], (len(self.result.t), 1)).T
        else:
            raise KeyError(f"Field '{key}' not found in compiled spec")

        if values.shape[0] == 1:
            return self.result.t, values[0]
        return self.result.t, values
