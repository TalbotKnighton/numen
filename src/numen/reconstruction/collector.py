from __future__ import annotations

import numpy as np

from numen.compiler.flatten import CompiledSpec
from numen.bridge.runtime import SolveResult
from numen.reconstruction.snapshot import reconstruct_snapshot


class SnapshotCollector:
    """Reconstructs GenericWorld snapshots from solver output at arbitrary times."""

    def __init__(self, world: any, spec: CompiledSpec, result: SolveResult):
        self.world = world
        self.spec = spec
        self.result = result

    def at(self, t: float) -> any:
        return reconstruct_snapshot(self.world, self.spec, self.result, t)

    def at_times(self, times: list[float]) -> list[tuple[float, any]]:
        return [(t, self.at(t)) for t in times]

    def uniform(self, n: int = 100) -> list[tuple[float, any]]:
        times = np.linspace(self.result.t[0], self.result.t[-1], n)
        return self.at_times(times.tolist())

    def field_series(self, entity_id: str, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract a single field's time series directly from the result arrays.

        Returns (t, values) — faster than reconstructing full snapshots.
        """
        key = f"{entity_id}.{field_name}"
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
