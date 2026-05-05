from __future__ import annotations

import numpy as np

from numen.compiler.flatten import CompiledSpec
from numen.bridge.runtime import SolveResult


def reconstruct_snapshot(world: any, spec: CompiledSpec, result: SolveResult, t: float) -> any:
    """Rebuild a GenericWorld snapshot from solver output at time t.

    Returns a deep copy of world with all compiled fields updated to their values at t.
    """
    idx = int(np.searchsorted(result.t, t))
    idx = min(idx, result.x.shape[1] - 1)
    x = result.x[:, idx]

    snapshot = world.model_copy(deep=True)

    for entity_id, component in snapshot.components.items():
        updates: dict[str, any] = {}
        for key, (start, end) in spec.state_index_map.items():
            eid, field_name = key.split(".", 1)
            if eid != entity_id:
                continue
            value = float(x[start]) if end - start == 1 else x[start:end].tolist()
            updates[field_name] = value

        if updates:
            snapshot.components[entity_id] = component.model_copy(update=updates)

    return snapshot
