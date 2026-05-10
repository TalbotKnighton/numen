from __future__ import annotations

import numpy as np

from numen.compiler.flatten import CompiledSpec
from numen.bridge.runtime import SolveResult


def reconstruct_snapshot(world: any, spec: CompiledSpec, result: SolveResult, t: float) -> any:
    """Rebuild a GenericWorld snapshot from solver output at time t.

    Returns a deep copy of world with all compiled fields updated to their values at t.
    State keys use the full path ``entity_id.component_kind.field_name``.
    """
    idx = int(np.searchsorted(result.t, t))
    idx = min(idx, result.x.shape[1] - 1)
    x = result.x[:, idx]

    snapshot = world.model_copy(deep=True)

    for entity_id, comps_by_kind in snapshot.components.items():
        for kind, component in list(comps_by_kind.items()):
            updates: dict[str, any] = {}
            prefix = f"{entity_id}.{kind}."
            for key, (start, end) in spec.state_index_map.items():
                if not key.startswith(prefix):
                    continue
                field_name = key[len(prefix):]
                value = float(x[start]) if end - start == 1 else x[start:end].tolist()
                updates[field_name] = value
            if updates:
                snapshot.components[entity_id][kind] = component.model_copy(update=updates)

    return snapshot
