# Coupled Spring Example

The coupled spring example demonstrates multi-entity topology: three masses connected
in a chain by two springs.

Source: `src/numen/examples/coupled_spring/`

## Topology

```
m1 --- s1 --- m2 --- s2 --- m3
```

Three masses (`MassComponent`) and two springs (`SpringComponent`).
The spring force system connects them via `entity_groups`.

## Components

```python
class MassComponent(Component):
    kind:     Literal["mass"] = "mass"
    position: Annotated[float, IntegratedField()] = 0.0
    velocity: Annotated[float, IntegratedField()] = 0.0
    mass:     Annotated[float, ParameterField()]  = 1.0

class SpringComponent(Component):
    kind:        Literal["spring"] = "spring"
    k:           Annotated[float, ParameterField()] = 1.0
    rest_length: Annotated[float, ParameterField()] = 0.0
```

`SpringComponent` has no state fields — it's pure parameters. The topology
(which spring connects which masses) lives entirely in the system.

## Systems

Two systems are required: kinematics (ẋ = v) and spring forces.

```python
# Single-entity system — auto-populates entity_ids from component_types
class MassKinematicsSystem(System):
    component_types: ClassVar = (MassComponent,)
    python_fn:       ClassVar[DynamicsFn] = staticmethod(mass_kinematics_dynamics)
    kind:            Literal["mass_kinematics"] = "mass_kinematics"
    dynamics_fn:     str = "CoupledSpring.mass_kinematics!"

# Multi-entity coupled system — topology declared via entity_slots + entity_groups
class SpringForceSystem(System):
    component_types: ClassVar = ()   # no auto-population
    entity_slots:    ClassVar[EntityGroup] = EntityGroup(
        MassComponent, SpringComponent, MassComponent   # group_size = 3
    )
    python_fn:  ClassVar[DynamicsFn] = staticmethod(spring_force_dynamics)
    kind:       Literal["spring_force"] = "spring_force"
    dynamics_fn: str = "CoupledSpring.spring_force!"
```

The spring force dynamics:

```python
def spring_force_dynamics(dx, x, p, t, spec, system):
    for id_a, id_s, id_b in system.entity_groups:   # group_size=3 unpacked here
        a  = spec.view(id_a, MassComponent,   x, p)
        b  = spec.view(id_b, MassComponent,   x, p)
        s  = spec.view(id_s, SpringComponent, x, p)
        da = spec.dx_view(id_a, MassComponent, dx)
        db = spec.dx_view(id_b, MassComponent, dx)
        force = s.k * ((b.position - a.position) - s.rest_length)
        da.velocity +=  force / a.mass
        db.velocity += -force / b.mass
```

## World assembly

```python
def make_world():
    return World(
        components={
            "m1": {"mass":   MassComponent(position=0.0, mass=1.0)},
            "s1": {"spring": SpringComponent(k=10.0, rest_length=1.0)},
            "m2": {"mass":   MassComponent(position=1.0, mass=1.0)},
            "s2": {"spring": SpringComponent(k=10.0, rest_length=1.0)},
            "m3": {"mass":   MassComponent(position=3.0, mass=1.0)},
        },
        systems={
            "kinematics": MassKinematicsSystem(),
            "spring_force": SpringForceSystem(entity_groups=[
                ["m1", "s1", "m2"],
                ["m2", "s2", "m3"],   # m2 in both groups: forces accumulate via +=
            ]),
        },
    )
```

## What `compile_spec` produces

For this world:
- `state_index_map` has 6 entries (position + velocity for each of m1, m2, m3).
- `param_index_map` has 7 entries (mass × 3, k × 2, rest_length × 2).
- `MassKinematicsSystem` entity_ids: `["m1", "m2", "m3"]`, group_size=1.
- `SpringForceSystem` entity_ids: `["m1", "s1", "m2", "m2", "s2", "m3"]`, group_size=3.

Note that `m2` appears twice in `SpringForceSystem.entity_ids`. The `+=` accumulation
in the dynamics function sums the forces from both springs correctly.

## Key concepts illustrated

1. **Topology in systems, not components.** `SpringComponent` doesn't know which masses it connects.
2. **EntityGroup** declares the slot types and group_size for multi-entity systems.
3. **`+=` accumulation** — multiple systems writing the same entity's derivative add correctly.
4. `compile_spec` validates each entity group: `["m1", "s1", "m2"]` must contain
   `(MassComponent, SpringComponent, MassComponent)` in that order.

## Run it

```bash
uv run numen run coupled_spring
```
