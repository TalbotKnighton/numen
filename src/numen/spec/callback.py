from pydantic import BaseModel


class Callback(BaseModel):
    """Base class for Python-tier callbacks (control logic, logging, snapshots).

    Called at discrete checkpoints between solver steps — not inside the integrator
    loop. For tight-loop state modifications, use Julia ContinuousCallback/DiscreteCallback.

    Example:
        class ControlCallback(Callback):
            kind: Literal["control"] = "control"
            dt: float = 0.01

            def __call__(self, world: GenericWorld, t: float) -> dict[str, float]:
                ...
    """

    model_config = {"frozen": True}
