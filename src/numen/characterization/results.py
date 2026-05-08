"""Typed result containers for characterization campaigns.

All containers are plain dataclasses (not Pydantic) so numpy arrays can be
stored directly.  JSON serialisation converts arrays to lists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Per-test result types
# ---------------------------------------------------------------------------

@dataclass
class FRFResult:
    """Result of a discrete frequency sweep — full Bode data + extracted metrics.

    magnitudes are dimensionless transfer function magnitudes |H(f)| =
    output_amplitude / input_amplitude.  phases_deg are in degrees.
    """
    name:          str
    frequencies:   np.ndarray         # Hz
    magnitudes:    np.ndarray         # |H(f)|
    phases_deg:    np.ndarray         # degrees
    f0:            float | None       # resonant frequency [Hz]
    Q:             float | None       # quality factor
    damping_ratio: float | None       # ζ = 1/(2Q)
    amplitude:     float = 0.0        # drive amplitude used
    dc_offset:     float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":          self.name,
            "type":          "frf",
            "frequencies":   self.frequencies.tolist(),
            "magnitudes":    self.magnitudes.tolist(),
            "phases_deg":    self.phases_deg.tolist(),
            "f0":            self.f0,
            "Q":             self.Q,
            "damping_ratio": self.damping_ratio,
            "amplitude":     self.amplitude,
            "dc_offset":     self.dc_offset,
        }


@dataclass
class OperatingPointMeasurement:
    """One small-signal measurement from a DC operating-point sweep."""
    dc_value:  float
    magnitude: float           # |H(f_probe)| at the probe frequency
    phase_deg: float           # degrees
    f0:        float | None    # resonant frequency if a sub-FRF was also run
    Q:         float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dc_value":  self.dc_value,
            "magnitude": self.magnitude,
            "phase_deg": self.phase_deg,
            "f0":        self.f0,
            "Q":         self.Q,
        }


@dataclass
class DCSweptFRFResult:
    """Result of a DC operating-point sweep."""
    name:            str
    probe_frequency: float
    measurements:    list[OperatingPointMeasurement] = field(default_factory=list)

    @property
    def dc_values(self) -> list[float]:
        return [m.dc_value for m in self.measurements]

    @property
    def magnitudes(self) -> list[float]:
        return [m.magnitude for m in self.measurements]

    @property
    def phases_deg(self) -> list[float]:
        return [m.phase_deg for m in self.measurements]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":            self.name,
            "type":            "dc_swept_frf",
            "probe_frequency": self.probe_frequency,
            "measurements":    [m.to_dict() for m in self.measurements],
        }


@dataclass
class ParameterFamilyResult:
    """Result of a parameter sweep — a family of sub-test results indexed by param value."""
    name:        str
    sweep_param: str
    param_values: list[float]
    sub_results:  list[Any]    = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        sub_dicts = []
        for r in self.sub_results:
            sub_dicts.append(r.to_dict() if hasattr(r, "to_dict") else r)
        return {
            "name":         self.name,
            "type":         "parameter_family",
            "sweep_param":  self.sweep_param,
            "param_values": self.param_values,
            "sub_results":  sub_dicts,
        }


# ---------------------------------------------------------------------------
# Campaign-level accumulator
# ---------------------------------------------------------------------------

@dataclass
class CampaignResults:
    """Accumulates results across a full characterization campaign."""
    config_version: str = "1.0"
    _entries: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def append(self, test_name: str, result: Any) -> None:
        self._entries.append({"test": test_name, "result": result})

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"CampaignResults({len(self)} tests)"

    def get(self, name: str) -> Any:
        """Return the result for a named test, or None."""
        for e in self._entries:
            if e["test"] == name:
                return e["result"]
        return None

    def all_results(self) -> list[Any]:
        return [e["result"] for e in self._entries]

    def save(self, path: str | Path) -> None:
        """Serialise all results to a JSON file."""
        out = []
        for e in self._entries:
            r = e["result"]
            out.append(r.to_dict() if hasattr(r, "to_dict") else {"test": e["test"], "data": str(r)})
        with open(path, "w") as f:
            json.dump({"version": self.config_version, "results": out}, f,
                      indent=2, default=_json_default)
