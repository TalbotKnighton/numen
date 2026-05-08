"""Typed result containers for characterization campaigns.

All containers are plain dataclasses (not Pydantic) so numpy arrays can be
stored directly.  JSON serialisation converts arrays to lists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
class AmplitudeSweepResult:
    """Result of an amplitude sweep at a fixed frequency.

    Varying the drive amplitude at a fixed frequency reveals amplitude-dependent
    nonlinearities: a linear system shows flat H_magnitudes regardless of amplitude.
    """
    name:                str
    frequency:           float
    drive_amplitudes:    np.ndarray    # input amplitudes swept [same units as force]
    response_amplitudes: np.ndarray   # output peak amplitude from lock-in
    phases_deg:          np.ndarray   # output phase [degrees]
    H_magnitudes:        np.ndarray   # response / drive (normalised transfer function)
    dc_offset:           float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":                self.name,
            "type":                "amplitude_sweep",
            "frequency":           self.frequency,
            "drive_amplitudes":    self.drive_amplitudes.tolist(),
            "response_amplitudes": self.response_amplitudes.tolist(),
            "phases_deg":          self.phases_deg.tolist(),
            "H_magnitudes":        self.H_magnitudes.tolist(),
            "dc_offset":           self.dc_offset,
        }


@dataclass
class ChirpResult:
    """Result of a continuous chirp solve with FRF extracted via cross-spectrum."""
    name:         str
    t:            np.ndarray    # time array
    output:       np.ndarray    # raw output state time series
    frequencies:  np.ndarray   # Hz — trimmed to swept band
    H_mag:        np.ndarray   # |H(f)|
    H_phase_deg:  np.ndarray   # degrees
    f_start:      float
    f_end:        float
    amplitude:    float
    dc_offset:    float = 0.0
    chirp_type:   str   = "log"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":        self.name,
            "type":        "chirp",
            "t":           self.t.tolist(),
            "output":      self.output.tolist(),
            "frequencies": self.frequencies.tolist(),
            "H_mag":       self.H_mag.tolist(),
            "H_phase_deg": self.H_phase_deg.tolist(),
            "f_start":     self.f_start,
            "f_end":       self.f_end,
            "amplitude":   self.amplitude,
            "dc_offset":   self.dc_offset,
            "chirp_type":  self.chirp_type,
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


@dataclass
class ParameterGridResult:
    """Result of a full-factorial or pairwise parameter grid sweep."""
    name:         str
    param_keys:   list[str]
    combinations: list[dict[str, float]]
    sub_results:  list[Any] = field(default_factory=list)
    mode:         str       = "full_factorial"

    def to_dict(self) -> dict[str, Any]:
        sub_dicts = [r.to_dict() if hasattr(r, "to_dict") else r for r in self.sub_results]
        return {
            "name":         self.name,
            "type":         "parameter_grid",
            "param_keys":   self.param_keys,
            "combinations": self.combinations,
            "mode":         self.mode,
            "sub_results":  sub_dicts,
        }


@dataclass
class DOESweepResult:
    """Result of a space-filling or classical DOE sweep over continuous parameter ranges."""
    name:         str
    design:       str
    param_keys:   list[str]
    combinations: list[dict[str, float]]
    sub_results:  list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        sub_dicts = [r.to_dict() if hasattr(r, "to_dict") else r for r in self.sub_results]
        return {
            "name":         self.name,
            "type":         "doe_sweep",
            "design":       self.design,
            "param_keys":   self.param_keys,
            "combinations": self.combinations,
            "sub_results":  sub_dicts,
        }


# ---------------------------------------------------------------------------
# DataFrame flattening helpers
# ---------------------------------------------------------------------------

def _result_to_rows(
    test_name: str,
    result: Any,
    param_context: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Recursively convert one result object to a list of flat row dicts.

    param_context is used for grid/DOE/family results to carry the current
    design-point parameter values down into sub-results.
    """
    ctx: dict[str, Any] = {"test": test_name, **(param_context or {})}

    if isinstance(result, FRFResult):
        peak_mag = float(np.max(result.magnitudes)) if len(result.magnitudes) else None
        return [{
            **ctx,
            "f0":            result.f0,
            "Q":             result.Q,
            "damping_ratio": result.damping_ratio,
            "peak_magnitude": peak_mag,
            "amplitude":     result.amplitude,
            "dc_offset":     result.dc_offset,
        }]

    if isinstance(result, AmplitudeSweepResult):
        rows = []
        for i, amp in enumerate(result.drive_amplitudes):
            rows.append({
                **ctx,
                "frequency":          result.frequency,
                "drive_amplitude":    float(amp),
                "response_amplitude": float(result.response_amplitudes[i]),
                "H_magnitude":        float(result.H_magnitudes[i]),
                "phase_deg":          float(result.phases_deg[i]),
                "dc_offset":          result.dc_offset,
            })
        return rows

    if isinstance(result, DCSweptFRFResult):
        rows = []
        for m in result.measurements:
            rows.append({
                **ctx,
                "probe_frequency": result.probe_frequency,
                "dc_value":        m.dc_value,
                "magnitude":       m.magnitude,
                "phase_deg":       m.phase_deg,
            })
        return rows

    if isinstance(result, ChirpResult):
        peak_mag = float(np.max(result.H_mag)) if len(result.H_mag) else None
        peak_idx = int(np.argmax(result.H_mag)) if len(result.H_mag) else -1
        peak_f   = float(result.frequencies[peak_idx]) if peak_idx >= 0 else None
        return [{
            **ctx,
            "f_start":    result.f_start,
            "f_end":      result.f_end,
            "amplitude":  result.amplitude,
            "dc_offset":  result.dc_offset,
            "chirp_type": result.chirp_type,
            "peak_magnitude": peak_mag,
            "peak_frequency": peak_f,
        }]

    if isinstance(result, ParameterFamilyResult):
        rows: list[dict[str, Any]] = []
        for val, sub in zip(result.param_values, result.sub_results):
            sub_ctx = {**ctx, result.sweep_param: val}
            rows.extend(_result_to_rows(test_name, sub, {k: v for k, v in sub_ctx.items() if k != "test"}))
        return rows

    if isinstance(result, ParameterGridResult):
        rows = []
        for combo, sub in zip(result.combinations, result.sub_results):
            sub_ctx = {**{k: v for k, v in ctx.items() if k != "test"}, **combo}
            rows.extend(_result_to_rows(test_name, sub, sub_ctx))
        return rows

    if isinstance(result, DOESweepResult):
        rows = []
        for combo, sub in zip(result.combinations, result.sub_results):
            sub_ctx = {**{k: v for k, v in ctx.items() if k != "test"}, **combo}
            rows.extend(_result_to_rows(test_name, sub, sub_ctx))
        return rows

    return []


# ---------------------------------------------------------------------------
# JSON → typed result reconstruction
# ---------------------------------------------------------------------------

def _dict_to_result(d: dict[str, Any]) -> Any:
    """Reconstruct a typed result object from a to_dict() output."""
    t = d.get("type")

    if t == "frf":
        return FRFResult(
            name          = d["name"],
            frequencies   = np.array(d["frequencies"]),
            magnitudes    = np.array(d["magnitudes"]),
            phases_deg    = np.array(d["phases_deg"]),
            f0            = d.get("f0"),
            Q             = d.get("Q"),
            damping_ratio = d.get("damping_ratio"),
            amplitude     = d.get("amplitude", 0.0),
            dc_offset     = d.get("dc_offset", 0.0),
        )

    if t == "dc_swept_frf":
        measurements = [
            OperatingPointMeasurement(
                dc_value  = m["dc_value"],
                magnitude = m["magnitude"],
                phase_deg = m["phase_deg"],
                f0        = m.get("f0"),
                Q         = m.get("Q"),
            )
            for m in d.get("measurements", [])
        ]
        return DCSweptFRFResult(
            name            = d["name"],
            probe_frequency = d["probe_frequency"],
            measurements    = measurements,
        )

    if t == "amplitude_sweep":
        return AmplitudeSweepResult(
            name                = d["name"],
            frequency           = d["frequency"],
            drive_amplitudes    = np.array(d["drive_amplitudes"]),
            response_amplitudes = np.array(d["response_amplitudes"]),
            phases_deg          = np.array(d["phases_deg"]),
            H_magnitudes        = np.array(d["H_magnitudes"]),
            dc_offset           = d.get("dc_offset", 0.0),
        )

    if t == "chirp":
        return ChirpResult(
            name        = d["name"],
            t           = np.array(d["t"]),
            output      = np.array(d["output"]),
            frequencies = np.array(d["frequencies"]),
            H_mag       = np.array(d["H_mag"]),
            H_phase_deg = np.array(d["H_phase_deg"]),
            f_start     = d["f_start"],
            f_end       = d["f_end"],
            amplitude   = d["amplitude"],
            dc_offset   = d.get("dc_offset", 0.0),
            chirp_type  = d.get("chirp_type", "log"),
        )

    if t == "parameter_family":
        return ParameterFamilyResult(
            name         = d["name"],
            sweep_param  = d["sweep_param"],
            param_values = d["param_values"],
            sub_results  = [_dict_to_result(r) for r in d.get("sub_results", [])],
        )

    if t == "parameter_grid":
        return ParameterGridResult(
            name         = d["name"],
            param_keys   = d["param_keys"],
            combinations = d["combinations"],
            mode         = d.get("mode", "full_factorial"),
            sub_results  = [_dict_to_result(r) for r in d.get("sub_results", [])],
        )

    if t == "doe_sweep":
        return DOESweepResult(
            name         = d["name"],
            design       = d["design"],
            param_keys   = d["param_keys"],
            combinations = d["combinations"],
            sub_results  = [_dict_to_result(r) for r in d.get("sub_results", [])],
        )

    # Unknown type — return the raw dict so callers can still iterate
    return d


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

    @classmethod
    def load(cls, path: str | Path) -> "CampaignResults":
        """Load a saved JSON file back into typed result objects."""
        with open(path) as f:
            data = json.load(f)
        obj = cls(config_version=data.get("version", "1.0"))
        for d in data.get("results", []):
            obj.append(d["name"], _dict_to_result(d))
        return obj

    def to_dataframe(self) -> "Any":
        """Flatten all results to a pandas DataFrame.

        Each row is one scalar measurement.  Composite results (parameter sweeps,
        DOE, grids) produce multiple rows — one per design point — with parameter
        values as additional columns.

        Returns a pandas DataFrame.  Requires pandas to be installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_dataframe(): pip install pandas") from e

        rows: list[dict[str, Any]] = []
        for entry in self._entries:
            rows.extend(_result_to_rows(entry["test"], entry["result"]))
        return pd.DataFrame(rows)
