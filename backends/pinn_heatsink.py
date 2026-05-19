"""
backends/pinn_heatsink.py
─────────────────────────
Loads the pre-computed PINN inference result (copper model, XY plane at
z_idx=28) and exposes it for heatsink temperature calculations.

The .npy file is generated once by:
    backends/Thermal_inference/extract_z28.py
and stored at:
    assets/pinn_heatsink_z28.npy

Array layout: T_C[ix, iy]
  ix  — along PINN x (flow direction through fins)
  iy  — along PINN y (fin height, vertical)

Board mapping convention used by thermal_sim.py:
  PINN x (flow direction) → board Y-axis of heatsink (long dimension)
  PINN y (fin height)     → averaged out (vertical, not in top-down view)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_FIELD_PATH = Path(__file__).parent.parent / "assets" / "pinn_heatsink_z28.npy"

# Module-level cache
_T_FIELD: np.ndarray | None = None   # shape (nx, ny)  T in °C
_XS_ND: np.ndarray | None = None     # (nx,) non-dim x values (flow direction)
_YS_ND: np.ndarray | None = None     # (ny,) non-dim y values (fin height)
_Z_ND: float | None = None
_LOADED: bool = False

_FALLBACK_TEMP = 27.0   # °C — used when .npy file is unavailable


def _load() -> None:
    global _T_FIELD, _XS_ND, _YS_ND, _Z_ND, _LOADED
    if _LOADED:
        return
    _LOADED = True
    if not _FIELD_PATH.exists():
        return
    try:
        data = np.load(_FIELD_PATH, allow_pickle=True).item()
        _T_FIELD = np.asarray(data["T_C"], dtype=np.float32)   # (nx, ny)
        _XS_ND   = np.asarray(data["xs_nd"], dtype=np.float32)
        _YS_ND   = np.asarray(data["ys_nd"], dtype=np.float32)
        _Z_ND    = float(data["z_nd"])
    except Exception:
        _T_FIELD = None


def is_available() -> bool:
    """Return True if the pre-computed PINN field was loaded successfully."""
    _load()
    return _T_FIELD is not None


def get_mean_temp() -> float:
    """Mean solid temperature (°C) across the entire XY slice."""
    _load()
    if _T_FIELD is None:
        return _FALLBACK_TEMP
    return float(_T_FIELD.mean())


def get_flow_profile() -> np.ndarray:
    """
    Return the temperature profile along the flow direction (°C).

    Shape: (nx,) — each value is the average over fin height (PINN y) at
    that flow-direction position.  ix=0 is the heatsink inlet, ix=-1 outlet.
    """
    _load()
    if _T_FIELD is None:
        return np.array([_FALLBACK_TEMP], dtype=np.float32)
    return _T_FIELD.mean(axis=1).astype(np.float32)   # (nx,)


def get_full_field() -> np.ndarray:
    """
    Return the full (nx, ny) temperature array in °C.
    ix: flow direction, iy: fin height.
    """
    _load()
    if _T_FIELD is None:
        return np.full((1, 1), _FALLBACK_TEMP, dtype=np.float32)
    return _T_FIELD
