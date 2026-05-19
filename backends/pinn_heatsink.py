"""
backends/pinn_heatsink.py
─────────────────────────
Loads pre-computed PINN inference fields for both copper and aluminum
heatsink models and exposes them for heatsink temperature calculations.

Copper (default):  assets/pinn_heatsink_z28.npy
  - NV_heatsink_copper model, z_idx=28 (hottest XY plane)
  - Domain x∈(0.15,1.15), z∈(0.0,5.70)

Aluminum (optimized): assets/pinn_heatsink_aluminum_z0.npy
  - NV_heatsink_aluminum model, z_idx=0 (hottest XY plane at z=-0.98)
  - Domain x∈(-0.70,0.70), z∈(-0.98,0.77)

Array layout: T_C[ix, iy]
  ix  — along PINN x (flow direction through fins)
  iy  — along PINN y (fin height, vertical)

Board mapping convention used by thermal_sim.py:
  PINN x (flow direction) → board Y-axis of heatsink (long dimension)
  PINN y (fin height)     → averaged out (vertical, not in top-down view)

Material selection:
  Call get_flow_profile(material="copper") or get_flow_profile(material="aluminum").
  thermal_sim.py detects material from live_positions["cu-base"]["material"].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_ASSETS = Path(__file__).parent.parent / "assets"
_FIELD_PATH_CU = _ASSETS / "pinn_heatsink_z28.npy"
_FIELD_PATH_AL = _ASSETS / "pinn_heatsink_aluminum_z0.npy"

_FALLBACK_TEMP = 27.0   # °C — used when .npy file is unavailable


# ── Per-material field cache ───────────────────────────────────────────────────

class _FieldCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._loaded = False
        self._T: np.ndarray | None = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            data = np.load(self.path, allow_pickle=True).item()
            self._T = np.asarray(data["T_C"], dtype=np.float32)  # (nx, ny)
        except Exception:
            self._T = None

    def available(self) -> bool:
        self._load()
        return self._T is not None

    def mean_temp(self) -> float:
        self._load()
        return float(self._T.mean()) if self._T is not None else _FALLBACK_TEMP

    def flow_profile(self) -> np.ndarray:
        """Temperature averaged over fin height at each flow-direction position. Shape (nx,)."""
        self._load()
        if self._T is None:
            return np.array([_FALLBACK_TEMP], dtype=np.float32)
        return self._T.mean(axis=1).astype(np.float32)

    def full_field(self) -> np.ndarray:
        """Full (nx, ny) temperature array in °C."""
        self._load()
        if self._T is None:
            return np.full((1, 1), _FALLBACK_TEMP, dtype=np.float32)
        return self._T


_CU_CACHE = _FieldCache(_FIELD_PATH_CU)
_AL_CACHE = _FieldCache(_FIELD_PATH_AL)


def _cache_for(material: str) -> _FieldCache:
    return _AL_CACHE if material.lower() == "aluminum" else _CU_CACHE


# ── Public API ─────────────────────────────────────────────────────────────────

def is_available(material: str = "copper") -> bool:
    """Return True if the PINN field for the given material is loaded."""
    return _cache_for(material).available()


def get_mean_temp(material: str = "copper") -> float:
    """Mean solid temperature (°C) across the XY slice for the given material."""
    return _cache_for(material).mean_temp()


def get_flow_profile(material: str = "copper") -> np.ndarray:
    """
    Temperature profile along the flow direction for the given material.

    Shape: (nx,) — averaged over fin height (PINN y).
    ix=0 is the heatsink inlet side, ix=-1 is the outlet side.

    material: "copper" (default) or "aluminum"
    """
    return _cache_for(material).flow_profile()


def get_full_field(material: str = "copper") -> np.ndarray:
    """Full (nx, ny) temperature array in °C for the given material."""
    return _cache_for(material).full_field()


# ── Backward-compatible aliases (copper only) ─────────────────────────────────

def _load() -> None:
    _CU_CACHE._load()


def is_available_copper() -> bool:
    return _CU_CACHE.available()


def is_available_aluminum() -> bool:
    return _AL_CACHE.available()
