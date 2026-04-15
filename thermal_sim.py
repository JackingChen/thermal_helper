"""
thermal_sim.py
──────────────
Generates a 2-D numpy temperature field used for the thermal heatmap
visualisation in the Design Workspace.

Usage
─────
    from thermal_sim import run_simulation
    temp_array = run_simulation("EE_PCB_demo0420")   # shape (100, 150)
"""
from __future__ import annotations

import numpy as np


# ── Component geometry (row, col, half-height, half-width, peak_temp) ─────────
_COMPONENTS: dict[str, dict] = {
    "EE_PCB_demo0420": {
        "cpu":      {"r": 35, "c": 45, "hr": 8,  "hc": 10, "T": 89.3},
        "heatsink": {"r": 35, "c": 45, "hr": 18, "hc": 20, "T": 72.0},
        "fan":      {"r": 60, "c": 90, "hr": 12, "hc": 15, "T": 55.0},
        "ambient":  25.0,
    },
    "ME-Placement-1041": {
        "cpu":      {"r": 30, "c": 50, "hr": 8,  "hc": 10, "T": 91.0},
        "heatsink": {"r": 30, "c": 50, "hr": 20, "hc": 22, "T": 74.0},
        "fan":      {"r": 65, "c": 100, "hr": 12, "hc": 16, "T": 57.0},
        "ambient":  25.0,
    },
}

_DEFAULT_KEY = "EE_PCB_demo0420"

# Grid shape: height=100, width=150 (rows × cols)
_H, _W = 100, 150


def _gaussian(rows: np.ndarray, cols: np.ndarray,
              r0: float, c0: float,
              sr: float, sc: float,
              amplitude: float) -> np.ndarray:
    """Return a 2-D Gaussian centred on (r0, c0)."""
    return amplitude * np.exp(
        -((rows - r0) ** 2) / (2 * sr ** 2)
        - ((cols - c0) ** 2) / (2 * sc ** 2)
    )


def run_simulation(project_name: str = _DEFAULT_KEY) -> np.ndarray:
    """
    Generate a 2-D temperature field (°C) for the given project.

    Parameters
    ----------
    project_name : str
        Key into internal component geometry table.

    Returns
    -------
    np.ndarray, shape (100, 150), dtype float32
        Temperature array, values roughly 25 – 90 °C.
    """
    cfg = _COMPONENTS.get(project_name, _COMPONENTS[_DEFAULT_KEY])
    ambient: float = cfg["ambient"]

    rows_idx, cols_idx = np.mgrid[0:_H, 0:_W].astype(np.float32)
    field = np.full((_H, _W), ambient, dtype=np.float32)

    for name, comp in cfg.items():
        if name == "ambient":
            continue
        delta = float(comp["T"]) - ambient
        # spread  ≈ 2× the half-extent so heat bleeds naturally
        sr = comp["hr"] * 2.0
        sc = comp["hc"] * 2.0
        field += _gaussian(rows_idx, cols_idx,
                           comp["r"], comp["c"],
                           sr, sc, delta).astype(np.float32)

    # Clip to realistic range and add tiny noise for realism
    rng = np.random.default_rng(42)
    field += rng.normal(0, 0.3, field.shape).astype(np.float32)
    field = np.clip(field, 20.0, 120.0)
    return field


def get_component_positions(project_name: str = _DEFAULT_KEY) -> dict:
    """
    Return pixel / grid positions of components for overlay annotations.

    Returns
    -------
    dict: {component_name: {"x": col, "y": row, "T": peak_temp_°C}}
    """
    cfg = _COMPONENTS.get(project_name, _COMPONENTS[_DEFAULT_KEY])
    result = {}
    for name, comp in cfg.items():
        if name == "ambient":
            continue
        result[name] = {
            "x": comp["c"],
            "y": comp["r"],
            "T": comp["T"],
        }
    return result
