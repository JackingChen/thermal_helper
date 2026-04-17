"""
thermal_sim.py
──────────────
Generates a 2-D numpy temperature field used for the thermal heatmap
visualisation in the Design Workspace.

Usage
─────
    from thermal_sim import run_simulation, get_component_positions

    # Legacy (hardcoded geometry):
    temp_array = run_simulation("EE_PCB_demo0420")

    # Live (from session state positions):
    temp_array = run_simulation(
        "EE_PCB_demo0420",
        live_positions=positions,        # session_state["component_positions"][project]
        component_temps=COMPONENT_TEMPS, # from backends.thermal
        board_w=150, board_h=100,
    )
"""
from __future__ import annotations

import numpy as np


# ── Legacy hardcoded component geometry ───────────────────────────────────────
# (row, col, half-height, half-width, peak_temp)  – still used as fallback
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

_AMBIENT = 25.0


def _gaussian(rows: np.ndarray, cols: np.ndarray,
              r0: float, c0: float,
              sr: float, sc: float,
              amplitude: float) -> np.ndarray:
    """Return a 2-D Gaussian centred on (r0, c0)."""
    return amplitude * np.exp(
        -((rows - r0) ** 2) / (2 * sr ** 2)
        - ((cols - c0) ** 2) / (2 * sc ** 2)
    )


def run_simulation(
    project_name: str = _DEFAULT_KEY,
    live_positions: dict | None = None,
    component_temps: dict | None = None,
    board_w: float = 150.0,
    board_h: float = 100.0,
) -> np.ndarray:
    """
    Generate a 2-D temperature field (°C) for the given project.

    When *live_positions* is provided the field is built from the current
    component positions (mm, centre-based) and *component_temps* temperatures.
    Otherwise the call falls back to the hardcoded table for backward compat.

    Parameters
    ----------
    project_name : str
        Key into the fallback internal table (used when live_positions is None).
    live_positions : dict | None
        Session-state positions dict {comp_name: {x, y, w, h, rotated, ...}}.
        The special ``_board`` key is ignored.
    component_temps : dict | None
        Mapping of lowercase component name → peak temperature (°C).
        Defaults to 40 °C for any component not listed.
    board_w, board_h : float
        Board dimensions in mm — used to map mm coordinates onto the grid.

    Returns
    -------
    np.ndarray, shape (100, 150), dtype float32
    """
    rows_idx, cols_idx = np.mgrid[0:_H, 0:_W].astype(np.float32)
    field = np.full((_H, _W), _AMBIENT, dtype=np.float32)

    if live_positions is not None:
        temps = component_temps or {}
        for name, comp in live_positions.items():
            if name == "_board":
                continue
            w = comp.get("h", 0) if comp.get("rotated") else comp.get("w", 0)
            h = comp.get("w", 0) if comp.get("rotated") else comp.get("h", 0)
            if w <= 0 or h <= 0:
                continue
            # Convert mm → grid indices
            col = float(comp["x"]) * (_W / board_w)
            row = float(comp["y"]) * (_H / board_h)
            hc  = (w / 2) * (_W / board_w)
            hr  = (h / 2) * (_H / board_h)
            # temperature lookup: exact name, then prefix match
            temp = _match_temp(name, temps)
            delta = temp - _AMBIENT
            # spread = 2× half-extent so heat bleeds naturally outside the part
            field += _gaussian(rows_idx, cols_idx, row, col,
                               hr * 2.0, hc * 2.0, delta).astype(np.float32)
    else:
        cfg = _COMPONENTS.get(project_name, _COMPONENTS[_DEFAULT_KEY])
        for name, comp in cfg.items():
            if name == "ambient":
                continue
            delta = float(comp["T"]) - _AMBIENT
            sr = comp["hr"] * 2.0
            sc = comp["hc"] * 2.0
            field += _gaussian(rows_idx, cols_idx,
                               comp["r"], comp["c"],
                               sr, sc, delta).astype(np.float32)

    rng = np.random.default_rng(42)
    field += rng.normal(0, 0.3, field.shape).astype(np.float32)
    field = np.clip(field, 20.0, 120.0)
    return field


def get_component_positions(
    project_name: str = _DEFAULT_KEY,
    live_positions: dict | None = None,
    component_temps: dict | None = None,
    board_w: float = 150.0,
    board_h: float = 100.0,
) -> dict:
    """
    Return positions + temperatures of components for overlay annotations.

    When *live_positions* is provided, coordinates are in mm (matching the
    ``extent=[0, board_w, board_h, 0]`` used by imshow).
    Otherwise returns legacy pixel/grid indices.

    Returns
    -------
    dict: {component_name: {"x": float, "y": float, "T": float}}
    """
    if live_positions is not None:
        temps = component_temps or {}
        result = {}
        for name, comp in live_positions.items():
            if name == "_board":
                continue
            temp = _match_temp(name, temps)
            result[name] = {
                "x": float(comp["x"]),
                "y": float(comp["y"]),
                "T": temp,
            }
        return result

    # Legacy fallback
    cfg = _COMPONENTS.get(project_name, _COMPONENTS[_DEFAULT_KEY])
    result = {}
    for name, comp in cfg.items():
        if name == "ambient":
            continue
        result[name] = {"x": comp["c"], "y": comp["r"], "T": comp["T"]}
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _match_temp(name: str, temps: dict, default: float = 40.0) -> float:
    """Look up temperature for *name* in *temps*, falling back to prefix match."""
    key = name.lower()
    if key in temps:
        return float(temps[key])
    for k, v in temps.items():
        if k in key or key in k:
            return float(v)
    return default

