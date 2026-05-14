"""
thermal_sim.py
──────────────
Generates a 2-D numpy temperature field used for the thermal heatmap
visualisation in the Design Workspace.

Physics model (live_positions path)
────────────────────────────────────
Each component's effective heat-rise is calculated from first principles:

    ΔT_eff = ΔT_base / (k_material × B_heatsink)

where:
  ΔT_base     – worst-case heat-rise for this component type when made of plain
                 metal (steel, k≈50 W/m·K).  Derived from  P × R_junction.
  k_material  – relative thermal conductivity vs. steel baseline.
                 copper(7.8) > graphite(2.4) > aluminum(3.9) > metal(1.0)
                 > ceramic(0.4) > plastic(0.08)
  B_heatsink  – proximity-weighted cooling bonus (≥ 1.0) from any Fin /
                 heatsink component present in the layout:
                    B += 0.8 × (area_mm² / 800) / (1 + (dist_mm / 25)²)
                 Larger fins and shorter distances give a stronger bonus.

This means:
  • Changing a component's material field immediately changes its temperature.
  • Placing or moving a heatsink automatically cools nearby components.
  • No preset lookups or global scaling factors are needed.

Usage
─────
    from thermal_sim import run_simulation, get_component_positions

    # Legacy (hardcoded geometry):
    temp_array = run_simulation("EE_PCB_demo0420")

    # Live (from session state positions — physics-driven):
    temp_array = run_simulation(
        "EE_PCB_demo0420",
        live_positions=positions,   # session_state["component_positions"][project]
        board_w=150, board_h=100,
    )
"""
from __future__ import annotations

import math

import numpy as np


# ── Physics tables ────────────────────────────────────────────────────────────

# Relative thermal conductivity vs. steel (k_steel ≈ 50 W/m·K).
# Higher value → better heat spreading → lower component temperature.
# "ceramic" here is thermally conductive ceramic (e.g. AlN, ~180 W/m·K),
# used as an electrically isolating but highly conductive substrate.
_MATERIAL_CONDUCTIVITY: dict[str, float] = {
    "metal":    1.00,   # steel / generic metal  ~50 W/m·K
    "aluminum": 3.90,   # ~200 W/m·K
    "copper":   7.80,   # ~400 W/m·K
    "graphite": 2.40,   # ~120 W/m·K (in-plane)
    "ceramic":  3.60,   # AlN / BeO ceramic  ~180 W/m·K (conductive substrate)
    "plastic":  0.08,   # ~4 W/m·K   (poor conductor)
}

# Base heat-rise (ΔT °C above ambient) for each component type when its
# material is plain steel/metal.  Represents P × R_junction at worst case.
_COMPONENT_BASE_DELTA: dict[str, float] = {
    "cpu":      55.0,   # 25 + 55 = 80 °C
    "heatsink": 30.0,   # 25 + 30 = 55 °C
    "fan":      10.0,   # 25 + 10 = 35 °C
    "wlan":     25.0,   # 25 + 25 = 50 °C
    "power":    20.0,   # 25 + 20 = 45 °C
    "vrm":      40.0,   # 25 + 40 = 65 °C
    "ssd":      25.0,   # 25 + 25 = 50 °C
    "ddr":      20.0,   # 25 + 20 = 45 °C
    "pcie":     15.0,   # 25 + 15 = 40 °C
    "io":        5.0,   # 25 +  5 = 30 °C
    "hinge":     2.0,   # 25 +  2 = 27 °C  (structural)
    "fin":       0.0,   # passive — no self-heating
}
_DEFAULT_BASE_DELTA: float = 15.0   # 25 + 15 = 40 °C for unrecognised types

# Component names containing these substrings are treated as passive
# heatsinks / spreaders: they add no heat themselves but cool neighbours.
_HEATSINK_KEYWORDS: tuple[str, ...] = ("fin", "heatsink", "spreader", "cooler", "sink")



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


# ── Physics helpers ───────────────────────────────────────────────────────────

def _is_heatsink(name: str) -> bool:
    """Return True if this component acts as a passive heatsink / spreader."""
    n = name.lower()
    return any(kw in n for kw in _HEATSINK_KEYWORDS)


def _base_delta(name: str) -> float:
    """Return worst-case heat-rise (°C, metal baseline) for a component type."""
    n = name.lower()
    for key, val in _COMPONENT_BASE_DELTA.items():
        if key in n:
            return val
    return _DEFAULT_BASE_DELTA


def _conductivity(material: str) -> float:
    """Return relative thermal conductivity for the given material string."""
    return _MATERIAL_CONDUCTIVITY.get(str(material or "metal").strip().lower(), 1.0)


def _heatsink_bonus(cx: float, cy: float, positions: dict) -> float:
    """
    Proximity-weighted cooling bonus (≥ 1.0) from nearby Fin/heatsink components.

    For each heatsink in *positions*:
        contribution = 0.8 × (area_mm² / 800) / (1 + (dist_mm / 25)²)

    Larger fins and shorter distances give a stronger bonus, reducing the
    effective thermal resistance of the target component.
    """
    bonus = 1.0
    for name, other in positions.items():
        if name == "_board" or not _is_heatsink(name):
            continue
        ox = float(other.get("x", 0)) + float(other.get("w", 0)) / 2
        oy = float(other.get("y", 0)) + float(other.get("h", 0)) / 2
        dist_mm = math.hypot(cx - ox, cy - oy) + 1e-3   # avoid division by zero
        area_mm2 = float(other.get("w", 10)) * float(other.get("h", 10))
        bonus += 0.8 * (area_mm2 / 800.0) / (1.0 + (dist_mm / 25.0) ** 2)
    return bonus


def _effective_delta(name: str, comp: dict, positions: dict) -> float:
    """
    Compute effective heat-rise (°C) for a component.

        ΔT_eff = ΔT_base / (k_material × B_heatsink)

    Parameters
    ----------
    name      : component name (used to identify type and heatsink role)
    comp      : position dict with keys x, y, w, h, material
    positions : full live_positions dict (used for heatsink proximity)
    """
    if _is_heatsink(name):
        return 0.0   # fin / spreader — no self-heating
    base = _base_delta(name)
    cond = _conductivity(comp.get("material", "metal"))
    cx = float(comp.get("x", 0)) + float(comp.get("w", 0)) / 2
    cy = float(comp.get("y", 0)) + float(comp.get("h", 0)) / 2
    bonus = _heatsink_bonus(cx, cy, positions)
    return base / (cond * bonus)




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
    component_temps: dict | None = None,   # kept for API compat; ignored in physics path
    board_w: float = 400.0,
    board_h: float = 400.0,
) -> np.ndarray:
    """
    Generate a 2-D temperature field (°C) for the given project.

    When *live_positions* is provided, temperatures are derived from physics
    (material conductivity + heatsink proximity) — *component_temps* is ignored.
    Otherwise falls back to the hardcoded table (legacy projects).

    Parameters
    ----------
    project_name : str
        Key into the fallback internal table (used when live_positions is None).
    live_positions : dict | None
        Session-state positions dict {comp_name: {x, y, w, h, material, ...}}.
        The special ``_board`` key is ignored.
    component_temps : dict | None
        Legacy parameter — only used by the hardcoded fallback path.
    board_w, board_h : float
        Board dimensions in mm — used to map mm coordinates onto the grid.

    Returns
    -------
    np.ndarray, shape (100, 150), dtype float32
    """
    rows_idx, cols_idx = np.mgrid[0:_H, 0:_W].astype(np.float32)
    field = np.full((_H, _W), _AMBIENT, dtype=np.float32)

    if live_positions is not None:
        for name, comp in live_positions.items():
            if name == "_board":
                continue
            w = comp.get("h", 0) if comp.get("rotated") else comp.get("w", 0)
            h = comp.get("w", 0) if comp.get("rotated") else comp.get("h", 0)
            if w <= 0 or h <= 0:
                continue
            # Convert mm → grid indices (top-left origin)
            col = float(comp["x"]) * (_W / board_w)
            row = float(comp["y"]) * (_H / board_h)
            hc  = (w / 2) * (_W / board_w)
            hr  = (h / 2) * (_H / board_h)
            # Physics-derived heat-rise — driven by material + heatsink proximity
            delta = _effective_delta(name, comp, live_positions)
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
    component_temps: dict | None = None,   # kept for API compat; ignored in physics path
    board_w: float = 400.0,
    board_h: float = 400.0,
) -> dict:
    """
    Return positions + temperatures of components for overlay annotations.

    In the live path, temperatures are physics-derived (same model as
    run_simulation).  In the legacy path, returns grid coordinates with
    hardcoded temperatures.

    Returns
    -------
    dict: {component_name: {"x": float, "y": float, "T": float}}
    """
    if live_positions is not None:
        result = {}
        for name, comp in live_positions.items():
            if name == "_board":
                continue
            delta = _effective_delta(name, comp, live_positions)
            result[name] = {
                "x": float(comp["x"]),
                "y": float(comp["y"]),
                "T": round(_AMBIENT + delta, 1),
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


