"""
backends/placement.py
─────────────────────
Primitive and high-level component placement operations.

Primitives  (operate on a single component dict in-place)
──────────
    move_right(comp, delta)   comp["x"] += delta
    move_left(comp, delta)    comp["x"] -= delta
    move_down(comp, delta)    comp["y"] += delta   (Y increases downward)
    move_up(comp, delta)      comp["y"] -= delta
    rotate(comp)              toggles comp["rotated"]

High-level helpers  (operate on a positions dict keyed by component name)
─────────────────
    move_component(positions, name, direction, delta_mm)
        Dispatch to the correct primitive; returns True if the component existed.

    apply_placement(positions)
        Scripted Step-4 action: CPU –2 mm left, heatsink rotated.

    apply_layout(positions)
        Scripted Step-10 action: fan +5 mm right.

    apply_sequence(positions, steps)
        Execute a list of move instructions sequentially.
        steps = [{"component": "fan", "direction": "right", "delta": 5.0}, ...]

All functions return the mutated positions dict so calls can be chained.
"""
from __future__ import annotations

# ── Primitives ─────────────────────────────────────────────────────────────────

def move_right(comp: dict, delta: float = 1.0) -> dict:
    """Shift component right by *delta* mm (X increases rightward)."""
    comp["x"] = round(comp["x"] + delta, 1)
    return comp


def move_left(comp: dict, delta: float = 1.0) -> dict:
    """Shift component left by *delta* mm."""
    comp["x"] = round(comp["x"] - delta, 1)
    return comp


def move_down(comp: dict, delta: float = 1.0) -> dict:
    """Shift component downward by *delta* mm (Y increases downward)."""
    comp["y"] = round(comp["y"] + delta, 1)
    return comp


def move_up(comp: dict, delta: float = 1.0) -> dict:
    """Shift component upward by *delta* mm."""
    comp["y"] = round(comp["y"] - delta, 1)
    return comp


def rotate(comp: dict) -> dict:
    """Toggle the rotated flag on a component."""
    comp["rotated"] = not comp.get("rotated", False)
    return comp


# ── Direction dispatch ─────────────────────────────────────────────────────────

_DIRECTION_FN = {
    "right": move_right,
    "left":  move_left,
    "down":  move_down,
    "up":    move_up,
}


def move_component(
    positions: dict,
    name: str,
    direction: str,
    delta: float = 1.0,
) -> bool:
    """
    Move *name* in *direction* by *delta* mm.

    Returns True if the component was found and moved, False otherwise.
    *direction* must be one of: "right", "left", "up", "down".
    """
    comp = positions.get(name.lower())
    if comp is None:
        return False
    fn = _DIRECTION_FN.get(direction.lower())
    if fn is None:
        raise ValueError(f"Unknown direction '{direction}'. Use: {list(_DIRECTION_FN)}")
    fn(comp, delta)
    return True


# ── High-level scripted actions ────────────────────────────────────────────────

def apply_placement(positions: dict) -> dict:
    """
    Scripted Step-4 optimised placement:
      • CPU  → –2 mm left
      • Heatsink → rotate 90°
    """
    move_component(positions, "cpu", "left", 2.0)
    comp = positions.get("heatsink")
    if comp is not None:
        comp["rotated"] = True
    return positions


def apply_layout(positions: dict) -> dict:
    """
    Scripted Step-10 layout adjustment:
      • Fan → +5 mm right
    """
    move_component(positions, "cpu", "right", 10.0)
    move_component(positions, "fan", "down", 12.0)
    move_component(positions, "heatsink", "up", 6.0)
    return positions


# ── Sequential move helper ─────────────────────────────────────────────────────

def apply_sequence(positions: dict, steps: list[dict]) -> dict:
    """
    Execute a list of move instructions in order.

    Each step dict:
        {
            "component": str,          # e.g. "fan", "cpu", "heatsink"
            "direction": str,          # "right" | "left" | "up" | "down"
            "delta":     float,        # mm (default 1.0 if omitted)
        }

    Example
    -------
        steps = [
            {"component": "fan",      "direction": "right", "delta": 5.0},
            {"component": "cpu",      "direction": "left",  "delta": 2.0},
            {"component": "heatsink", "direction": "up",    "delta": 3.0},
        ]
        apply_sequence(positions, steps)
    """
    for step in steps:
        move_component(
            positions,
            name=step["component"],
            direction=step["direction"],
            delta=float(step.get("delta", 1.0)),
        )
    return positions
