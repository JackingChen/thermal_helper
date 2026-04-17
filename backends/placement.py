"""
backends/placement.py
─────────────────────
Primitive and high-level component placement operations.

Instruction schema (for LLM / JSON-driven moves)
─────────────────────────────────────────────────
A placement instruction is a plain dict:

    {
        "action": "move_sequence",          # required — see ACTIONS below
        "steps": [                          # for move_sequence
            {
                "component": "fan",         # component name (case-insensitive)
                "direction": "right",       # "right" | "left" | "up" | "down"
                "delta": 5.0               # mm (default 1.0 if omitted)
            },
            {
                "component": "heatsink",
                "rotate": true              # set rotated flag instead of move
            }
        ]
    }

Supported actions
─────────────────
    "move_sequence"   — execute steps list (default for LLM responses)
    "apply_placement" — run the built-in optimised-placement preset
    "apply_layout"    — run the built-in layout-adjustment preset

Entry points
────────────
    execute_instruction(positions, instruction)
        Accept a dict (e.g. parsed LLM JSON) and apply it.

    apply_placement(positions)   apply_layout(positions)
        Convenience wrappers around execute_instruction with built-in presets.

    apply_sequence(positions, steps)
        Low-level: run a steps list directly.

Primitives
──────────
    move_right / move_left / move_up / move_down (comp, delta)
    rotate(comp)
    move_component(positions, name, direction, delta)
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


# ── Sequential move helper ─────────────────────────────────────────────────────

def apply_sequence(positions: dict, steps: list[dict]) -> dict:
    """
    Execute a list of step dicts in order.

    Each step:
        {"component": str, "direction": str, "delta": float}   — move
        {"component": str, "rotate": true}                      — toggle rotation
    """
    for step in steps:
        name = step["component"]
        if step.get("rotate"):
            comp = positions.get(name.lower())
            if comp is not None:
                comp["rotated"] = True
        else:
            move_component(
                positions,
                name=name,
                direction=step["direction"],
                delta=float(step.get("delta", 1.0)),
            )
    return positions


# ── Built-in preset data ───────────────────────────────────────────────────────
# Stored as data so LLM can produce the same format to override them.

_PRESET_PLACEMENT: dict = {
    "action": "move_sequence",
    "label": "Optimised placement (Step 4)",
    "steps": [
        {"component": "cpu",      "direction": "left", "delta": 2.0},
        {"component": "heatsink", "rotate": True},
    ],
}

_PRESET_LAYOUT: dict = {
    "action": "move_sequence",
    "label": "Layout adjustment (Step 10)",
    "steps": [
        {"component": "cpu",      "direction": "right", "delta": 10.0},
        {"component": "fan",      "direction": "down",  "delta": 12.0},
        {"component": "heatsink", "direction": "up",    "delta": 6.0},
    ],
}

_PRESETS: dict[str, dict] = {
    "apply_placement": _PRESET_PLACEMENT,
    "apply_layout":    _PRESET_LAYOUT,
}


# ── Main entry point ───────────────────────────────────────────────────────────

def execute_instruction(positions: dict, instruction: dict) -> dict:
    """
    Apply a placement instruction dict to *positions*.

    The *instruction* dict is the canonical format for LLM responses:

        {
            "action": "move_sequence",
            "steps": [
                {"component": "fan", "direction": "right", "delta": 5.0},
                {"component": "heatsink", "rotate": true}
            ]
        }

    Named preset shortcuts are also accepted:
        {"action": "apply_placement"}
        {"action": "apply_layout"}

    Returns the mutated positions dict.
    """
    action = instruction.get("action", "move_sequence")

    # Resolve named presets to their step lists
    if action in _PRESETS:
        instruction = _PRESETS[action]
        action = "move_sequence"

    if action == "move_sequence":
        steps = instruction.get("steps", [])
        apply_sequence(positions, steps)
    else:
        raise ValueError(f"Unknown placement action '{action}'.")

    return positions


# ── Convenience wrappers ───────────────────────────────────────────────────────

def apply_placement(positions: dict) -> dict:
    """Run the built-in optimised-placement preset."""
    return execute_instruction(positions, {"action": "apply_placement"})


def apply_layout(positions: dict) -> dict:
    """Run the built-in layout-adjustment preset."""
    return execute_instruction(positions, {"action": "apply_layout"})


# ── Constraint checking ────────────────────────────────────────────────────────

def check_overlaps(positions: dict) -> list[tuple[str, str]]:
    """
    Return list of (name_a, name_b) component pairs whose bounding boxes overlap.

    Ignores the special ``_board`` metadata key and zero-size components.
    """
    comps = [
        (k, v) for k, v in positions.items()
        if k != "_board" and v.get("w", 0) > 0 and v.get("h", 0) > 0
    ]
    pairs: list[tuple[str, str]] = []
    for i, (na, a) in enumerate(comps):
        wa = a["h"] if a.get("rotated") else a["w"]
        ha = a["w"] if a.get("rotated") else a["h"]
        ax0, ax1 = a["x"] - wa / 2, a["x"] + wa / 2
        ay0, ay1 = a["y"] - ha / 2, a["y"] + ha / 2
        for nb, b in comps[i + 1:]:
            wb = b["h"] if b.get("rotated") else b["w"]
            hb = b["w"] if b.get("rotated") else b["h"]
            bx0, bx1 = b["x"] - wb / 2, b["x"] + wb / 2
            by0, by1 = b["y"] - hb / 2, b["y"] + hb / 2
            if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
                pairs.append((na, nb))
    return pairs

