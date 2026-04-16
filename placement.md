# Placement Rules — How to Write and Modify

This file explains how to configure, add, and modify component placement actions
in `backends/placement.py`.

---

## Project JSON File Location

Each project lives in its own subdirectory:

```
data/
├── EE_PCB_demo0420/
│   ├── EE_PCB_demo0420.json   ← geometry (array format)
│   └── EE_PCB_demo0420.vti    ← simulation field data (optional)
├── Machu14/
│   └── Machu14.json
└── <new_project>/
    ├── <new_project>.json
    └── <new_project>.vti
```

To add a new project:
1. Create `data/<project_name>/` directory
2. Add `<project_name>.json` in array format (see below)
3. Optionally add `<project_name>.vti` for thermal simulation
4. Register the project in `data/projects.json` under `"projects"` so it appears in the Project Manager sidebar

---

## Project JSON Format

Each `.json` is an **array** of component objects:

```json
[
  {"id": "cpu",         "x": 35,  "y": 27, "w": 20, "h": 16, "rotated": 0, "mapping": 0, "level": 0},
  {"id": "heatsink",    "x": 25,  "y": 17, "w": 40, "h": 36, "rotated": 0, "mapping": 0, "level": 0},
  {"id": "fan",         "x": 75,  "y": 48, "w": 30, "h": 24, "rotated": 0, "mapping": 0, "level": 0},
  {"id": "edge_left",   "x": 0,   "y": 0,  "w": 0,  "h": 100, "rotated": 0, "mapping": 0, "level": 0},
  {"id": "edge_right",  "x": 150, "y": 0,  "w": 0,  "h": 100, "rotated": 0, "mapping": 0, "level": 0},
  {"id": "edge_top",    "x": 0,   "y": 100,"w": 150,"h": 0,   "rotated": 0, "mapping": 0, "level": 0},
  {"id": "edge_bottom", "x": 0,   "y": 0,  "w": 150,"h": 0,   "rotated": 0, "mapping": 0, "level": 0}
]
```

| Field | Type | Description |
|---|---|---|
| `id` | str | Component name — used as lookup key (lowercased internally) and display label |
| `x`, `y` | float | **Top-left corner** of the bounding box in mm |
| `w`, `h` | float | Width and height in mm |
| `rotated` | int | Rotation bit 0 (see below) |
| `mapping` | int | Rotation bit 1 (see below) |
| `level` | int | Layer level — currently ignored by the app |

**Edge markers** (`id` starting with `edge_`) are not rendered as components.
They define the board canvas size:
- `edge_right.x` → board width
- `edge_top.y` → board height

---

## Rotation Encoding

```
angle (°CCW) = (rotated + mapping * 2) * 90

rotated  mapping  angle
   0        0       0°   (no rotation)
   1        0      90°   (swap w/h)
   0        1     180°   (no swap)
   1        1     270°   (swap w/h)
```

For axis-aligned bounding boxes only 90° and 270° matter — they swap `w` and `h`.
The conversion happens automatically in `_parse_geometry_array()` (app.py).

---

## Coordinate System

```
(0,0) ──────────────► X  (right = +X)
  │
  │   PCB board  W × H mm  (defined by edge markers in .json)
  │
  ▼
  Y  (down = +Y)
```

**JSON file** stores `x`, `y` as **top-left corner** of each component.  
**Internal dict** (used by placement functions and renderer) stores `x`, `y` as **centre**.  
Conversion happens automatically in `_parse_geometry_array()` on load.

There is also a metadata key `"_board": {"w": W, "h": H}` added to the internal
positions dict — the renderer uses it for the board outline and filters it out
before drawing components.

---

## Primitives

Operate directly on a single component dict:

```python
from backends.placement import move_right, move_left, move_up, move_down, rotate

comp = positions["fan"]

move_right(comp, 5.0)   # fan X += 5 mm
move_left(comp, 3.0)    # fan X -= 3 mm
move_down(comp, 2.0)    # fan Y += 2 mm
move_up(comp, 1.0)      # fan Y -= 1 mm
rotate(comp)            # toggle rotated flag
```

All primitives default to `delta=1.0` mm if omitted.

---

## Moving by Name (recommended)

Use `move_component` when you have the full positions dict and only know the
component name as a string:

```python
from backends.placement import move_component

move_component(positions, "fan",      "right", 5.0)
move_component(positions, "cpu",      "left",  2.0)
move_component(positions, "heatsink", "up",    3.0)
```

Returns `True` if the component was found, `False` if the name did not exist
(no exception — safe to call speculatively).

---

## Adding a New Scripted Action

A scripted action is a plain function that takes `positions: dict` and calls
primitives. Add it at the bottom of the `# High-level scripted actions` section:

```python
def apply_my_new_rule(positions: dict) -> dict:
    """Describe what this rule does and why."""
    move_component(positions, "cpu",      "right", 5.0)
    move_component(positions, "heatsink", "down",  3.0)
    comp = positions.get("fan")
    if comp is not None:
        comp["rotated"] = True
    return positions
```

Then wire the trigger in **two places**:

1. **`chat_responses.py`** — add a regex keyword that returns the action string:

   ```python
   if _match(txt, r"apply my new rule", r"apply.*new.*rule"):
       return _MY_NEW_RULE_RESPONSE, "my_new_action"
   ```

2. **`app.py` `_handle_chat()`** — add the elif branch:

   ```python
   elif action == "my_new_action" and project:
       from backends.placement import apply_my_new_rule
       apply_my_new_rule(_get_positions(project))
       st.session_state["sim_data"] = None
   ```

---

## Sequential Move

Use `apply_sequence` to execute a list of steps in order — useful when one chat
trigger should animate multiple component moves:

```python
from backends.placement import apply_sequence

steps = [
    {"component": "fan",      "direction": "right", "delta": 5.0},
    {"component": "cpu",      "direction": "left",  "delta": 2.0},
    {"component": "heatsink", "direction": "up",    "delta": 3.0},
]
apply_sequence(positions, steps)
```

Each step dict keys:

| Key | Required | Description |
|---|---|---|
| `component` | yes | Component name (case-insensitive) |
| `direction` | yes | `"right"` \| `"left"` \| `"up"` \| `"down"` |
| `delta` | no | Distance in mm (default `1.0`) |

To turn a sequence into a reusable scripted action:

```python
_MY_STEPS = [
    {"component": "fan",  "direction": "right", "delta": 5.0},
    {"component": "cpu",  "direction": "left",  "delta": 2.0},
]

def apply_my_sequence(positions: dict) -> dict:
    return apply_sequence(positions, _MY_STEPS)
```

---

## Existing Scripted Actions

| Function | Trigger phrase (chat) | What it does |
|---|---|---|
| `apply_placement` | `Apply optimized placement` | CPU −2 mm left, heatsink rotated |
| `apply_layout` | `Apply layout adjustment` | Fan +5 mm right |

To change the distance or direction of an existing action, edit the
`move_component(...)` call inside that function in `backends/placement.py`.

---

## Changing Default Positions

Starting positions live in `data/<project>/<project>.json` as **top-left coords**:

```json
[
  {"id": "cpu",      "x": 35, "y": 27, "w": 20, "h": 16, "rotated": 0, "mapping": 0, "level": 0},
  {"id": "heatsink", "x": 25, "y": 17, "w": 40, "h": 36, "rotated": 0, "mapping": 0, "level": 0},
  {"id": "fan",      "x": 75, "y": 48, "w": 30, "h": 24, "rotated": 0, "mapping": 0, "level": 0}
]
```

Edit `x`/`y` here to move where components start before any chat action runs.
Deltas from scripted actions are applied on top of these values at runtime.
