# CAD T — Thermal Design Assistant

A Streamlit-based live demo simulating an AI-assisted CAE/EE design workflow. The app presents a four-panel UI covering project management, a material library, an interactive design workspace (2D modeling, thermal heatmap, 3D preview), and a scripted AI chat assistant.

---

## Demo Flow

The demo walks through 11 scripted steps across two feature sets:

**Feature 1 — Passive Thermal Strategy (Steps 1–5)**
1. Open web app → 4-panel layout loads
2. Select project `EE_PCB_demo0420` → 2D Modeling view renders
3. Ask about CPU/heatsink spacing → AI returns passive-thermal impact analysis + recommendations
4. Type `Apply passive thermal approach` → workspace applies optimized thermal strategy updates (geometry + material profile)
5. Type `3D` → workspace switches to 3D preview image

**Feature 2 — Live Simulation (Steps 6–11)**
6. Switch mode to Thermal Simulation → 2D heatmap renders (20–120 °C)
7. Ask about chassis surface temperature → AI cites IEC 62368-1 limits from FAQ CSV
8. Ask about fan/antenna interference → classified as RF/EMI, switches knowledge domain
9. AI returns EMI source list + design recommendations from RF expert system
10. Type `Apply layout adjustment` → fan module moves +5 mm, thermal re-simulation triggered
11. Type `3D` → final 3D preview confirmation

---

## Architecture

```
thermal_helper/
├── app.py                  # Main Streamlit app — layout, routing, state
├── chat_responses.py       # All scripted responses + keyword router
├── qa_loader.py            # CSV loader + keyword-match FAQ retriever
├── thermal_sim.py          # NumPy-based 2D temperature field generator
├── backends/
│   ├── __init__.py         # BackendResult dataclass
│   ├── placement.py        # Placement backend stub (future: solver)
│   ├── thermal.py          # Thermal backend stub (future: RC-network solver)
│   ├── llm_backend.py      # Azure OpenAI Responses API client
│   ├── pinn_heatsink.py    # PINN field loader for heatsink project (copper + aluminum)
│   └── Thermal_inference/  # PINN model weights + one-off extraction scripts
│       ├── infer_thermal_solid.py          # FourierNetArch inference helper
│       ├── extract_z28.py                  # Pre-computes copper field → pinn_heatsink_z28.npy
│       ├── extract_aluminum_z0.py          # Pre-computes aluminum field → pinn_heatsink_aluminum_z0.npy
│       └── model/
│           ├── NV_heatsink_copper/         # Copper PINN checkpoint
│           └── NV_heatsink_aluminum/       # Aluminum PINN checkpoint
├── data/
│   ├── projects.json       # Project tree + component metadata + default positions
│   └── (AIC 0408) Ai for design_Thermal Expert System_FAQ v.1.csv
├── assets/
│   ├── ThermalOnPCB.png                    # Static 3D preview image
│   ├── pinn_heatsink_z28.npy               # Pre-computed copper PINN field (64×32)
│   ├── pinn_heatsink_aluminum_z0.npy       # Pre-computed aluminum PINN field (64×32)
│   └── project_rule/
│       ├── Hitatori/                       # Hitatori project rules + preset JSONs
│       └── heatsink/                       # Heatsink project rules + preset JSONs
│           ├── heatsink.md                 # Project constraints (for LLM context)
│           ├── heatsink_optimized_hint.md  # Per-project apply_optimized trigger text
│           ├── initial.json                # Default copper layout
│           └── heatsink_optimized.json     # Aluminum upgrade layout
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Key design decisions

| Concern | Approach |
|---|---|
| Chat logic | Keyword regex routing (`chat_responses.py`) or live Azure OpenAI call (`llm_backend.py`) |
| Thermal simulation | In-process NumPy Gaussian field + PINN overlay for heatsink project |
| PINN inference | FourierNetArch (PhysicsNeMo) run once offline; result cached as `.npy` in `assets/` |
| FAQ retrieval | CSV keyword-overlap scoring via `qa_loader.find_answer()` |
| Workspace mutations | Stored in `st.session_state["component_positions"]`; workspace re-renders on rerun |
| 3D preview | Interactive Plotly `Mesh3d` scene; heatsink uses PINN-derived temps and per-component z-heights |
| Future AI backends | `backends/placement.py` and `backends/thermal.py` stubs with `BackendResult` interface |

---

## UI Panels

| Panel | Location | Description |
|---|---|---|
| **Project Manager** | Left top | Collapsible project tree; clicking a project loads its component positions |
| **Material Library** | Left bottom | Component catalogue with TDP and Tcase_max values |
| **Design Workspace** | Right top | Mode-switched view: Modeling / Thermal Simulation / 3D |
| **Design Assistant** | Right bottom | Chat input + history; sends to `route_message()` |

---

## Running

### Docker (recommended)

```bash
# Build and start
docker compose up -d

# Access at
http://<host>:8601
```

The host directory is bind-mounted into the container (`. → /app`), so any file edits on the host are live immediately — Streamlit auto-reloads on save.

### Local

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

---

## AI Prompting Scheme

The AI Assistant uses Azure OpenAI via the Responses API (`POST /openai/responses?api-version=2025-04-01-preview`). Each request builds a message list with four layers, assembled in `backends/llm_backend.py`.

### Message structure

```
[system]      _SYSTEM_PROMPT
[user]        [GEOMETRY CONTEXT] + PROJECT RULES (<project>.md)  ← primed
[assistant]   {"response": "Context received.", ...}              ← primed
[user]        <previous turn 1>
[assistant]   <previous turn 1 reply>
  ...
[user]        <current user message>
```

#### 1. System prompt (`_SYSTEM_PROMPT`)

A static string injected once per request. Contains:

| Section | Purpose |
|---|---|
| Role & views | Describes the three workspace modes the user sees |
| Coordinate system | Origin top-left, X right, Y down, mm units |
| Response format | Instructs GPT to reply with a single JSON object (see below) |
| Placement rules | Supported actions: `move_sequence` steps with `direction`/`delta`/`rotate` |
| Mode switch rules | When to set `"mode_switch": "3D"` or `"Thermal Simulation"` |
| Thermal knowledge | IEC 62368-1 limits, CPU Tcase bounds, fan clearance, fin orientation |
| Project rules preamble | Instructs GPT to treat any project rules that follow as authoritative constraints |

#### 2. Geometry context (primed user turn)

Built by `_geometry_context()` and injected as a hidden user message that GPT acknowledges but does not respond to:

```
[GEOMETRY CONTEXT — do not respond to this]
CURRENT WORKSPACE MODE: Modeling
BOARD SIZE: 319 × 160 mm
COMPONENT POSITIONS:
  CPU: centre=(192.0, 95.5) mm, size=54.0×29.0 mm
  DDR-1: centre=(255.0, 64.5) mm, size=72.0×33.0 mm
  ...

PROJECT RULES (Hitatori):
<full contents of assets/project_rule/Hitatori.md>
```

The project rules block is only appended when a matching `.md` file exists under `assets/project_rule/`.

#### 3. Project rules files

`assets/project_rule/<ProjectName>.md` is the single source of truth for per-project constraints. Any project can have its own file — no code changes are needed.

Each file has two kinds of constraints:

| Kind | Marker | LLM instruction |
|---|---|---|
| **Fixed** | `= 0`, `_align` | Must never be violated |
| **Optimizable** | `> N mm` (lower bound) | May be increased for better thermal/airflow; minimum must be respected |

#### 4. Expected JSON response

GPT is instructed to always reply with one of these `placement` shapes:

```json
{
  "response": "<Markdown for chat bubble>",
  "mode_switch": null | "3D" | "Thermal Simulation",
  "placement": null
    | {"action": "move_sequence", "steps": [{"component": "<name>", "direction": "right|left|up|down", "delta": 5}, ...]}
    | {"action": "apply_optimized"}
}
```

`_parse_llm_output()` handles code-fence stripping and JSON decode errors, falling back to escaped plain text. The `action` returned to `app.py` follows the same contract as `chat_responses.route_message()`:

| Parsed value | Action returned | Effect |
|---|---|---|
| `mode_switch: "3D"` | `"switch_3d"` | Switches workspace to 3D preview |
| `mode_switch: "Thermal Simulation"` | `"switch_thermal"` | Switches workspace to heatmap |
| `placement.action: "move_sequence"` | the placement dict | `execute_instruction()` moves components by delta mm |
| `placement.action: "apply_optimized"` | `"apply_optimized"` | `_apply_optimized()` overwrites session positions from preset file; for heatsink, also switches PINN thermal field |
| `placement.action: "apply_optimized_thermal"` | `"apply_optimized_thermal"` | `_apply_optimized_thermal()` loads the `_optimized_4_thermal` preset (Step B) |
| anything else | `None` | No workspace change |

#### 5. Optimized preset (`apply_optimized`)

When `assets/project_rule/<project>_optimized.json` exists, the following chain fires:

```
1. _has_optimized_preset(project) → True
      ↓
2. Appended to primed context message:
     "OPTIMIZED PRESET AVAILABLE: yes
      If the user asks to optimize … you MUST respond with
      'placement': {'action': 'apply_optimized'}"
      ↓
3. User says something like:
     "請提供最佳元件擺放建議" / "optimize" / "apply best layout"
      ↓
4. GPT returns:
     {"placement": {"action": "apply_optimized"}, "response": "<thermal rationale>"}
      ↓
5. _parse_llm_output() → returns (response_text, "apply_optimized")
      ↓
6. app.py chat handler:
     elif action == "apply_optimized" and project:
         _apply_optimized(project)
      ↓
7. _apply_optimized() loads <project>_optimized.json,
   parses it through _parse_geometry_array(),
   overwrites st.session_state["component_positions"][project],
   clears st.session_state["sim_data"] → workspace re-renders
```

**Conditions required for `apply_optimized` to fire:**

| Condition | Where checked |
|---|---|
| Chat mode is **AI Assistant** (not Scripted) | `app.py` `_handle_chat()` |
| `assets/project_rule/<project>_optimized.json` exists | `_has_optimized_preset()` in `llm_backend.py` |
| User message requests optimization / best placement | GPT intent classification |
| GPT returns `"placement": {"action": "apply_optimized"}` | `_parse_llm_output()` |

**To add an optimized preset for a new project:**

1. Create `assets/project_rule/<ProjectName>/<ProjectName>_optimized.json` with the same array format as `initial.json`.
2. *(Optional)* Create `assets/project_rule/<ProjectName>/<ProjectName>_optimized_hint.md` to give GPT project-specific instructions about when and how to trigger the preset. If this file does not exist, `_load_optimized_hint()` falls back to the generic Hitatori-style prompt.

No code changes are needed for step 1. Step 2 lets you control exactly how the LLM describes and triggers the optimization for each project.

---

## Heatsink Project & PINN Thermal Inference

### Overview

The `heatsink` project demonstrates physics-informed neural network (PINN) thermal prediction integrated directly into the 2D heatmap and 3D views. Two pre-trained `FourierNetArch` (PhysicsNeMo) models are supported:

| Model | Checkpoint | Hottest plane | T range | Use case |
|---|---|---|---|---|
| **Copper** | `NV_heatsink_copper/thermal_solid_network.0.pth` | z_idx=28 (z=5.15, near outlet) | 25.1–28.9°C | Default / baseline layout |
| **Aluminum** | `NV_heatsink_aluminum/thermal_solid_network.0.pth` | z_idx=0 (z=−0.98, chip inlet) | 21.6–34.8°C | Optimized layout (post `apply_optimized`) |

Temperature reference: inlet coolant at 0°C; field values are `theta_s × 273.15`.

### Pre-computation

PINN inference runs inside the `physicsnemo` Docker image (GPU-accelerated). Results are saved once to `.npy` files and loaded at app startup — no GPU is needed at runtime:

```bash
# Copper field (z_idx=28)
docker run --rm --gpus all \
  -v /home/jack/thermal_helper/backends/Thermal_inference:/workspace \
  -v /home/jack/thermal_helper/assets:/assets \
  -w /workspace \
  data-service.inventec.com:1443/physicsnemo:latest \
  python extract_z28.py

# Aluminum field (z_idx=0)
docker run --rm --gpus all \
  -v /home/jack/thermal_helper/backends/Thermal_inference:/workspace \
  -v /home/jack/thermal_helper/assets:/assets \
  -w /workspace \
  data-service.inventec.com:1443/physicsnemo:latest \
  python extract_aluminum_z0.py
```

Each script saves a dict `{T_C, xs_nd, ys_nd, z_nd, description}` as a `.npy` file with `allow_pickle=True`.

### Coordinate systems

| Model | x (flow direction) | y (fin height) | z (depth) |
|---|---|---|---|
| Copper | [0.15, 1.15] | [0.00, 0.60] | [0.00, 5.70] |
| Aluminum | [−0.70, 0.70] | [−0.39, 0.39] | [−0.98, 0.77] |

> **Important:** never query a model outside its own training domain — PINNs extrapolate catastrophically outside their trained coordinate bounds.

### Field → board mapping

`thermal_sim._apply_pinn_heatsink_overlay()` maps the PINN XY slice onto the heatsink components:

- **PINN x** (flow direction) → **board Y-axis** of each heatsink body component
- **PINN y** (fin height) → averaged out (top-down 2D view)
- Temperature is interpolated linearly from the flow-direction profile at each component's Y centre

### Material-based field switching

`thermal_sim._hs_material(live_positions)` reads `live_positions["cu-base"]["material"]`:

- `copper` → copper PINN profile (uniform gradient, 2.2°C span)
- `aluminum` → aluminum PINN profile (steeper gradient, 6.7°C span)

This means the thermal overlay switches automatically when `apply_optimized` changes the layout from copper to aluminum — no extra code path needed.

### Heatsink optimization flow

```
User: "switch to aluminum" / "optimize heatsink" / "apply"
       ↓
LLM reads heatsink_optimized_hint.md from context
       ↓
GPT returns: {"placement": {"action": "apply_optimized"}, "mode_switch": "Thermal Simulation"}
       ↓
_apply_optimized("heatsink")
  → loads heatsink_optimized.json  (CU-BASE, HP-1..5, HP-xbar-1..5 material: copper → aluminum)
  → overwrites session positions
  → clears sim_data
       ↓
thermal_sim detects cu-base.material == "aluminum"
  → uses aluminum PINN profile
  → 2D heatmap and 3D boxes re-render with aluminum temperature field
```

### 3D thermal inspection

In the heatsink project, switching to **3D** mode (or clicking **🔲 View in 3D (thermal)** from the Thermal Simulation panel) renders each component as an interactive `Mesh3d` box. Components are physically stacked in z:

| Component | z extent (mm) |
|---|---|
| CPU-Substrate | 0.0 – 1.5 |
| CPU-Lid / CPU-Die | 1.5 – 4.0 |
| TIM1 / PTM7900 | 4.0 – 4.5 |
| CU-BASE | 4.5 – 9.0 |
| HP-1..5 (heat pipes) | 9.0 – 45.0 |
| HP-xbar-1..5 | 9.0 – 14.0 |
| FIN-array | 9.0 – 45.0 |

In thermal mode, each box is colored by its **PINN-derived temperature** (not a hardcoded table). Hovering over any part shows its predicted temperature, name, and exact position.

---

## Configuration

**`data/projects.json`** — defines the project tree, material library, and default component positions (x, y, w, h) for each project's workspace layout.

**`chat_responses.py`** — all scripted response strings are module-level constants at the top of the file; edit them here before a demo without touching routing logic.

**`thermal_sim.py`** — Gaussian heat source parameters (position, spread, peak temperature) are in the `_COMPONENTS` dict; adjust to change the heatmap appearance.

### Material options (for project presets)

Component material is configured per item in project preset JSON files, for example:

```json
{
      "id": "CPU",
      "x": 186,
      "y": 81,
      "w": 54,
      "h": 29,
      "rotated": 0,
      "mapping": 0,
      "level": 8,
      "material": "metal"
}
```

Supported named materials with dedicated colors in Modeling view:

- `metal`
- `copper`
- `aluminum`
- `ceramic`
- `graphite`
- `plastic`

Notes:

- Material values are normalized to lowercase when loaded.
- Any unknown material string is allowed and rendered with a fallback color.
- Default material is `metal` when the field is missing.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit >= 1.32` | UI framework |
| `matplotlib` | 2D modeling and thermal heatmap rendering |
| `numpy` | Temperature field simulation |
| `pandas` | CSV FAQ loading |
| `plotly` | Backend stub visualisations |
| `scikit-learn` | Available for future semantic search |
| `sentence-transformers` + `torch` | Available for future embedding-based retrieval |

Base Docker image: `pytorch-transformer-gpu-emc` (Inventec internal registry) — provides CUDA + PyTorch pre-installed.
