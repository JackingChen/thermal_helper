# CAD T — System Workflow

> Maps every script in the project to its responsibilities, the data it consumes/produces, and how the pieces connect at runtime.

---

## High-Level Architecture

```
Browser (User)
     │  HTTP
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  app.py  — Streamlit entry point, UI layout, state machine      │
│                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│   │ Project Mgr  │   │  Workspace   │   │  Design Assistant │  │
│   │  (left top)  │   │ (right top)  │   │  (right bottom)   │  │
│   └──────┬───────┘   └──────┬───────┘   └────────┬──────────┘  │
└──────────┼────────────────── ┼─────────────────── ┼────────────┘
           │                   │                     │
   data/projects.json   thermal_sim.py          chat_responses.py
                        assets/ThermalOnPCB.png      │
                                                qa_loader.py
                                                     │
                                           data/FAQ.csv
                                                     │
                                      backends/placement.py  (stub)
                                      backends/thermal.py    (stub)
```

---

## Script-by-Script Breakdown

### `app.py` — Main Application Controller

**Role:** Streamlit entry point. Owns the entire UI layout, all session state, and the event-dispatch loop.

| Responsibility | Key Functions |
|---|---|
| Page setup & global CSS | `st.set_page_config()`, inline `<style>` block |
| Static data loading (cached) | `_load_projects()` → reads `data/projects.json`; `_load_qa_cached()` → delegates to `qa_loader.load_qa()` |
| Session state initialisation | `_init_state()` — seeds `selected_project`, `mode`, `chat_history`, `component_positions`, `sim_data`, `locked` |
| Component position management | `_get_positions(project)` — lazy-copies default positions from JSON into `st.session_state` |
| Simulation cache management | `_ensure_sim(project)` — calls `thermal_sim.run_simulation()` once per project selection |
| 2-D Modeling renderer | `_render_modeling(project, positions)` — draws PCB + component rectangles via Matplotlib |
| Thermal heatmap renderer | `_render_thermal(project, positions)` — overlays Matplotlib heatmap on sim array |
| 3-D preview renderer | `_render_3d()` — displays static `assets/ThermalOnPCB.png` |
| Workspace mode routing | `render_workspace()` — dispatches to one of the three renderers above based on `st.session_state["mode"]` |
| Placement mutation | `_apply_placement(project)` — shifts CPU x −2 mm, marks heatsink `rotated=True`, invalidates sim cache |
| Layout mutation | `_apply_layout(project)` — moves fan x +5 mm, invalidates sim cache |
| Typewriter animation | `_typewriter(placeholder, lines)` — streams assistant response lines with `time.sleep` |
| Chat dispatch | `_handle_chat(user_input)` — appends user message, calls `chat_responses.route_message()`, executes returned workspace action, appends assistant message |
| UI layout | Four panels: Project Manager (left-top), Material Library (left-bottom), Design Workspace (right-top), Design Assistant (right-bottom) |

**Data flow (input → output):**
```
data/projects.json ──► _load_projects() ──► project tree / material catalogue / default positions
qa_loader.load_qa() ──► _qa_data (list[dict]) ──► passed to route_message() on each chat
thermal_sim.run_simulation() ──► st.session_state["sim_data"] (ndarray) ──► heatmap render
route_message() ──► (response_text, action) ──► chat history + workspace mutation
```

---

### `chat_responses.py` — Chat Router & Scripted Responses

**Role:** All AI chat logic lives here. Contains every scripted string and a keyword-regex router. No live LLM calls — all responses are deterministic.

| Responsibility | Key Elements |
|---|---|
| Scripted response constants | Module-level `_STEP3_PLACEMENT`, `_STEP4_PLACEMENT_PROGRESS`, …, `_STEP11_3D_CONFIRM`, `_FALLBACK` — edit these to change demo text |
| Keyword matcher helper | `_match(text, *patterns)` — applies `re.search(pattern, text, IGNORECASE)` across multiple patterns |
| Main router | `route_message(user_input, mode, qa_data)` → `tuple[str, str | None]` |
| Workspace action signals | Returns one of `None`, `"apply_placement"`, `"apply_layout"`, `"switch_3d"`, `"switch_thermal"` as the second tuple element |
| CSV lookup delegation | `_csv_lookup(query, qa_data)` — calls `qa_loader.find_answer()` for FAQ-backed answers |
| CSV response formatter | `_format_csv_row(row)` — renders a FAQ dict into a Markdown block (Q / answer / guideline / reference) |

**Routing priority (first match wins):**

```
1. "apply optimized placement" / "apply.*placement"  → _STEP4  + action:"apply_placement"
2. "apply layout adjustment"  / "apply.*layout"      → _STEP10 + action:"apply_layout"
3. "3D" / "switch.*3D" variants                      → _STEP11 + action:"switch_3d"
4. "thermal sim" / "live sim" variants                → _STEP6  + action:"switch_thermal"
5. Antenna / EMI / RF keywords                        → CSV lookup → _STEP8 + _STEP9
6. Surface temperature / IEC 62368 keywords           → CSV lookup → _STEP7
7. Spacing / placement / heatsink keywords            → _STEP3
8. Generic → CSV lookup → _FALLBACK
```

**Public API:**
```python
route_message(user_input: str, mode: str, qa_data: list | None) -> tuple[str, str | None]
```

---

### `qa_loader.py` — FAQ CSV Loader & Retriever

**Role:** Loads the traditional-Chinese FAQ CSV once (LRU-cached) and provides keyword-overlap scoring for retrieval.

| Responsibility | Key Functions |
|---|---|
| CSV loading with encoding fallback | `load_qa(path?)` — tries big5 → gbk → utf-8-sig → utf-8 → cp950; renames columns using `_COL_MAP` |
| Column normalisation | `_COL_MAP` maps Traditional Chinese headers → short English keys (`category`, `question`, `expert_answer`, `guideline`, `reference`) |
| Text normalisation | `_normalise(text)` — NFKC unicode, lowercase, collapse whitespace |
| Tokenisation | `_tokenise(text)` — extracts ASCII words + individual CJK characters as tokens |
| Best-match retrieval | `find_answer(query, qa_data?, top_n=1)` — token-overlap score across question (×2 weight), problem_reference, category fields |

**Data flow:**
```
data/(AIC 0408)…FAQ v.1.csv
        │  (big5/gbk/utf-8 auto-detect)
        ▼
load_qa()  ──LRU cached──►  list[dict]  (one dict per FAQ row)
        │
        ▼
find_answer(query)  ──►  best matching dict | None
```

**Column mapping (`_COL_MAP`):**

| CSV header (Traditional Chinese) | Normalised key |
|---|---|
| 分類模組 | `category` |
| 核心設計問題 (Question) | `question` |
| 技術參考背景 (Problem Reference) | `problem_reference` |
| 專家建議答案 (Expert Answer) | `expert_answer` |
| 答案技術指標/設計準則 (Technical Guideline) | `guideline` |
| 參考資料 | `reference` |

---

### `thermal_sim.py` — 2-D Temperature Field Generator

**Role:** Pure-NumPy simulation engine for generating the PCB thermal heatmap. No external solvers — uses Gaussian heat source superposition.

| Responsibility | Key Functions |
|---|---|
| Component geometry table | `_COMPONENTS` dict — per-project heat source positions (row/col), half-extents, and peak temperatures |
| Gaussian field builder | `_gaussian(rows, cols, r0, c0, sr, sc, amplitude)` — single 2-D Gaussian kernel |
| Temperature field simulation | `run_simulation(project_name)` — superimposes all component Gaussians over ambient, clips to [20, 120] °C, adds ±0.3 °C noise |
| Component position export | `get_component_positions(project_name)` — returns `{name: {x, y, T}}` dict for annotation overlays |

**Data flow:**
```
_COMPONENTS[project_name]
        │
        ▼
run_simulation()  ──►  ndarray shape (100, 150), dtype float32
                                │
                                ▼
                   app._render_thermal() → Matplotlib imshow heatmap

get_component_positions()  ──►  {name: {x, y, T}}
                                │
                                ▼
                   app._render_thermal() → annotation badges
```

**Simulation parameters (editable in `_COMPONENTS`):**

| Field | Meaning |
|---|---|
| `r`, `c` | Row/col centre of heat source on the 100×150 grid |
| `hr`, `hc` | Half-height / half-width (spread = 2× these) |
| `T` | Peak temperature (°C) |
| `ambient` | Background temperature (°C) |

---

### `backends/__init__.py` — Backend Plugin Interface

**Role:** Defines the `BackendResult` dataclass that all backend modules must return.

```python
@dataclass
class BackendResult:
    title:   str                  # Panel label
    figure:  plotly.graph_objects.Figure   # Plotly visualisation
    summary: str                  # Text description
```

Any module in `backends/` that implements `run(context: dict) -> BackendResult` is a valid backend plugin.

---

### `backends/placement.py` — Placement Backend (Stub)

**Role:** Future home of a PCB component placement optimisation solver.

| Current state | Placeholder 3-D Plotly figure labelled "reserved for future implementation" |
|---|---|
| Planned function | Accept PCB geometry + component list → run placement optimisation → return 3-D layout figure + placement report |
| Entry point | `run(context: dict) -> BackendResult` |

---

### `backends/thermal.py` — Thermal Backend (Stub)

**Role:** Future home of an RC-network thermal solver.

| Current state | Placeholder Plotly figure labelled "reserved for future implementation" |
|---|---|
| Planned function | Accept TDP / Rth values + material properties → run RC-network simulation → return temperature distribution figure + numeric summary |
| Entry point | `run(context: dict) -> BackendResult` |

---

## End-to-End Request Flow

### Chat message → workspace update

```
User types text
      │
      ▼
app._handle_chat(user_input)
      │
      ├─► logs user message to st.session_state["chat_history"]
      │
      ├─► chat_responses.route_message(user_input, mode, qa_data)
      │         │
      │         ├─ keyword match (regex)
      │         │         │
      │         │         └─ FAQ query? ──► qa_loader.find_answer()
      │         │                                │
      │         │                         data/FAQ.csv (cached)
      │         │
      │         └─► (response_text, workspace_action)
      │
      ├─► execute workspace_action:
      │       "apply_placement"  ──► app._apply_placement()  → mutates session_state positions
      │       "apply_layout"     ──► app._apply_layout()     → mutates session_state positions
      │       "switch_3d"        ──► session_state["mode"] = "3D"
      │       "switch_thermal"   ──► session_state["mode"] = "Thermal Simulation"
      │
      ├─► logs assistant message to st.session_state["chat_history"]
      │
      └─► st.rerun()  ──► full UI re-render with updated state
```

### Project selection → thermal heatmap

```
User clicks project button
      │
      ▼
app selects project, resets sim_data & mode="Modeling"
      │
      ▼
st.rerun() → render_workspace() → _render_modeling()
      │   (matplotlib rectangles from session_state positions)
      │
User switches mode → "Thermal Simulation"
      │
      ▼
render_workspace() → _render_thermal()
      │
      ├─► _ensure_sim(project) → thermal_sim.run_simulation(project)
      │         └─► returns ndarray (100×150), cached in session_state["sim_data"]
      │
      ├─► matplotlib imshow(sim_data)
      │
      └─► thermal_sim.get_component_positions(project)
                └─► annotation badges (name + °C) overlaid on heatmap
```

---

## Data Files

| File | Consumer | Purpose |
|---|---|---|
| `data/projects.json` | `app.py` | Project tree, material library catalogue, default component positions (x, y, w, h) per project |
| `data/(AIC 0408)…FAQ v.1.csv` | `qa_loader.py` | Thermal/RF expert Q&A knowledge base (Traditional Chinese headers) |
| `assets/ThermalOnPCB.png` | `app._render_3d()` | Static 3-D preview image shown in 3D workspace mode |

---

## State Variables (`st.session_state`)

| Key | Type | Set by | Read by |
|---|---|---|---|
| `selected_project` | `str \| None` | Project Manager buttons | All renderers, `_handle_chat` |
| `mode` | `str` | Mode radio, `route_message` actions | `render_workspace`, mode switcher |
| `chat_history` | `list[dict]` | `_handle_chat` | Chat history HTML renderer |
| `component_positions` | `dict[project → dict]` | `_get_positions`, `_apply_*` | All workspace renderers |
| `sim_data` | `ndarray \| None` | `_ensure_sim` | `_render_thermal` |
| `locked` | `bool` | (reserved) | Mode radio, chat input `disabled=` |

---

## Dependency Graph

```
app.py
  ├── chat_responses.py
  │       └── qa_loader.py
  │               └── data/FAQ.csv
  ├── qa_loader.py           (direct cache pre-load)
  ├── thermal_sim.py
  ├── backends/__init__.py   (BackendResult — imported by backend stubs)
  │       ├── backends/placement.py
  │       └── backends/thermal.py
  └── data/projects.json
      assets/ThermalOnPCB.png
```
