# CAD T — Thermal Design Assistant

A Streamlit-based live demo simulating an AI-assisted CAE/EE design workflow. The app presents a four-panel UI covering project management, a material library, an interactive design workspace (2D modeling, thermal heatmap, 3D preview), and a scripted AI chat assistant.

---

## Demo Flow

The demo walks through 11 scripted steps across two feature sets:

**Feature 1 — Smart Placement (Steps 1–5)**
1. Open web app → 4-panel layout loads
2. Select project `EE_PCB_demo0420` → 2D Modeling view renders
3. Ask about CPU/heatsink spacing → AI returns placement impact analysis + recommendations
4. Type `Apply optimized placement` → workspace animates component position updates (CPU shifts –2 mm, heatsink rotates 90°)
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
│   └── thermal.py          # Thermal backend stub (future: RC-network solver)
├── data/
│   ├── projects.json       # Project tree + component metadata + default positions
│   └── (AIC 0408) Ai for design_Thermal Expert System_FAQ v.1.csv
├── assets/
│   └── ThermalOnPCB.png    # Static 3D preview image
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Key design decisions

| Concern | Approach |
|---|---|
| Chat logic | Keyword regex routing in `chat_responses.py` — no live LLM calls |
| Thermal simulation | In-process NumPy Gaussian field, `run_simulation()` → 100×150 array |
| FAQ retrieval | CSV keyword-overlap scoring via `qa_loader.find_answer()` |
| Workspace mutations | Stored in `st.session_state["component_positions"]`; workspace re-renders on rerun |
| 3D preview | Static `st.image()` swap — no interactive 3D engine |
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

## Configuration

**`data/projects.json`** — defines the project tree, material library, and default component positions (x, y, w, h) for each project's workspace layout.

**`chat_responses.py`** — all scripted response strings are module-level constants at the top of the file; edit them here before a demo without touching routing logic.

**`thermal_sim.py`** — Gaussian heat source parameters (position, spread, peak temperature) are in the `_COMPONENTS` dict; adjust to change the heatmap appearance.

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
