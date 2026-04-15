# Plan: Streamlit CAE Engineer Demo Page

**TL;DR**: A single Streamlit app with 4 visible panels, two functional modes (Modeling + Thermal Simulation), fully scripted chat responses matched by keyword, a numpy-based thermal heatmap simulation, and a CSV-backed QA system.

---

## File Structure
```
demo/
├── app.py                  ← main Streamlit app, layout + wiring
├── thermal_sim.py          ← generates 2D numpy temperature field
├── qa_loader.py            ← loads CSV + keyword-match for FAQs
├── chat_responses.py       ← hardcoded scripted response library
├── data/
│   ├── qa_database.csv     ← columns: id, question, answer, category
│   └── projects.json       ← project tree + component metadata
├── assets/
│   ├── 3d_preview.png      ← static 3D view placeholder
│   └── pcb_layout_base.png ← optional PCB background
└── requirements.txt
```

---

## Phase 1 — Layout Shell
1. `st.columns([1, 3])` for left panel (Project Manager + Material Library) and right panel (workspace + chat)
2. Custom CSS via `st.markdown()` to render card-style panel borders matching the wireframe
3. `st.session_state` initialized on startup: `selected_project`, `mode`, `view`, `chat_history`, `component_positions`, `sim_data`

## Phase 2 — Project Manager + Material Library
4. Nested `st.expander` blocks to simulate a collapsible project tree (▼ ME-Placement-1041 → Heat-exchanger_Thermal_02...)
5. Clicking a project name sets `session_state.selected_project` → workspace re-render triggered
6. Material Library: same expander pattern (▼ CPU → Intel Core Ultra, AMD Ryzen 9000)

## Phase 3 — Design Workspace
7. Mode switcher: `st.radio(["Modeling", "Thermal Simulation"], horizontal=True)` above workspace
8. `render_workspace(mode, view, component_positions)` dispatcher:
   - **Modeling**: matplotlib figure with labeled rectangles (CPU, heatsink, fan) on a PCB grid
   - **Thermal Simulation**: `plt.imshow()` with `"hot"` colormap, colorbar 20–120°C, temperature annotations on components
   - **3D**: `st.image("assets/3d_preview.png")`
9. `thermal_sim.py` → `run_simulation(project_name)` returns 150×100 numpy array with CPU hotspot ~89°C; runs inside `st.spinner()`

## Phase 4 — Design Assistant Chat
10. `st.chat_input("請在此處輸入對話...")` + `st.chat_message` bubbles from `session_state.chat_history`
11. `route_message(user_input, mode)` in `chat_responses.py` — keyword pattern → `(response_text, workspace_action)`:

| Keyword trigger | Response | Action |
|---|---|---|
| "spacing" / "間距" / "placement" | Step 3 placement analysis text | — |
| "Apply optimized placement" | Step 4 progress animation | `apply_placement` |
| "3D" / "Switch to 3D" | Confirm text | `switch_3d` |
| "表面溫度" / "surface temperature" | IEC 62368-1 thermal limits (from CSV) | — |
| "天線" / "antenna" / "EMI" | "Query classified as RF/EMI..." + recommendations | — |
| "Apply layout adjustment" | Fan move progress animation | `apply_layout` |

12. `apply_placement` action: shifts CPU x-position by -2mm in `component_positions`, heatsink flag set to rotated → workspace re-renders
13. Progress animations use `st.empty()` + `time.sleep(0.5)` typewriter pattern for "Applying placement update... → Updating positions → ..."

## Phase 5 — QA CSV Integration
14. `qa_loader.py`: `load_qa(path)` → list of dicts; `find_answer(query, qa_data)` → keyword overlap score across `question` + `category` columns
15. Thermal FAQ (step 7) and RF expert (steps 8–9) invoke `find_answer()` before falling back to hardcoded defaults

## Phase 6 — Polish
16. Disable mode switch / project selection to avoid accidental re-runs mid-chat
17. "Clear chat" button to reset session
18. `requirements.txt`: `streamlit`, `matplotlib`, `numpy`, `pandas`

---

## Relevant Files to Create
- `app.py` — wires everything together; main layout
- `thermal_sim.py` — numpy simulation, reusable function
- `qa_loader.py` — CSV loading + matching
- `chat_responses.py` — all scripted text lives here (easy to edit before demo)
- `data/qa_database.csv` — sample rows for thermal + RF categories

---

## Verification
1. `streamlit run app.py` — 4 panels render, no overflow
2. Select project → workspace shows 2D PCB layout
3. Switch to Thermal Simulation → heatmap with colorbar appears
4. Type placement query → Step 3 scripted response rendered
5. Type "Apply optimized placement" → typewriter progress → workspace component positions shift
6. Type "3D" → workspace swaps to static 3D image
7. Type "風扇轉輪對天線信號的干擾" → RF expert branch triggered
8. CSV QA loads cleanly; `find_answer()` returns relevant thermal row

---

## Decisions
- No real LLM calls — all chat logic is keyword routing in `chat_responses.py` (swappable later)
- Simulation is an in-process Python function, not subprocess
- Project selection via click, no drag-and-drop
- 3D preview is a static image swap, not interactive

---

## Open Questions
- Should scripted responses appear as typewriter/streaming effect (`st.write_stream`) or as instant full text blocks?
- Should the QA CSV fallback show a "no match found" message or silently use the hardcoded default?
- What PCB board dimensions and component default coordinates should be used in the 2D Modeling view?
- Should Material Library selection affect which components appear in the Design Workspace?
