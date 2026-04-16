"""
app.py
───────
Main Streamlit application — CAE Engineer Design Assistant Demo.

Layout
──────
  Left column  : Project Manager  (top)  + Material Library (bottom)
  Right column : Design Workspace (top)  + Design Assistant chat (bottom)

Run
───
    streamlit run app.py
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from backends.placement import apply_placement as _placement_action
from backends.placement import apply_layout as _layout_action
from backends.placement import execute_instruction as _exec_placement
from backends.llm_backend import call_azure_llm
from chat_responses import route_message
from qa_loader import load_qa
from thermal_sim import get_component_positions, run_simulation

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_DATA_DIR           = _HERE / "data"
_PROJECTS_JSON      = _HERE / "data" / "projects.json"
_ASSET_3D           = _HERE / "assets" / "ThermalOnPCB.png"
_ASSET_FLOW_GEO     = _HERE / "assets" / "vti_geometry_3d.png"
_ASSET_FLOW_LINES   = _HERE / "assets" / "vti_flow_streamlines.png"
_ASSET_FLOW_HTML    = _HERE / "assets" / "model_3d_demo.html"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAD T — Design Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #1a1a2e;
        color: #e0e0e0;
        font-family: "Segoe UI", sans-serif;
    }
    /* hide default Streamlit header/footer */
    header[data-testid="stHeader"], footer { display: none !important; }

    /* ── panel cards ── */
    .panel-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        min-height: 60px;
    }
    .panel-title {
        color: #e94560;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    /* ── mode switcher ── */
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        border: 1px solid #0f3460;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.8rem;
        cursor: pointer;
    }

    /* ── chat bubbles ── */
    .chat-user   { text-align: right; color: #a0c4ff; }
    .chat-assist { text-align: left;  color: #caffbf; }

    /* ── project tree items ── */
    .tree-item {
        padding: 2px 0 2px 10px;
        font-size: 0.82rem;
        cursor: pointer;
        color: #b0b8cc;
    }
    .tree-item:hover { color: #ffffff; }
    .tree-selected   { color: #e94560 !important; font-weight: 600; }

    /* scrollable chat history */
    .chat-scroll {
        max-height: 340px;
        overflow-y: auto;
        padding-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load static data ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_projects() -> dict:
    with open(_PROJECTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _parse_geometry_array(items: list[dict]) -> dict:
    """
    Convert new array geometry format → internal positions dict.

    Input  : list of {id, x, y (top-left), w, h, rotated (bit0), mapping (bit1), level}
    Output : {comp_name: {x (centre), y (centre), w, h, label, rotated (bool)}}
             plus special key "_board": {w, h} derived from edge_right / edge_top markers.

    Rotation encoding:
        angle = (rotated + mapping * 2) * 90 °CCW
        90° / 270° swaps w and h (axis-aligned bounding box).
    """
    positions: dict = {}
    board_w: float = 150.0
    board_h: float = 100.0

    # Only render the highest level present (avoids duplicate level 7 + 8 overlaps)
    non_edge = [it for it in items if not it["id"].startswith("edge_")]
    if non_edge:
        max_level = max(int(it.get("level", 0)) for it in non_edge)
    else:
        max_level = 0

    for item in items:
        name = item["id"]
        # Edge markers define board extents — not rendered as components
        if name.startswith("edge_"):
            if name == "edge_right":
                board_w = float(item["x"])
            elif name == "edge_top":
                board_h = float(item["y"])
            continue

        # Skip levels below the highest
        if int(item.get("level", 0)) < max_level:
            continue

        rot_bits = int(item.get("rotated", 0)) + int(item.get("mapping", 0)) * 2
        angle = rot_bits * 90  # degrees CCW
        w = float(item["w"])
        h = float(item["h"])
        if angle in (90, 270):
            w, h = h, w  # swap for axis-aligned rotation

        # JSON stores top-left; internal format uses centre
        positions[name.lower()] = {
            "x": float(item["x"]) + w / 2,
            "y": float(item["y"]) + h / 2,
            "w": w,
            "h": h,
            "label": item["id"],          # original case for display
            "rotated": angle in (90, 270),
        }

    positions["_board"] = {"w": board_w, "h": board_h}
    return positions


@st.cache_data(show_spinner=False)
def _load_extra_positions() -> dict:
    """
    Scan data/<project>/<project>.json subdirectories.
    Supports both new array format (list) and legacy dict format.
    """
    merged: dict = {}
    for proj_dir in sorted(_DATA_DIR.iterdir()):
        if not proj_dir.is_dir():
            continue
        json_file = proj_dir / f"{proj_dir.name}.json"
        if not json_file.exists():
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                merged[proj_dir.name] = _parse_geometry_array(data)
            elif isinstance(data, dict):
                # Legacy dict format {comp_name: {...}}
                merged[proj_dir.name] = data
        except Exception:
            pass
    return merged


@st.cache_data(show_spinner=False)
def _load_qa_cached() -> list:
    return load_qa()


_proj_data       = _load_projects()
_extra_positions = _load_extra_positions()
_qa_data         = _load_qa_cached()


# ── Session state initialisation ───────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "selected_project":    None,
        "mode":                "Modeling",   # "Modeling" | "Thermal Simulation" | "3D"
        "chat_mode":           "AI Assistant",   # "Scripted" | "AI Assistant"
        "chat_history":        [],           # list of {"role": "user"|"assistant", "text": str}
        "component_positions": {},           # mutable copy of default positions
        "sim_data":            None,         # cached numpy array
        "locked":              False,        # disable controls mid-demo
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_positions(project: str | None) -> dict:
    """Return mutable component position dict for the current project."""
    if not project:
        return {}
    if project not in st.session_state["component_positions"]:
        # 1. Primary: data/<project>/<project>.json subdirectory files
        pos = _extra_positions.get(project)
        # 2. Fallback: projects.json default_component_positions (legacy)
        if pos is None:
            defaults = _proj_data.get("default_component_positions", {})
            pos = defaults.get(project)
        st.session_state["component_positions"][project] = copy.deepcopy(pos or {})
    return st.session_state["component_positions"][project]


def _ensure_sim(project: str | None) -> np.ndarray | None:
    if not project:
        return None
    if st.session_state["sim_data"] is None:
        st.session_state["sim_data"] = run_simulation(project)
    return st.session_state["sim_data"]


# ── Workspace renderers ────────────────────────────────────────────────────────

def _render_modeling(project: str, positions: dict) -> None:
    """2-D PCB component layout with matplotlib rectangles."""
    # Board dimensions from _board key (set by _parse_geometry_array)
    board_info = positions.get("_board", {})
    board_w = float(board_info.get("w", 150))
    board_h = float(board_info.get("h", 100))

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    # PCB board outline
    board = mpatches.FancyBboxPatch(
        (0, 0), board_w, board_h,
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor="#2a9d8f", facecolor="#0a1628",
    )
    ax.add_patch(board)

    colours = {
        "heatsink": ("#264653", "#264653"),
        "cpu":      ("#e9c46a", "#e9c46a"),
        "fan":      ("#2a9d8f", "#2a9d8f"),
    }

    # Exclude internal metadata key before rendering
    render_positions = {k: v for k, v in positions.items() if k != "_board"}
    # Draw larger components first so smaller ones (CPU) render on top
    sorted_comps = sorted(render_positions.items(), key=lambda kv: kv[1]["w"] * kv[1]["h"], reverse=True)
    for name, comp in sorted_comps:
        # Apply runtime rotation: swap w/h when rotated flag is set
        w = comp["h"] if comp.get("rotated") else comp["w"]
        h = comp["w"] if comp.get("rotated") else comp["h"]
        x, y = comp["x"] - w / 2, comp["y"] - h / 2
        fc, ec = colours.get(name, ("#6c757d", "#adb5bd"))
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0.5", linewidth=1,
            edgecolor=ec, facecolor=fc, alpha=0.85,
        )
        ax.add_patch(rect)
        label = comp.get("label", name.upper())
        # Offset CPU label upward slightly so it doesn't overlap Heatsink label
        y_offset = -h * 0.18 if name == "heatsink" else 0
        ax.text(
            comp["x"], comp["y"] + y_offset, label,
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold",
        )

    ax.set_xlim(-5, board_w + 5)
    ax.set_ylim(-5, board_h + 5)
    ax.set_aspect("equal")
    ax.set_title(f"{project}  |  Mode: Modeling", color="#adb5bd", fontsize=9)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1d3557")

    ax.grid(True, color="#1d3557", linewidth=0.5, alpha=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_thermal(project: str, positions: dict) -> None:
    """2-D thermal heatmap with component annotations."""
    sim = _ensure_sim(project)
    if sim is None:
        st.warning("No simulation data — select a project first.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    img = ax.imshow(
        sim, cmap="hot", origin="upper",
        vmin=20, vmax=120,
        extent=[0, 150, 100, 0],
        aspect="auto",
    )
    cbar = fig.colorbar(img, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Temperature (°C)", color="#adb5bd", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#adb5bd")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#adb5bd")

    # Annotate components with temperature badges
    sim_positions = get_component_positions(project)
    for name, info in sim_positions.items():
        ax.annotate(
            f"{info['T']:.1f}°C\n{name.upper()}",
            xy=(info["x"], info["y"]),
            fontsize=6.5, color="white", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#333333cc", edgecolor="#ffffff55"),
        )

    ax.set_title(
        f"{project}  |  Thermal Simulation (20–120 °C)",
        color="#adb5bd", fontsize=9,
    )
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1d3557")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_3d(project: str | None, positions: dict) -> None:
    """Interactive 3-D plotly PCB component visualisation."""

    # ── Board dimensions ──────────────────────────────────────────────────────
    board_info = positions.get("_board", {})
    board_w = float(board_info.get("w", 150))
    board_h = float(board_info.get("h", 100))

    # ── Per-component z-extents (mm) and colours ──────────────────────────────
    _Z = {
        "heatsink": (1.5, 20.0),
        "cpu":      (1.5,  8.0),
        "fan":      (1.5, 18.0),
    }
    _CLR = {
        "heatsink": "#264653",
        "cpu":      "#e9c46a",
        "fan":      "#2a9d8f",
    }

    # ── Box mesh helper ───────────────────────────────────────────────────────
    def _box(x0, y0, z0, x1, y1, z1, color, name):
        """
        Return a Mesh3d trace for one axis-aligned box.
        Vertices (8): v0=(x0,y0,z0) … v7=(x1,y1,z1)
        12 triangles covering all 6 faces.
        """
        vx = [x0,x1,x0,x1, x0,x1,x0,x1]
        vy = [y0,y0,y1,y1, y0,y0,y1,y1]
        vz = [z0,z0,z0,z0, z1,z1,z1,z1]
        # face normals point outward
        i_ = [0, 0,  4, 4,  0, 0,  2, 2,  0, 0,  1, 1]
        j_ = [1, 3,  6, 7,  4, 5,  3, 7,  2, 6,  5, 7]
        k_ = [3, 2,  7, 5,  5, 1,  7, 6,  6, 4,  7, 3]
        return go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=i_, j=j_, k=k_,
            color=color, opacity=0.88,
            name=name, showlegend=True,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"x: {x0:.1f}–{x1:.1f} mm<br>"
                f"y: {y0:.1f}–{y1:.1f} mm<br>"
                f"z: {z0:.1f}–{z1:.1f} mm<extra></extra>"
            ),
        )

    traces: list[go.BaseTraceType] = []

    # PCB board (thin green slab)
    traces.append(_box(0, 0, 0, board_w, board_h, 1.5, "#0a3622", "PCB Board"))

    # Components
    render_positions = {k: v for k, v in positions.items() if k != "_board"}
    for name, comp in render_positions.items():
        cx, cy = comp["x"], comp["y"]
        w, h    = comp["w"], comp["h"]
        z0, z1  = _Z.get(name, (1.5, 6.0))
        color   = _CLR.get(name, "#6c757d")
        label   = comp.get("label", name.upper())
        traces.append(_box(cx - w/2, cy - h/2, z0, cx + w/2, cy + h/2, z1, color, label))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            bgcolor="#0d1b2a",
            xaxis=dict(title="X (mm)", color="#adb5bd", gridcolor="#1d3557", showbackground=False),
            yaxis=dict(title="Y (mm)", color="#adb5bd", gridcolor="#1d3557", showbackground=False),
            zaxis=dict(title="Z (mm)", color="#adb5bd", gridcolor="#1d3557", showbackground=False),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
        legend=dict(bgcolor="#0d1b2a", bordercolor="#0f3460", borderwidth=1),
        title=dict(
            text=f"{project or ''}  |  3D Preview",
            font=dict(color="#adb5bd", size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_flow_geometry() -> None:
    """CFD geometry views generated from grid_inference_flow.vti."""
    if not _ASSET_FLOW_GEO.exists():
        st.warning(
            "Flow geometry assets not yet generated. "
            "Run **vti_render.ipynb** inside the Docker container first."
        )
        return

    tab_iso, tab_stream, tab_interactive = st.tabs(
        ["Isometric View", "Streamlines", "Interactive 3D"]
    )

    with tab_iso:
        if _ASSET_FLOW_GEO.exists():
            st.image(str(_ASSET_FLOW_GEO),
                     caption="Surface geometry coloured by pressure (Pa)")
        else:
            st.info("Geometry image not found.")

    with tab_stream:
        if _ASSET_FLOW_LINES.exists():
            st.image(str(_ASSET_FLOW_LINES),
                     caption="Velocity streamlines seeded from inlet face")
        else:
            st.info("Streamline image not found — re-run cell 6 in vti_render.ipynb.")

    with tab_interactive:
        if _ASSET_FLOW_HTML.exists():
            with open(_ASSET_FLOW_HTML, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=False)
        else:
            st.info("Interactive HTML not found — re-run cell 7 in vti_render.ipynb.")


def render_workspace() -> None:
    project   = st.session_state["selected_project"]
    mode      = st.session_state["mode"]
    positions = _get_positions(project)

    if mode == "3D":
        _render_3d(project, positions)
    elif mode == "Thermal Simulation":
        if project:
            _render_thermal(project, positions)
        else:
            st.info("Select a project to run thermal simulation.")
    elif mode == "Flow Geometry":
        _render_flow_geometry()
    else:  # Modeling
        if project:
            _render_modeling(project, positions)
        else:
            st.info("Select a project from the Project Manager to begin.")


# ── Workspace action handlers ──────────────────────────────────────────────────

def _apply_placement(project: str) -> None:
    """Scripted Step-4: delegate to backends.placement."""
    _placement_action(_get_positions(project))
    st.session_state["sim_data"] = None


def _apply_layout(project: str) -> None:
    """Scripted Step-10: delegate to backends.placement."""
    _layout_action(_get_positions(project))
    st.session_state["sim_data"] = None


def _apply_instruction(project: str, instruction: dict) -> None:
    """Apply any instruction dict (from LLM or scripted preset) to the current project."""
    _exec_placement(_get_positions(project), instruction)
    st.session_state["sim_data"] = None


# ── Progress animation helper ──────────────────────────────────────────────────

def _typewriter(placeholder, lines: list[str], delay: float = 0.35) -> None:
    """Display lines one by one with a short pause."""
    displayed = ""
    for line in lines:
        displayed += line + "\n"
        placeholder.markdown(displayed)
        time.sleep(delay)


# ── Chat handler ───────────────────────────────────────────────────────────────

def _handle_chat(user_input: str) -> None:
    project = st.session_state["selected_project"]
    mode    = st.session_state["mode"]
    chat_mode = st.session_state["chat_mode"]

    # Log user message
    st.session_state["chat_history"].append({"role": "user", "text": user_input})

    # Route — scripted or LLM
    if chat_mode == "AI Assistant":
        positions = _get_positions(project) if project else {}
        with st.spinner("AI Assistant is thinking…"):
            status_box = st.empty()
            status_box.markdown(
                "_Sending geometry context and chat history to Azure OpenAI…_"
            )
            response_text, action = call_azure_llm(
                user_input,
                st.session_state["chat_history"],
                positions,
                mode,
            )
            status_box.markdown("_Parsing response…_")
            status_box.empty()
    else:
        response_text, action = route_message(user_input, mode, _qa_data)

    # Execute workspace actions
    if isinstance(action, dict) and project:
        _apply_instruction(project, action)
    elif action == "switch_3d":
        st.session_state["mode"] = "3D"
    elif action == "switch_thermal":
        st.session_state["mode"] = "Thermal Simulation"

    st.session_state["chat_history"].append({"role": "assistant", "text": response_text})


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════

# Top title bar
st.markdown(
    "<h3 style='color:#e94560; margin:0 0 8px 0; font-size:1.1rem; "
    "letter-spacing:0.05em;'>● CAD T — Thermal Design Assistant</h3>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 3], gap="medium")

# ── LEFT COLUMN ────────────────────────────────────────────────────────────────
with left_col:
    # Project Manager
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Project Manager</div>', unsafe_allow_html=True)
    st.markdown("▼ **Project Name:**", unsafe_allow_html=False)

    # Projects from projects.json
    for proj in _proj_data.get("projects", []):
        pid = proj["id"]
        selected = st.session_state["selected_project"] == pid
        if st.button(
            f"▶ {pid}",
            key=f"proj_{pid}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            if st.session_state["selected_project"] != pid:
                st.session_state["selected_project"] = pid
                st.session_state["sim_data"] = None
                st.session_state["mode"] = "Modeling"
            st.rerun()

        for child in proj.get("children", []):
            cid = child["id"]
            st.markdown(
                f'<div class="tree-item">  ▷ {cid}</div>',
                unsafe_allow_html=True,
            )

    # Extra projects loaded from data/*.json (e.g. grid_inference_flow.json)
    existing_ids = {p["id"] for p in _proj_data.get("projects", [])}
    for pid in _extra_positions:
        if pid in existing_ids:
            continue   # already shown above
        selected = st.session_state["selected_project"] == pid
        if st.button(
            f"▶ {pid}",
            key=f"proj_{pid}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            if st.session_state["selected_project"] != pid:
                st.session_state["selected_project"] = pid
                st.session_state["sim_data"] = None
                st.session_state["mode"] = "Modeling"
            st.rerun()
        # List component names as tree children
        for comp_name in _extra_positions[pid]:
            label = _extra_positions[pid][comp_name].get("label", comp_name)
            st.markdown(
                f'<div class="tree-item">  ▷ {label}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Material Library
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Material Library</div>', unsafe_allow_html=True)
    st.markdown("▼ **Component Name:**", unsafe_allow_html=False)

    for comp_name, comp_data in _proj_data.get("components", {}).items():
        with st.expander(f"▼ {comp_name}", expanded=(comp_name == "CPU")):
            for child in comp_data.get("children", []):
                tdp  = child.get("TDP_W", "—")
                tmax = child.get("Tcase_max_C", "—")
                st.markdown(
                    f'<div class="tree-item">▷ {child["label"]}'
                    f'<br><small style="color:#666"> TDP: {tdp} W · Tcase_max: {tmax} °C</small></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT COLUMN ───────────────────────────────────────────────────────────────
with right_col:
    # Design Workspace panel
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Design Workspace</div>', unsafe_allow_html=True)

    # Mode switcher
    mode_options = ["Modeling", "Thermal Simulation", "3D"]
    current_mode_idx = mode_options.index(st.session_state["mode"])
    chosen_mode = st.radio(
        label="Mode",
        options=mode_options,
        index=current_mode_idx,
        horizontal=True,
        label_visibility="collapsed",
        disabled=st.session_state["locked"],
        key="mode_radio",
    )
    if chosen_mode != st.session_state["mode"]:
        st.session_state["mode"] = chosen_mode
        st.rerun()

    # Workspace rendering
    render_workspace()
    st.markdown("</div>", unsafe_allow_html=True)

    # Design Assistant panel
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    # Header row: title + mode toggle
    hdr_col, toggle_col = st.columns([2, 1])
    with hdr_col:
        st.markdown('<div class="panel-title">Design Assistant</div>', unsafe_allow_html=True)
    with toggle_col:
        chosen_chat_mode = st.radio(
            "chat_mode_radio",
            options=["Scripted", "AI Assistant"],
            index=0 if st.session_state["chat_mode"] == "Scripted" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="chat_mode_radio",
        )
        if chosen_chat_mode != st.session_state["chat_mode"]:
            st.session_state["chat_mode"] = chosen_chat_mode
            st.rerun()

    # Chat history display
    history = st.session_state["chat_history"]
    if history:
        chat_html_parts = []
        for msg in history:
            role  = msg["role"]
            text  = msg["text"].replace("\n", "<br>")
            if role == "user":
                chat_html_parts.append(
                    f'<div style="text-align:right; color:#a0c4ff; '
                    f'margin:4px 0; font-size:0.85rem;">'
                    f'<b>You:</b> {text}</div>'
                )
            else:
                chat_html_parts.append(
                    f'<div style="text-align:left; color:#caffbf; '
                    f'margin:4px 0; font-size:0.84rem;">'
                    f'<b>Assistant:</b> {text}</div>'
                )

        chat_html = (
            '<div class="chat-scroll">'
            + "".join(chat_html_parts)
            + "</div>"
        )
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color:#555; font-size:0.82rem;">Chat history will appear here…</p>',
            unsafe_allow_html=True,
        )

    # Input area
    col_input, col_clear = st.columns([5, 1])
    with col_input:
        user_msg = st.chat_input(
            "請在此處輸入對話…",
            disabled=st.session_state["locked"],
            key="chat_input",
        )
    with col_clear:
        if st.button("Clear", use_container_width=True, key="clear_chat"):
            st.session_state["chat_history"] = []
            st.session_state["component_positions"] = {}
            st.session_state["sim_data"] = None
            st.session_state["mode"] = "Modeling"
            st.rerun()

    if user_msg:
        _handle_chat(user_msg)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
