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
from PIL import Image as _PILImage

from backends.placement import apply_placement as _placement_action
from backends.placement import apply_layout as _layout_action
from backends.placement import execute_instruction as _exec_placement
from backends.placement import check_overlaps as _check_overlaps
from backends.thermal import COMPONENT_TEMPS as _COMPONENT_TEMPS
from backends.thermal import TEMP_VMIN as _TEMP_VMIN
from backends.thermal import TEMP_VMAX as _TEMP_VMAX
from backends.llm_backend import call_azure_llm
from chat_responses import route_message
from qa_loader import load_qa
from thermal_sim import get_component_positions, run_simulation
import session_config as _session_cfg

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_DATA_DIR           = _HERE / "data"
_PROJECT_RULE_DIR   = _HERE / "assets" / "project_rule"
_PROJECTS_JSON      = _HERE / "data" / "projects.json"
_ASSET_3D           = _HERE / "assets" / "ThermalOnPCB.png"
_ASSET_FLOW_GEO     = _HERE / "assets" / "vti_geometry_3d.png"
_ASSET_FLOW_LINES   = _HERE / "assets" / "vti_flow_streamlines.png"
_ASSET_FLOW_HTML    = _HERE / "assets" / "model_3d_demo.html"
_ICON_DIR           = _HERE / "assets" / "icons"

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
    # .panel-card {
    #     background: #16213e;
    #     border: 1px solid #0f3460;
    #     border-radius: 8px;
    #     padding: 12px 14px;
    #     margin-bottom: 10px;
    #     min-height: 200px;
    # }

    # /* Design Workspace panel — user-resizable vertically */
    # .panel-workspace {
    #     background: #16213e;
    #     border: 1px solid #0f3460;
    #     border-radius: 8px;
    #     padding: 12px 14px;
    #     margin-bottom: 10px;
    #     min-height: 300px;
    #     resize: vertical;
    #     overflow: auto;
    # }
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
    Output : {comp_name: {x (centre), y (centre), w, h, label, rotated (bool), material}}
             plus special key "_board": {w, h} derived from edge_right / edge_top markers.

    Rotation encoding:
        angle = (rotated + mapping * 2) * 90 °CCW
        90° / 270° swaps w and h (axis-aligned bounding box).
    """
    positions: dict = {}
    board_w: float = 400.0
    board_h: float = 400.0

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
            "material": str(item.get("material", "metal")).lower(),
        }

    positions["_board"] = {"w": board_w, "h": board_h}
    return positions


@st.cache_data(show_spinner=False)
def _load_extra_positions() -> dict:
    """
    Load per-project geometry presets from:
    1) data/<project>/<project>.json (legacy)
    2) assets/project_rule/<project>/initial.json (preferred)

    Supports both new array format (list) and legacy dict format.
    """
    merged: dict = {}

    def _load_one(project: str, json_file: Path) -> None:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                merged[project] = _parse_geometry_array(data)
            elif isinstance(data, dict):
                merged[project] = data
        except Exception:
            pass

    for proj_dir in sorted(_DATA_DIR.iterdir()):
        if not proj_dir.is_dir():
            continue
        json_file = proj_dir / f"{proj_dir.name}.json"
        if not json_file.exists():
            continue
        _load_one(proj_dir.name, json_file)

    if _PROJECT_RULE_DIR.exists():
        for proj_dir in sorted(_PROJECT_RULE_DIR.iterdir()):
            if not proj_dir.is_dir():
                continue
            initial_file = proj_dir / "initial.json"
            if not initial_file.exists():
                continue
            _load_one(proj_dir.name, initial_file)

    return merged


@st.cache_data(show_spinner=False)
def _load_qa_cached() -> list:
    return load_qa()


@st.cache_data(show_spinner=False)
def _load_icons() -> dict:
    """
    Load all PNG icons from assets/icons/ keyed by lowercase stem.
    Returns uint8 RGBA numpy arrays.
    """
    icons: dict = {}
    if _ICON_DIR.exists():
        for p in sorted(_ICON_DIR.glob("*.png")):
            try:
                icons[p.stem.lower()] = np.array(_PILImage.open(p).convert("RGBA"))
            except Exception:
                pass
    return icons


def _find_icon(name: str, label: str, icons: dict):
    """Case-insensitive match: exact then substring (e.g. 'ddr' matches 'DDR-1')."""
    for cand in [name.lower(), label.lower()]:
        if cand in icons:
            return icons[cand]
        for stem, img in icons.items():
            if stem in cand:
                return img
    return None


def _img_to_surface_colorscale(arr: np.ndarray):
    """
    Convert a uint8 RGBA array to (surfacecolor_norm, colorscale) for plotly Surface.
    Packs R,G,B into a single int per pixel, builds a tight per-colour colorscale.
    """
    r = arr[:, :, 0].astype(np.int64)
    g = arr[:, :, 1].astype(np.int64)
    b = arr[:, :, 2].astype(np.int64)
    packed = r * 65536 + g * 256 + b
    p_min, p_max = int(packed.min()), int(packed.max())
    if p_min == p_max:
        p_max = p_min + 1
    norm = (packed - p_min).astype(float) / (p_max - p_min)
    cscale = []
    for v in np.unique(packed.ravel()):
        v_n = float(v - p_min) / (p_max - p_min)
        cscale.append([v_n, f"rgb({(v>>16)&0xFF},{(v>>8)&0xFF},{v&0xFF})"])
    if cscale[0][0] > 0:
        cscale.insert(0, [0.0, cscale[0][1]])
    if cscale[-1][0] < 1.0:
        cscale.append([1.0, cscale[-1][1]])
    return norm, cscale


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
        "preset_stage":        "initial",    # "initial" | "optimized" | "optimized_thermal"
        "locked":              False,        # disable controls mid-demo
        "thermal_3d":          False,        # True when 3D was triggered from Thermal Simulation mode
        "chat_feedback":       {},           # {msg_index: "like" | "dislike"}
        # OpenClaw session config showcase
        "session_cfg":         _session_cfg.default_config(),
        "show_config":         False,
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
        # 1. Primary: assets/project_rule/<project>/initial.json and data/<project>/<project>.json
        pos = _extra_positions.get(project)
        # 2. Fallback: projects.json default_component_positions (legacy)
        if pos is None:
            defaults = _proj_data.get("default_component_positions", {})
            pos = defaults.get(project)
        st.session_state["component_positions"][project] = copy.deepcopy(pos or {})
    return st.session_state["component_positions"][project]


def _ensure_sim(project: str | None, positions: dict | None = None) -> np.ndarray | None:
    if not project:
        return None
    if st.session_state["sim_data"] is None:
        board_info = (positions or {}).get("_board", {})
        board_w = float(board_info.get("w", 400))
        board_h = float(board_info.get("h", 400))
        st.session_state["sim_data"] = run_simulation(
            project,
            live_positions=positions,
            board_w=board_w,
            board_h=board_h,
        )
    return st.session_state["sim_data"]


def _material_style(material: str) -> tuple[str, str]:
    """Return (fill, edge) colour for a given material name."""
    key = str(material or "metal").strip().lower()
    palette = {
        "metal": ("#4f5d75", "#d7deea"),
        "copper": ("#d97706", "#ffd8a8"),
        "aluminum": ("#5fa8ff", "#d8ecff"),
        "ceramic": ("#f6c453", "#fff1c8"),
        "graphite": ("#4b5563", "#cfd4dc"),
        "plastic": ("#2cb67d", "#c7f3df"),
    }
    if key in palette:
        return palette[key]
    return ("#b37feb", "#f0ddff")


def _material_thermal_edge(material: str) -> tuple[str, int]:
    """Return (edge_color, line_width) for thermal view — high-contrast against hot heatmap.

    Conductivity-inspired hue: high-k materials (copper, aluminum) → cool cyan/blue;
    low-k materials (plastic, ceramic) → warm green/yellow; unknown → white.
    """
    key = str(material or "metal").strip().lower()
    # (color, width)
    palette = {
        "copper":   ("#00e5ff", 3),   # bright cyan  — excellent conductor
        "aluminum": ("#3b82f6", 3),   # vivid blue   — good conductor
        "metal":    ("#94a3b8", 2),   # steel-grey   — moderate
        "graphite": ("#a78bfa", 2),   # violet       — moderate anisotropic
        "ceramic":  ("#facc15", 2),   # bright amber — lower conductivity
        "plastic":  ("#4ade80", 2),   # bright green — poor conductor
    }
    return palette.get(key, ("#ffffff", 2))


def _load_optimized_layout(project: str, suffix: str) -> dict | None:
    """Load and parse optimized geometry from assets/project_rule/<project>/...json."""
    opt_path = _PROJECT_RULE_DIR / project / f"{project}{suffix}.json"
    try:
        with open(opt_path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not load optimized preset: {exc}")
        return None
    return _parse_geometry_array(data) if isinstance(data, list) else data


def _apply_optimized_preset(project: str, profile_positions: dict | None) -> None:
    """Overwrite session component positions (geometry + material) from an optimized preset."""
    if not profile_positions:
        return

    # Build new positions dict: take geometry+material from preset, fall back to current for unknowns
    current = _get_positions(project)
    merged: dict = {"_board": profile_positions.get("_board", current.get("_board", {}))}
    for name, target in profile_positions.items():
        if name == "_board":
            continue
        merged[name] = {
            "x":        target.get("x", current.get(name, {}).get("x", 0.0)),
            "y":        target.get("y", current.get(name, {}).get("y", 0.0)),
            "w":        target.get("w", current.get(name, {}).get("w", 10.0)),
            "h":        target.get("h", current.get(name, {}).get("h", 10.0)),
            "label":    target.get("label", current.get(name, {}).get("label", name)),
            "rotated":  target.get("rotated", current.get(name, {}).get("rotated", False)),
            "material": str(target.get("material", current.get(name, {}).get("material", "metal"))).lower(),
        }
    # Preserve any components present in current but missing from the preset
    for name, comp in current.items():
        if name not in merged:
            merged[name] = comp

    st.session_state["component_positions"][project] = merged
    st.session_state["sim_data"] = None


# ── Workspace renderers ────────────────────────────────────────────────────────

def _render_modeling(project: str, positions: dict) -> None:
    """2-D PCB component layout — interactive Plotly figure."""
    board_info = positions.get("_board", {})
    board_w = float(board_info.get("w", 400))
    board_h = float(board_info.get("h", 400))

    render_positions = {k: v for k, v in positions.items() if k != "_board"}
    non_metal_materials: set[str] = set()

    margin_x, margin_y = board_w * 0.10, board_h * 0.10
    shapes = [
        # PCB board fill
        dict(type="rect", x0=0, y0=0, x1=board_w, y1=board_h,
             line=dict(color="#2a9d8f", width=2), fillcolor="#0a1628", layer="below"),
        # Outer dashed workspace boundary
        dict(type="rect", x0=-margin_x, y0=-margin_y,
             x1=board_w + margin_x, y1=board_h + margin_y,
             line=dict(color="#444466", width=1, dash="dash"), fillcolor="rgba(0,0,0,0)"),
    ]
    annotations = []

    sorted_comps = sorted(render_positions.items(),
                          key=lambda kv: kv[1]["w"] * kv[1]["h"], reverse=True)
    for name, comp in sorted_comps:
        w = comp["h"] if comp.get("rotated") else comp["w"]
        h = comp["w"] if comp.get("rotated") else comp["h"]
        x0_c, y0_c = comp["x"] - w / 2, comp["y"] - h / 2
        x1_c, y1_c = comp["x"] + w / 2, comp["y"] + h / 2
        material = str(comp.get("material", "metal")).lower()
        fc, ec = _material_style(material)
        if material != "metal":
            non_metal_materials.add(material)
        label = comp.get("label", name.upper())
        shapes.append(dict(
            type="rect", x0=x0_c, y0=y0_c, x1=x1_c, y1=y1_c,
            line=dict(color=ec, width=1.5), fillcolor=fc, opacity=0.85,
        ))
        annotations.append(dict(
            x=comp["x"], y=comp["y"], text=label,
            showarrow=False, font=dict(color="white", size=9),
            bgcolor="rgba(0,0,0,0)", xanchor="center", yanchor="middle",
        ))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[board_w / 2], y=[board_h / 2],
        mode="markers", marker=dict(opacity=0),
        showlegend=False, hoverinfo="skip",
    ))
    for material in sorted(non_metal_materials):
        fill, edge = _material_style(material)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=11,
                color=fill,
                line=dict(color=edge, width=1.5),
            ),
            name=f"Material: {material}",
            showlegend=True,
            hoverinfo="skip",
        ))
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            range=[-margin_x - 2, board_w + margin_x + 2],
            title="X (mm)", color="#adb5bd",
            gridcolor="#1d3557", gridwidth=0.5, zeroline=False,
            tickfont=dict(color="#adb5bd"),
        ),
        yaxis=dict(
            range=[board_h + margin_y + 2, -margin_y - 2],
            title="Y (mm)", color="#adb5bd",
            gridcolor="#1d3557", gridwidth=0.5, zeroline=False,
            tickfont=dict(color="#adb5bd"),
            scaleanchor="x", scaleratio=1,
        ),
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
        title=dict(
            text=f"{project}  |  Mode: Modeling  [{board_w:.0f} × {board_h:.0f} mm]",
            font=dict(color="#adb5bd", size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
        legend=dict(
            bgcolor="rgba(10,22,40,0.95)",
            bordercolor="#74c0fc",
            borderwidth=1,
            font=dict(color="#f8fbff", size=12),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    if non_metal_materials:
        st.caption("Legend: colored components indicate material has changed (default material is metal).")

    # ── Overlap & out-of-bounds warnings ──────────────────────────────────────
    render_only = {k: v for k, v in positions.items() if k != "_board"}
    overlaps = _check_overlaps(render_only)
    if overlaps:
        names = ", ".join(f"**{a}** & **{b}**" for a, b in overlaps[:4])
        st.warning(f"⚠️ Component overlap detected: {names}")
    out_of_bounds = []
    for name, comp in render_only.items():
        cw = comp["h"] if comp.get("rotated") else comp["w"]
        ch = comp["w"] if comp.get("rotated") else comp["h"]
        if (comp["x"] - cw / 2 < 0 or comp["x"] + cw / 2 > board_w
                or comp["y"] - ch / 2 < 0 or comp["y"] + ch / 2 > board_h):
            out_of_bounds.append(name)
    if out_of_bounds:
        st.warning(f"⚠️ Component(s) outside PCB boundary: **{', '.join(out_of_bounds)}**")


def _render_thermal(project: str, positions: dict) -> None:
    """2-D thermal heatmap — interactive Plotly figure."""
    board_info = positions.get("_board", {})
    board_w = float(board_info.get("w", 400))
    board_h = float(board_info.get("h", 400))

    sim = _ensure_sim(project, positions)
    if sim is None:
        st.warning("No simulation data — select a project first.")
        return

    render_positions = {k: v for k, v in positions.items() if k != "_board"}
    sim_pos = get_component_positions(
        project,
        live_positions=positions,
        component_temps=_COMPONENT_TEMPS,
        board_w=board_w,
        board_h=board_h,
    )

    _nrows, _ncols = sim.shape
    traces = [go.Heatmap(
        z=sim,
        x=np.linspace(0, board_w, _ncols),
        y=np.linspace(0, board_h, _nrows),
        colorscale="hot",
        zmin=20, zmax=120,
        colorbar=dict(
            title=dict(text="°C", font=dict(color="#adb5bd")),
            tickfont=dict(color="#adb5bd"),
            thickness=12, len=0.6,
        ),
        showscale=True,
        hovertemplate="x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>T: %{z:.1f} °C<extra></extra>",
    )]

    shapes = []
    for name, comp in render_positions.items():
        w = comp["h"] if comp.get("rotated") else comp["w"]
        h = comp["w"] if comp.get("rotated") else comp["h"]
        x0_c, y0_c = comp["x"] - w / 2, comp["y"] - h / 2
        x1_c, y1_c = comp["x"] + w / 2, comp["y"] + h / 2
        # High-contrast material edge color for thermal view
        material = str(comp.get("material", "metal")).lower()
        ec, lw = _material_thermal_edge(material)
        shapes.append(dict(
            type="rect", x0=x0_c, y0=y0_c, x1=x1_c, y1=y1_c,
            line=dict(color=ec, width=lw), fillcolor="rgba(0,0,0,0)",
        ))

    annotations = []
    for name, info in sim_pos.items():
        annotations.append(dict(
            x=info["x"], y=info["y"],
            text=f"{info['T']:.1f}°C<br>{name.upper()}",
            showarrow=False,
            font=dict(color="white", size=9),
            bgcolor="rgba(51,51,51,0.8)",
            bordercolor="rgba(255,255,255,0.33)",
            borderwidth=1,
            xanchor="center", yanchor="middle",
        ))

    margin = board_w * 0.02
    fig = go.Figure(data=traces)
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            range=[-margin, board_w + margin],
            title="X (mm)", color="#adb5bd",
            gridcolor="#1d3557", zeroline=False,
            tickfont=dict(color="#adb5bd"),
        ),
        yaxis=dict(
            range=[board_h + margin, -margin],
            title="Y (mm)", color="#adb5bd",
            gridcolor="#1d3557", zeroline=False,
            tickfont=dict(color="#adb5bd"),
            scaleanchor="x", scaleratio=1,
        ),
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
        title=dict(
            text=f"{project}  |  Thermal Simulation (20–120 °C)",
            font=dict(color="#adb5bd", size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_3d(project: str | None, positions: dict, thermal: bool = False) -> None:
    """Interactive 3-D plotly PCB component visualisation.

    Parameters
    ----------
    thermal : bool
        When True, colour components by temperature (from _COMPONENT_TEMPS)
        using the hot colourmap instead of fixed colours.
    """

    # ── Board dimensions ──────────────────────────────────────────────────────
    board_info = positions.get("_board", {})
    board_w = float(board_info.get("w", 400))
    board_h = float(board_info.get("h", 400))

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
    def _box(x0, y0, z0, x1, y1, z1, color, name, temp=None):
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
        temp_line = f"T: {temp:.1f} °C<br>" if temp is not None else ""
        return go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=i_, j=j_, k=k_,
            color=color, opacity=0.88,
            name=name, showlegend=True,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"{temp_line}"
                f"x: {x0:.1f}–{x1:.1f} mm<br>"
                f"y: {y0:.1f}–{y1:.1f} mm<br>"
                f"z: {z0:.1f}–{z1:.1f} mm<extra></extra>"
            ),
        )

    traces: list[go.BaseTraceType] = []

    # PCB board (thin green slab)
    traces.append(_box(0, 0, 0, board_w, board_h, 1.5, "#0a3622", "PCB Board"))

    # Thermal layer: a coloured 2-D heatmap slab elevated just above the PCB
    if thermal:
        _sim_layer = _ensure_sim(project, positions)
        if _sim_layer is not None:
            _SLAB_Z0, _SLAB_Z1 = 1.5, 3.5   # 2 mm thick slab
            # Down-sample to keep plotly fast (max 80 × 80)
            _ds = max(1, max(_sim_layer.shape[0] // 80, _sim_layer.shape[1] // 80))
            _layer = _sim_layer[::_ds, ::_ds].astype(float)
            _nrows, _ncols = _layer.shape
            _Xs = np.linspace(0, board_w, _ncols)
            _Ys = np.linspace(0, board_h, _nrows)
            _XX, _YY = np.meshgrid(_Xs, _Ys)
            # Top face of slab at z = SLAB_Z1
            _ZZ_top = np.full_like(_XX, _SLAB_Z1)
            traces.append(go.Surface(
                x=_XX, y=_YY, z=_ZZ_top,
                surfacecolor=_layer,
                colorscale="hot",
                cmin=_TEMP_VMIN, cmax=_TEMP_VMAX,
                showscale=True,
                colorbar=dict(
                    title=dict(text="°C", font=dict(color="#adb5bd")),
                    tickfont=dict(color="#adb5bd"),
                    thickness=12, len=0.5,
                ),
                opacity=0.75,
                name="Thermal Layer",
                showlegend=True,
            ))

    # Components
    icons = _load_icons()
    render_positions = {k: v for k, v in positions.items() if k != "_board"}

    # Gather per-component temps first so we can normalise against the local range
    _comp_temps = {
        name: _COMPONENT_TEMPS.get(name.lower(), 40.0)
        for name in render_positions
    }
    _local_min = min(_comp_temps.values()) if _comp_temps else _TEMP_VMIN
    _local_max = max(_comp_temps.values()) if _comp_temps else _TEMP_VMAX
    if _local_max == _local_min:
        _local_max = _local_min + 1.0

    for name, comp in render_positions.items():
        cx, cy = comp["x"], comp["y"]
        # Respect runtime rotation (swap w/h)
        w = comp["h"] if comp.get("rotated") else comp["w"]
        h = comp["w"] if comp.get("rotated") else comp["h"]
        z0, z1 = _Z.get(name, (1.5, 6.0))
        label  = comp.get("label", name.upper())
        temp  = _comp_temps[name]
        if thermal:
            norm = max(0.0, min(1.0, (temp - _local_min) / (_local_max - _local_min)))
            rgba = plt.cm.plasma(norm)
            color = f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"
        else:
            material = str(comp.get("material", "metal")).lower()
            color, _edge = _material_style(material)
        traces.append(_box(cx - w/2, cy - h/2, z0, cx + w/2, cy + h/2, z1, color, label, temp=temp))

        # Icon texture on top face
        icon_arr = _find_icon(name, label, icons)
        if icon_arr is not None and w > 0 and h > 0:
            nc = max(2, min(20, int(w)))
            nr = max(2, min(20, int(h)))
            resized = np.array(_PILImage.fromarray(icon_arr).resize((nc, nr)))
            Xs = np.linspace(cx - w/2, cx + w/2, nc)
            Ys = np.linspace(cy - h/2, cy + h/2, nr)
            XX, YY = np.meshgrid(Xs, Ys)
            ZZ = np.full_like(XX, z1 + 0.05)
            surf_color, cscale = _img_to_surface_colorscale(resized)
            traces.append(go.Surface(
                x=XX, y=YY, z=ZZ,
                surfacecolor=surf_color,
                colorscale=cscale,
                showscale=False,
                name=f"{label} (icon)",
                showlegend=False,
                opacity=0.95,
            ))

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
        legend=dict(
            bgcolor="rgba(10,22,40,0.95)",
            bordercolor="#74c0fc",
            borderwidth=1,
            font=dict(color="#f8fbff", size=12),
        ),
        title=dict(
            text=f"{project or ''}  |  {'Thermal 3D Preview' if thermal else '3D Preview'}",
            font=dict(color="#adb5bd", size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# def _render_flow_geometry() -> None:
#     """CFD geometry views generated from grid_inference_flow.vti."""
#     if not _ASSET_FLOW_GEO.exists():
#         st.warning(
#             "Flow geometry assets not yet generated. "
#             "Run **vti_render.ipynb** inside the Docker container first."
#         )
#         return

#     tab_iso, tab_stream, tab_interactive = st.tabs(
#         ["Isometric View", "Streamlines", "Interactive 3D"]
#     )

#     with tab_iso:
#         if _ASSET_FLOW_GEO.exists():
#             st.image(str(_ASSET_FLOW_GEO),
#                      caption="Surface geometry coloured by pressure (Pa)")
#         else:
#             st.info("Geometry image not found.")

#     with tab_stream:
#         if _ASSET_FLOW_LINES.exists():
#             st.image(str(_ASSET_FLOW_LINES),
#                      caption="Velocity streamlines seeded from inlet face")
#         else:
#             st.info("Streamline image not found — re-run cell 6 in vti_render.ipynb.")

#     with tab_interactive:
#         if _ASSET_FLOW_HTML.exists():
#             with open(_ASSET_FLOW_HTML, "r", encoding="utf-8") as f:
#                 html_content = f.read()
#             st.components.v1.html(html_content, height=600, scrolling=False)
#         else:
#             st.info("Interactive HTML not found — re-run cell 7 in vti_render.ipynb.")


def render_workspace() -> None:
    project   = st.session_state["selected_project"]
    mode      = st.session_state["mode"]
    positions = _get_positions(project)

    if mode == "3D":
        _render_3d(project, positions, thermal=st.session_state.get("thermal_3d", False))
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


def _apply_optimized(project: str) -> None:
    """Apply passive-thermal preset: update geometry + material; sim re-derives temps from physics."""
    optimized_positions = _load_optimized_layout(project, "_optimized")
    _apply_optimized_preset(project, optimized_positions)
    st.session_state["preset_stage"] = "optimized"


def _apply_optimized_thermal(project: str) -> None:
    """Apply full thermal-optimized preset: update geometry + material; sim re-derives temps from physics."""
    optimized_positions = _load_optimized_layout(project, "_optimized_4_thermal")
    _apply_optimized_preset(project, optimized_positions)
    st.session_state["preset_stage"] = "optimized_thermal"


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
                project or "",
                st.session_state.get("session_cfg", {}),
                st.session_state.get("preset_stage", "initial"),
            )
            status_box.markdown("_Parsing response…_")
            status_box.empty()
    else:
        response_text, action = route_message(user_input, mode, _qa_data)

    # Execute workspace actions
    if isinstance(action, tuple) and project:
        # Composite action: preset + mode switch (returned by LLM when both are needed)
        preset_act, mode_act = action[0], action[1]
        if preset_act == "apply_optimized":
            _apply_optimized(project)
        elif preset_act == "apply_optimized_thermal":
            _apply_optimized_thermal(project)
        if mode_act == "switch_thermal":
            st.session_state["thermal_3d"] = False
            st.session_state["mode"] = "Thermal Simulation"
        elif mode_act == "switch_3d":
            st.session_state["thermal_3d"] = (st.session_state.get("mode") == "Thermal Simulation")
            st.session_state["mode"] = "3D"
    elif isinstance(action, dict) and project:
        _apply_instruction(project, action)
    elif action == "apply_optimized" and project:
        _apply_optimized(project)
    elif action == "apply_optimized_thermal" and project:
        _apply_optimized_thermal(project)
    elif action == "switch_3d":
        # Remember if we came from Thermal mode so 3D renders with temperature colours
        st.session_state["thermal_3d"] = (st.session_state.get("mode") == "Thermal Simulation")
        st.session_state["mode"] = "3D"
    elif action == "switch_thermal":
        st.session_state["thermal_3d"] = False
        st.session_state["mode"] = "Thermal Simulation"

    # ── Log subagent invocations to OpenClaw session config ───────────────────
    cfg = st.session_state.get("session_cfg")
    if cfg is not None:
        if chat_mode == "AI Assistant":
            _session_cfg.log_subagent(
                cfg, "llm_reasoning_agent",
                f"user: {user_input[:80]}",
                "Returned JSON action from Azure OpenAI",
            )
        if isinstance(action, tuple):
            preset_act, mode_act = action[0], action[1]
            _session_cfg.log_subagent(
                cfg, "placement_agent",
                preset_act,
                "Loaded and applied preset layout from project_rule/ + mode switch",
            )
            if mode_act == "switch_thermal":
                _session_cfg.log_subagent(
                    cfg, "thermal_sim_agent",
                    "mode switch → Thermal Simulation",
                    "Queued thermal field generation (100 × 150 grid)",
                )
        elif isinstance(action, dict):
            steps = len(action.get("steps", []))
            _session_cfg.log_subagent(
                cfg, "placement_agent",
                "move_sequence instruction from LLM",
                f"Applied {steps} placement step(s)",
            )
        elif action in ("apply_optimized", "apply_optimized_thermal"):
            _session_cfg.log_subagent(
                cfg, "placement_agent",
                action,
                "Loaded and applied preset layout from project_rule/",
            )
        elif action == "switch_thermal":
            _session_cfg.log_subagent(
                cfg, "thermal_sim_agent",
                "mode switch → Thermal Simulation",
                "Queued thermal field generation (100 × 150 grid)",
            )
        # FAQ-backed scripted responses route through rag_faq_agent
        if chat_mode == "Scripted" and action not in (
            "apply_placement", "apply_layout", "switch_3d", "switch_thermal",
        ):
            _session_cfg.log_subagent(
                cfg, "rag_faq_agent",
                f"query: {user_input[:80]}",
                "Scored FAQ rows by token overlap and returned best match",
            )

    st.session_state["chat_history"].append({"role": "assistant", "text": response_text})




# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════

# Top title bar
st.markdown(
    "<h3 style='color:#e94560; margin:0 0 8px 0; font-size:1.1rem; "
    "letter-spacing:0.05em;'>● Thermal Agent</h3>",
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
        # Collapsible component tree
        with st.expander("▶ Components", expanded=False):
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
    st.markdown('<div class="panel-workspace">', unsafe_allow_html=True)
    _stage = st.session_state.get("preset_stage", "initial")
    _stage_labels = {
        "initial":           ("⬜ Initial",           "#4f5d75"),
        "optimized":         ("🟡 Material Swapped",  "#d97706"),
        "optimized_thermal": ("🔵 Heat Pipes Added",  "#3b82f6"),
    }
    _stage_text, _stage_color = _stage_labels.get(_stage, ("⬜ Initial", "#4f5d75"))
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span class="panel-title">Design Workspace</span>'
        f'<span style="font-size:0.75rem;padding:2px 8px;border-radius:10px;'
        f'background:{_stage_color}22;border:1px solid {_stage_color};color:{_stage_color};'
        f'font-weight:600;letter-spacing:0.04em;">{_stage_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

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
        st.session_state["thermal_3d"] = False  # manual switch always resets thermal 3D
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
    feedback = st.session_state["chat_feedback"]
    if history:
        with st.container(height=260):
            for i, msg in enumerate(history):
                role = msg["role"]
                text = msg["text"].replace("\n", "<br>")
                if role == "user":
                    st.markdown(
                        f'<div style="text-align:right; color:#a0c4ff; '
                        f'margin:4px 0; font-size:0.85rem;">'
                        f'<b>You:</b> {text}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    msg_col, fb_col1, fb_col2 = st.columns([10, 1, 1])
                    with msg_col:
                        st.markdown(
                            f'<div style="text-align:left; color:#caffbf; '
                            f'margin:4px 0; font-size:0.84rem;">'
                            f'<b>Assistant:</b> {text}</div>',
                            unsafe_allow_html=True,
                        )
                    with fb_col1:
                        icon = "✅" if feedback.get(i) == "like" else "👍"
                        if st.button(icon, key=f"like_{i}", help="Helpful"):
                            st.session_state["chat_feedback"][i] = "like"
                            st.rerun()
                    with fb_col2:
                        icon = "❌" if feedback.get(i) == "dislike" else "👎"
                        if st.button(icon, key=f"dislike_{i}", help="Not helpful"):
                            st.session_state["chat_feedback"][i] = "dislike"
                            st.rerun()
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
            st.session_state["chat_feedback"] = {}
            st.session_state["component_positions"] = {}
            st.session_state["sim_data"] = None
            st.session_state["mode"] = "Modeling"
            st.session_state["thermal_3d"] = False
            st.rerun()

    if user_msg:
        _handle_chat(user_msg)
        st.rerun()

    # ── OpenClaw agent config controls ────────────────────────────────────────
    oc_col1, oc_col2 = st.columns(2)
    with oc_col1:
        if st.button(
            "💾 Save Advice",
            use_container_width=True,
            key="save_advice_btn",
            help="Save the last user message as a memory fact in the session config",
        ):
            history = st.session_state["chat_history"]
            last_user = next(
                (m["text"] for m in reversed(history) if m["role"] == "user"), None
            )
            if last_user:
                _session_cfg.add_memory(
                    st.session_state["session_cfg"],
                    last_user,
                    source="user_advice",
                )
                st.toast("Advice saved to session memory.")
            else:
                st.toast("No user message found to save.")
            st.rerun()
    with oc_col2:
        cfg_label = "🔧 Hide Config" if st.session_state["show_config"] else "🔧 Agent Config"
        if st.button(cfg_label, use_container_width=True, key="agent_config_btn",
                     help="View the current OpenClaw session config (skills, memory, subagents)"):
            st.session_state["show_config"] = not st.session_state["show_config"]
            st.rerun()

    if st.session_state["show_config"]:
        st.markdown(
            _session_cfg.as_markdown(st.session_state["session_cfg"]),
        )

    st.markdown("</div>", unsafe_allow_html=True)
