"""
backends/thermal.py
────────────────────
Thermal computation backend (stub / future implementation).

Entry point
───────────
    run(context: dict) -> BackendResult

`context` keys (from FAQ retriever row + extras):
    核心設計問題 (Question)         : str
    技術參考背景 (Problem Reference) : str
    答案技術指標/設計準則 (Technical Guideline) : str
    ... (all FAQ columns, see data_loader.py)
    params : dict   (optional caller-supplied parameters)

What this backend will do (TODO)
──────────────────────────────────
  - Accept thermal parameters (e.g. TDP, Rth values, material properties)
  - Run a simplified RC-network thermal simulation
  - Build a Plotly figure: temperature distribution, Rth chart, etc.
  - Return a BackendResult with the figure + numeric summary

Current state: returns a clearly-labelled placeholder figure so the
subwindow renders immediately without errors.
"""
from __future__ import annotations

import plotly.graph_objects as go

from backends import BackendResult

# ── Component thermal defaults ─────────────────────────────────────────────────
# Default steady-state temperatures (°C) used for Thermal 3D colouring.
# Keys are lowercase component names (matched case-insensitively in app.py).
# NOTE: for the heatsink project specifically, temperatures are overridden by
# the PINN inference result inside thermal_sim.py (pinn_heatsink module).
COMPONENT_TEMPS: dict[str, float] = {
    "cpu":      80.0,
    "heatsink": 55.0,
    "fan":      35.0,
    "ssd":      50.0,
    "ddr":      45.0,
    "ddr4":     45.0,
    "ddr5":     45.0,
    "pcie":     40.0,
    "vrm":      65.0,
    "power":    45.0,
}
TEMP_VMIN: float = 20.0
TEMP_VMAX: float = 120.0


def run(context: dict) -> BackendResult:
    """
    Thermal backend entry point.

    Parameters
    ----------
    context : dict
        Merged FAQ row + optional caller params under key 'params'.

    Returns
    -------
    BackendResult
        title   : section label for the subwindow
        figure  : Plotly figure (placeholder until real solver is wired in)
        summary : text description of the result
    """
    question = context.get("核心設計問題 (Question)", "")
    guideline = context.get("答案技術指標/設計準則 (Technical Guideline)", "")
    params: dict = context.get("params", {})

    # ── Placeholder visualisation ─────────────────────────────────────────────
    # TODO: replace with actual thermal solver output
    fig = _build_placeholder_figure(question, params)

    summary = (
        "🔧 Thermal backend is reserved for future implementation.\n"
        f"Triggered by: {question!r}\n"
        f"Guideline hint: {guideline[:120] if guideline else 'n/a'}"
    )

    return BackendResult(
        title="🌡️ Thermal Simulation",
        figure=fig,
        summary=summary,
        metadata={"params": params, "question": question},
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_placeholder_figure(question: str, params: dict) -> go.Figure:
    """
    Returns a labelled placeholder figure.
    Replace the body of this function with real solver output.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=(
            "🌡️ <b>Thermal Simulation</b><br>"
            "Placeholder – solver not yet implemented.<br><br>"
            f"<i>Question:</i> {question[:80] if question else 'n/a'}"
        ),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font={"size": 14},
        align="center",
    )
    fig.update_layout(
        title="Thermal Analysis (stub)",
        xaxis_visible=False,
        yaxis_visible=False,
        height=320,
        paper_bgcolor="#fff8f0",
    )
    return fig
