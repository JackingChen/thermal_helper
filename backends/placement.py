"""
backends/placement.py
──────────────────────
Component placement computation backend (stub / future implementation).

Entry point
───────────
    run(context: dict) -> BackendResult

`context` keys (from FAQ retriever row + extras):
    核心設計問題 (Question)          : str
    技術參考背景 (Problem Reference)  : str
    答案技術指標/設計準則 (Technical Guideline) : str
    ... (all FAQ columns, see data_loader.py)
    params : dict   (optional caller-supplied parameters)

What this backend will do (TODO)
──────────────────────────────────
  - Accept PCB / chassis geometry and component list
  - Run a placement optimisation or rule-check
  - Build a Plotly 3-D figure: component layout colour view (like the
    PCB Component Layout wireframe in the design spec)
  - Return a BackendResult with the figure + placement report

Current state: returns a clearly-labelled placeholder 3-D figure.
"""
from __future__ import annotations

import plotly.graph_objects as go

from backends import BackendResult


def run(context: dict) -> BackendResult:
    """
    Placement backend entry point.

    Parameters
    ----------
    context : dict
        Merged FAQ row + optional caller params under key 'params'.

    Returns
    -------
    BackendResult
    """
    question = context.get("核心設計問題 (Question)", "")
    params: dict = context.get("params", {})

    fig = _build_placeholder_3d(question, params)

    summary = (
        "🔧 Placement backend is reserved for future implementation.\n"
        f"Triggered by: {question!r}"
    )

    return BackendResult(
        title="📦 Component Placement",
        figure=fig,
        summary=summary,
        metadata={"params": params, "question": question},
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_placeholder_3d(question: str, params: dict) -> go.Figure:
    """
    Returns a labelled placeholder 3-D scatter figure.
    Replace with real placement solver output (e.g. box meshes per component).
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n = 12
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 8, n)
    z = rng.uniform(0, 3, n)
    labels = [f"Comp-{i+1}" for i in range(n)]
    colors = rng.integers(0, 10, n)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=10, color=colors, colorscale="Turbo", opacity=0.85),
    )])

    fig.update_layout(
        title="PCB Component Layout – Colour Coded View (stub)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
        ),
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
