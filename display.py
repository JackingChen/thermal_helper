"""
display.py
Renders FAQ results inside the chat conversation (Display window).

Architecture
────────────
render_chat_answer(results)  ← called from app.py inside an assistant bubble.

Each result is a collapsible card with text fields.  When a result carries
a 'figure' key (plotly Figure), the chart is rendered inline as a bordered
sub-window below the text — matching the wireframe design.

To add new display types, add a branch inside _render_result_card() for
the new `display_type` value.

Current display_type values:
  • (unset / "text") – default rich-text answer card
  • "3d_plot"        – text card + embedded plotly figure sub-window
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from data_loader import (
    COL_ID, COL_CATEGORY, COL_QUESTION,
    COL_REFERENCE, COL_ANSWER, COL_GUIDELINE, COL_SOURCE,
)


def render_chat_answer(
    results: list[dict],
    show_score: bool = True,
    backend_result=None,
) -> None:
    """
    Render a list of retrieved FAQ results inside an assistant chat bubble.
    Each result gets a collapsible card.  If a BackendResult is provided,
    its figure is rendered as an inline sub-window after the cards.
    """
    for result in results:
        _render_result_card(result, show_score=show_score)

    # Backend sub-window (placement / thermal / future backends)
    if backend_result is not None:
        _render_backend_subwindow(backend_result)


# ── Single result card ────────────────────────────────────────────────────────

def _render_result_card(result: dict, show_score: bool) -> None:
    """Render one FAQ result card.  Inline figure sub-window when present."""
    score = result.get("score", 0.0)
    rank = result.get("rank", 1)
    display_type = result.get("display_type", "text")

    score_badge = f" · `{score:.2f}`" if show_score else ""
    header = f"#{rank} · [{result[COL_ID]}] {result[COL_CATEGORY]}{score_badge}"

    with st.expander(header, expanded=(rank == 1)):
        # Question
        st.markdown(f"**❓ {result[COL_QUESTION]}**")
        st.divider()

        # Text answer columns
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**📖 Problem Reference**")
            st.write(result.get(COL_REFERENCE) or "—")
            st.markdown("**✅ Expert Answer**")
            st.write(result.get(COL_ANSWER) or "—")
        with col_b:
            st.markdown("**📐 Technical Guideline**")
            st.write(result.get(COL_GUIDELINE) or "—")
            st.markdown("**🔗 Source**")
            st.write(result.get(COL_SOURCE) or "—")

        # Inline figure sub-window (shown when result directly carries a figure)
        fig: go.Figure | None = result.get("figure")
        if display_type == "3d_plot" or fig is not None:
            _render_figure_subwindow(title="📊 Visualisation", fig=fig)


# ── Backend sub-window ────────────────────────────────────────────────────────

def _render_backend_subwindow(backend_result) -> None:
    """
    Render a BackendResult (from backends/thermal.py, backends/placement.py, etc.)
    as a bordered inline sub-window inside the chat bubble.
    """
    title = backend_result.title or "📊 Computation Result"
    st.divider()
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if backend_result.summary:
            st.caption(backend_result.summary)
        if backend_result.figure is not None:
            st.plotly_chart(backend_result.figure, use_container_width=True)
        else:
            st.info("Backend computation result will appear here once implemented.")


# ── Generic figure sub-window ─────────────────────────────────────────────────

def _render_figure_subwindow(title: str, fig: go.Figure | None) -> None:
    """
    Render a Plotly figure as a bordered sub-window inside the chat bubble.
    """
    st.divider()
    st.markdown(f"**{title}**")
    with st.container(border=True):
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🛸 A 3-D visualisation will be rendered here in a future update.")
