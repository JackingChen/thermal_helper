"""
backends/
─────────
Plugin directory for compute backends.

Each backend module exposes a single entry-point function:

    run(context: dict) -> BackendResult

where `context` is the matched FAQ row dict (from the retriever) merged
with any extra parameters the caller provides.

BackendResult is defined here and imported by display.py to render the
result as an inline sub-window inside the chat bubble.

To register a new backend:
  1. Create backends/<name>.py and implement `run(context) -> BackendResult`.
  2. Add the trigger keywords to KEYWORD_MAP below.
  3. display.py will automatically call and render it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import plotly.graph_objects as go


@dataclass
class BackendResult:
    """
    Returned by every backend's `run()` function.

    Attributes
    ──────────
    title       : Short label shown above the sub-window.
    figure      : Optional Plotly figure to render.
    summary     : Text summary of the computation result.
    metadata    : Arbitrary extra data (tables, arrays, etc.) for future use.
    """
    title: str = ""
    figure: go.Figure | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Keyword → backend name mapping ───────────────────────────────────────────
# If any keyword appears in the matched FAQ question or category (case-insensitive),
# the corresponding backend is triggered automatically.
KEYWORD_MAP: dict[str, str] = {
    # thermal keywords
    "熱": "thermal",
    "temperature": "thermal",
    "thermal": "thermal",
    "散熱": "thermal",
    "heat": "thermal",
    "rth": "thermal",
    "junction": "thermal",
    # placement keywords
    "placement": "placement",
    "layout": "placement",
    "擺放": "placement",
    "佈局": "placement",
    "component": "placement",
    "opening": "placement",
    "開孔": "placement",
}


def resolve_backend(question: str, category: str) -> str | None:
    """
    Return the backend name triggered by question/category text, or None.
    First match wins (thermal takes precedence over placement when tied).
    """
    text = (question + " " + category).lower()
    for keyword, backend in KEYWORD_MAP.items():
        if keyword.lower() in text:
            return backend
    return None
