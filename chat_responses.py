"""
chat_responses.py
──────────────────
All scripted chat text and routing logic.

Public API
──────────
    route_message(user_input: str, mode: str) -> tuple[str, str | None]

Returns
───────
    (response_text, workspace_action)

workspace_action is one of:
    None               – no workspace change
    dict               – placement instruction (passed to execute_instruction)
    "switch_3d"        – swap workspace to 3D image
    "switch_thermal"   – switch mode to Thermal Simulation
"""
from __future__ import annotations

import re
from typing import Optional

from backends.placement import _PRESET_LAYOUT
from qa_loader import find_answer, load_qa


# ── Scripted response library ──────────────────────────────────────────────────

_STEP3_PLACEMENT = """\
**Impact Analysis:** Spacing reduced from 10 mm to 8 mm, airflow restricted by ~15–20%. CPU hotspot estimated to rise **+3 °C ~ +5 °C** under full load — risk of exceeding safe operating temperature.

**Passive Thermal Recommendations:**
① CPU Shift left 2 mm (X: 45.0 → 43.0 mm)
② Heatsink Rotate +90° to align with airflow direction, maintain clearance ≥10 mm

*Next Step: Type* ***Apply passive thermal approach*** *to execute all updates automatically.*\
"""

_STEP4_PLACEMENT_PROGRESS = """\
Applying passive thermal approach…
- Updating component positions
- Checking spacing constraints
- Rebuilding layout

✅ **Placement updated.**
Switch to 3D preview for validation?
*Type* ***3D*** *to switch to 3D preview mode automatically.*\
"""

_STEP5_3D_CONFIRM = "🖥️ **3D view ready.** Design Workspace switched to 3D preview."

_STEP6_THERMAL_MODE = """\
Switched to **Thermal Simulation** mode.
Displaying chassis 2D temperature map (colour scale: 20 – 120 °C).\
"""

_STEP7_SURFACE_TEMP = """\
**Based on IEC 62368-1 and common thermal design practices:**

- Accessible metal surfaces: typically below **~48–55 °C**
- Plastic surfaces: typically below **~60 °C**

**Note:** Actual limits depend on product category and user exposure conditions.

**Current design:**
- CPU hotspot exceeds typical safe surface temperature range

**Recommendation:** Prioritise hotspot reduction in CPU region.\
"""

_STEP8_RF_ROUTING = """\
Query classified as **RF / EMI-related** issue.
Switching to RF knowledge domain…
Identified keywords: *antenna interference*, *motor EMI*. Retrieving RF design guidelines…\
"""

_STEP9_RF_ANSWER = """\
**Potential EMI sources identified:**
- Fan motor switching noise
- Power supply noise coupling
- Close proximity between motor and antenna

**Design recommendations:**
- Increase distance between antenna and motor module
- Avoid placing antenna directly above motor
- Add shielding or ground reference near antenna
- Improve power filtering (LC filter)

*Next Step:* ***[Apply layout adjustment]*** *to execute all updates automatically.*\
"""

_STEP10_LAYOUT_PROGRESS = """\
Applying geometry update…
- Moving fan module (+5 mm)
- Updating airflow path
- Re-running thermal simulation…

✅ **Layout updated.**
*Type* ***3D*** *to switch to 3D preview mode automatically.*\
"""

_STEP11_3D_CONFIRM = "🖥️ **3D view ready.** Design Workspace switched to 3D preview."

_FALLBACK = """\
I'm searching the knowledge base for an answer…
Please provide more details or try rephrasing your question.\
"""


# ── Keyword patterns  ──────────────────────────────────────────────────────────

def _match(text: str, *patterns: str) -> bool:
    """Return True if any pattern matches text (case-insensitive)."""
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


# ── Main router ────────────────────────────────────────────────────────────────

def route_message(
    user_input: str,
    mode: str = "Modeling",
    qa_data: list | None = None,
) -> tuple[str, Optional[str]]:
    """
    Keyword-route user input to a scripted response + optional workspace action.

    Parameters
    ----------
    user_input : str
        Raw user message.
    mode : str
        Current workspace mode ("Modeling" | "Thermal Simulation" | "3D").
    qa_data : list | None
        Pre-loaded QA rows (loaded automatically when None).

    Returns
    -------
    (response_text, workspace_action)
    """
    txt = user_input.strip()

    # ── Exact / near-exact script triggers ────────────────────────────────────
    if _match(
        txt,
        r"apply passive thermal approach",
        r"passive thermal approach",
        r"apply.*thermal approach",
        r"apply optimized thermal approach",
    ):
        return _STEP4_PLACEMENT_PROGRESS, "apply_optimized"

    if _match(txt, r"apply layout adjustment", r"apply.*layout"):
        return _STEP10_LAYOUT_PROGRESS, _PRESET_LAYOUT

    if _match(txt, r"\b3[Dd]\b", r"switch.*3[Dd]", r"3[Dd].*preview",
              r"switch to 3d", r"切換.*3D", r"3D.*確認"):
        return _STEP11_3D_CONFIRM, "switch_3d"

    # ── Thermal simulation mode switch ─────────────────────────────────────────
    if _match(txt, r"thermal sim", r"live sim", r"切換.*熱", r"啟動.*simulation"):
        return _STEP6_THERMAL_MODE, "switch_thermal"

    # ── RF / antenna / EMI ─────────────────────────────────────────────────────
    if _match(txt, r"天線", r"antenna", r"\bEMI\b", r"風扇.*干擾", r"\bRF\b",
              r"motor.*noise", r"interference"):
        # Try CSV lookup first
        row = _csv_lookup(txt, qa_data)
        if row and row.get("expert_answer"):
            csv_text = _format_csv_row(row)
            return f"{_STEP8_RF_ROUTING}\n\n{csv_text}\n\n{_STEP9_RF_ANSWER}", None
        return f"{_STEP8_RF_ROUTING}\n\n{_STEP9_RF_ANSWER}", None

    # ── Thermal FAQ: surface temperature / IEC limit ───────────────────────────
    if _match(txt, r"表面溫度", r"surface.*temp", r"機殼.*溫度",
              r"iec", r"62368", r"溫度.*限制", r"temperature.*limit"):
        row = _csv_lookup(txt, qa_data)
        if row and row.get("expert_answer"):
            return f"{_STEP7_SURFACE_TEMP}\n\n---\n*Source: {row.get('reference','')}*", None
        return _STEP7_SURFACE_TEMP, None

    # ── Placement / spacing analysis ───────────────────────────────────────────
    if _match(txt, r"間距", r"spacing", r"arrangement", r"距離.*縮小",
              r"散熱模組", r"heatsink", r"cpu.*shift", r"排列"):
        return _STEP3_PLACEMENT, None

    # ── Generic thermal question → CSV lookup ──────────────────────────────────
    row = _csv_lookup(txt, qa_data)
    if row and row.get("expert_answer"):
        return _format_csv_row(row), None

    return _FALLBACK, None


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _csv_lookup(query: str, qa_data: list | None) -> Optional[dict]:
    try:
        data = qa_data if qa_data is not None else load_qa()
        return find_answer(query, data)
    except Exception:
        return None


def _format_csv_row(row: dict) -> str:
    """Format a FAQ row into a readable markdown block."""
    lines = []
    if row.get("question"):
        lines.append(f"**Q: {row['question']}**")
    if row.get("expert_answer"):
        lines.append(f"\n{row['expert_answer']}")
    if row.get("guideline"):
        lines.append(f"\n**Guideline:** {row['guideline']}")
    if row.get("reference"):
        lines.append(f"*Reference: {row['reference']}*")
    return "\n".join(lines)
