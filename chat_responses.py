"""
chat_responses.py
──────────────────
All scripted chat text and routing logic.

Public API
──────────
    route_message(user_input: str, mode: str, qa_data: list | None = None, preset_stage: str = "initial") -> tuple[str, str | tuple | dict | None]

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

_HEATSINK_ALUMINUM_PROGRESS = """\
Switching heatsink material to **aluminum alloy**…
- Replacing copper base and heat pipes with aluminum
- Updating material properties (conductivity, density)
- Re-running PINN thermal prediction (aluminum model)

✅ **Aluminum heatsink applied.**
Thermal simulation now reflects the aluminum model prediction.
Switch to **Thermal Simulation** to view the updated temperature field.\
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


_SCENARIO_1_TEXT = """\
**Recommendation:**
- Swap Heatsink-1 and Heatsink-2 from Copper C1100 to Aluminum 6061.
- Aluminum's lower density reduces thermal mass, improving the heatsink's ability to respond to rapid temperature changes from the CPU.
- This change typically yields a ~6°C reduction in CPU hotspot temperature during peak 30-second loads, with no loss in overall cooling under steady airflow.\
"""

_SCENARIO_2_TEXT = """\
Applying the recommended heatsink material swap…
- Replacing Fin-1 and Fin-2 from Copper C1100 to Aluminum 6061
- Updating material properties (conductivity, density, thermal mass)
- Re-running PINN thermal prediction with aluminum model

✅ **Aluminum heatsink applied.** Switching to Thermal Simulation view.\
"""

_SCENARIO_3_TEXT = """\
Based on the current simulation result, I found that this case is outside the coverage of my training dataset.
The model confidence is relatively low, so the prediction may not be reliable enough for engineering decision-making.
I recommend collecting additional simulation data under similar thermal conditions and retraining the model to improve accuracy and coverage.

Still, i can do somthing to the best of my knowledge. In my experience, Integrate 3× 6 mm sintered copper heat pipes (Heatsink-HP-1, 2, 3) 
bridging the CPU cold-plate directly to the rear fin stack.\
"""

_SCENARIO_4_TEXT = """\
Applying the heat pipe enhancement…
- Integrating 3× 6 mm sintered copper heat pipes (Heatsink-HP-1, 2, 3)
- Bridging the CPU cold-plate directly to the rear fin stack
- Re-running thermal simulation with updated layout

✅ **Heat pipe layout applied.** Switching to Thermal Simulation view.\
"""

_SEMANTIC_ROUTES = [
    {
        "query": "The CPU temperature exceeds the limit, help me optimize the thermal design.",
        "response": _SCENARIO_1_TEXT,
        "action": None,
        "required_stage": "initial"
    },
    {
        "query": "Go ahead and apply it.",
        "response": _SCENARIO_2_TEXT,
        "action": ("apply_optimized", "switch_thermal"),
        "required_stage": "initial"
    },
    {
        "query": "Based on existing heat pipe / dual-zone heat spreading design cases, give me a better solution than the current one.",
        "response": _SCENARIO_3_TEXT,
        "action": None,
        "required_stage": "optimized"
    },
    {
        "query": "Apply.",
        "response": _SCENARIO_4_TEXT,
        "action": ("apply_optimized_thermal", "switch_thermal"),
        "required_stage": "optimized"
    },
    # ── Scenario 2 ────────────────────────────────────────────────────────────
    {
        "query": "Please give me thermal optimization suggestions for the CPU.",
        "response": _SCENARIO_1_TEXT,
        "action": None,
        "required_stage": "initial"
    },
    {
        "query": "Do it.",
        "response": _SCENARIO_2_TEXT,
        "action": ("apply_optimized", "switch_thermal"),
        "required_stage": "initial"
    },
    {
        "query": "Refer to past projects with high-performance, thin-and-light thermal designs and propose a new suggestion.",
        "response": _SCENARIO_3_TEXT,
        "action": None,
        "required_stage": "optimized"
    },
    {
        "query": "OK, let's do it this way.",
        "response": _SCENARIO_4_TEXT,
        "action": ("apply_optimized_thermal", "switch_thermal"),
        "required_stage": "optimized"
    },
]

_encoder = None
_SEMANTIC_EMBEDDINGS = None

def _get_semantic_match(user_input: str, preset_stage: str):
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
    except ImportError:
        return None, None
        
    global _encoder, _SEMANTIC_EMBEDDINGS
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
    if _SEMANTIC_EMBEDDINGS is None:
        queries = [r["query"] for r in _SEMANTIC_ROUTES]
        _SEMANTIC_EMBEDDINGS = _encoder.encode(queries, convert_to_tensor=True)
        
    user_emb = _encoder.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_emb, _SEMANTIC_EMBEDDINGS)[0]
    
    # Sort indices by score descending
    sorted_indices = torch.argsort(cos_scores, descending=True)
    
    for idx in sorted_indices:
        idx = idx.item()
        score = cos_scores[idx].item()
        if score < 0.75:
            break  # since it's sorted, remaining scores are lower
            
        route = _SEMANTIC_ROUTES[idx]
        if not route.get("required_stage") or route["required_stage"] == preset_stage:
            return route["response"], route["action"]
            
    return None, None



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
    preset_stage: str = "initial",
) -> tuple[str, Optional[str | tuple | dict]]:
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

    # ── Semantic text similarity triggers ──────────────────────────────────────
    semantic_response, semantic_action = _get_semantic_match(txt, preset_stage)
    if semantic_response is not None:
        return semantic_response, semantic_action

    # ── Exact / near-exact script triggers ────────────────────────────────────
    if _match(
        txt,
        r"apply passive thermal approach",
        r"passive thermal approach",
        r"apply.*thermal approach",
        r"apply optimized thermal approach",
    ):
        return _STEP4_PLACEMENT_PROGRESS, "apply_optimized"

    if _match(
        txt,
        r"switch.*aluminum",
        r"apply.*aluminum",
        r"copper.*aluminum",
        r"aluminum heatsink",
        r"change.*copper.*aluminum",
        r"optimize.*heatsink",
        r"apply.*optimized",
    ):
        return _HEATSINK_ALUMINUM_PROGRESS, "apply_optimized"

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
