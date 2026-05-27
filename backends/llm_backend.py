"""
backends/llm_backend.py
────────────────────────
Ollama Chat API backend for the Design Assistant.

Credentials are read from environment variables — never hardcode keys:
    AZURE_OPENAI_KEY           API key
    AZURE_OPENAI_ENDPOINT      Base URL  (default: https://gpt4fordg.openai.azure.com)
    AZURE_OPENAI_DEPLOYMENT    Model deployment name (default: gpt-4o)

Public API
──────────
    call_azure_llm(user_message, chat_history, positions, workspace_mode)
        → (response_text: str, action: dict | str | None)

The returned action has the same shape as chat_responses.route_message():
    None               — no workspace change
    dict               — placement instruction for execute_instruction()
    "switch_3d"        — switch workspace to 3D mode
    "switch_thermal"   — switch workspace to Thermal Simulation mode

GPT response format
───────────────────
GPT is instructed to respond with a single JSON object:
{
  "response": "<Markdown text to display in chat>",
  "mode_switch": null | "3D" | "Thermal Simulation",
  "placement": null | {
    "action": "move_sequence",
    "steps": [
      {"component": "<name>", "direction": "<right|left|up|down>", "delta": <mm>},
      {"component": "<name>", "rotate": true}
    ]
  }
}
"""
from __future__ import annotations

import html
import json
import logging
import os
import time
from typing import Optional

from pathlib import Path

import requests

_log = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────────
_ENDPOINT   = os.environ.get(
    "OLLAMA_ENDPOINT", "http://127.0.0.1:11435"
).rstrip("/")
_DEPLOYMENT = os.environ.get("OLLAMA_MODEL", "gemma:2b")
_API_URL    = f"{_ENDPOINT}/api/chat"
_TIMEOUT    = 60  # seconds


# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are CAD T, an AI-assisted CAE/EE thermal design assistant embedded in a PCB layout tool.
The user sees three views: 2D Modeling (component rectangles), Thermal Simulation (heatmap 20–120°C), and 3D Preview.

COORDINATE SYSTEM:
- Board origin (0, 0) is top-left
- X increases rightward, Y increases downward
- All distances are in millimetres

YOUR RESPONSE FORMAT:
Always reply with a single JSON object — no prose, no code fences, no extra keys:
{
  "response": "<Markdown text to display in the chat bubble — may use bold, bullets, tables>",
  "mode_switch": null | "3D" | "Thermal Simulation",
  "placement": null
    | {"action": "move_sequence", "steps": [{"component": "<name>", "direction": "<right|left|up|down>", "delta": <mm>}, ...]}
    | {"action": "apply_optimized"}
    | {"action": "apply_optimized_thermal"}
}

COMPONENT ADJUSTMENT RULES:
- Only include "placement" when you are actually moving or rotating components
- "direction" must be exactly one of: "right", "left", "up", "down"
- "delta" is a positive number in millimetres
- Use {"component": "<name>", "rotate": true} to toggle the rotated flag
- Component names must match the names provided in the geometry context (case-insensitive)
- Set "placement" to null if no geometry change is requested
- Use {"action": "apply_optimized"} ONLY when the context says "PASSIVE THERMAL PRESET AVAILABLE: yes"
  AND the user is confirming a material change or thermal strategy recommendation (e.g. "Apply", "Commit", "Yes", "Do it" after a material swap or passive thermal approach has been recommended in the conversation)
  You MAY also set "mode_switch": "Thermal Simulation" in the same response — both actions will be applied together
- Use {"action": "apply_optimized_thermal"} ONLY when the context says "THERMAL PRESET AVAILABLE: yes"
  AND the user is confirming a thermal enhancement recommendation (e.g. "Apply", "Commit", "Yes" after heat pipe or thermal upgrade has been recommended in the conversation)
  You MAY also set "mode_switch": "Thermal Simulation" in the same response — both actions will be applied together

MODE SWITCH RULES:
- Set "mode_switch": "3D" when the user wants a 3D preview
- Set "mode_switch": "Thermal Simulation" when the user wants to see the heatmap
- Set "mode_switch": null otherwise

THERMAL KNOWLEDGE:
- IEC 62368-1: touch-accessible surfaces must be < 48°C; operator areas < 70°C
- CPU Tcase typically < 95–100°C; heatsink base should stay < 70°C under load
- Fan clearance ≥ 10 mm recommended to avoid wake turbulence on adjacent components
- Heatsink fins should be oriented parallel to airflow direction
- Reducing CPU-to-heatsink gap below 8 mm restricts airflow by ~15–20%, raising hotspot ~3–5°C

When project-specific layout constraints are provided below, treat them as authoritative
constraints for the current project. Fixed constraints (marked with = or _align) must
never be violated. Optimizable constraints list lower bounds that can be increased.\
"""


# ── Project rules loader ──────────────────────────────────────────────────────

_RULES_DIR = Path(__file__).parent.parent / "assets" / "project_rule"


def _load_project_rules(project: str) -> str:
    """Return the contents of assets/project_rule/<project>/<project>.md, or '' if not found."""
    if not project:
        return ""
    path = _RULES_DIR / project / f"{project}.md"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        _log.warning("[LLM] could not read project rules %s: %s", path, exc)
        return ""


def _has_optimized_preset(project: str) -> bool:
    """Return True when assets/project_rule/<project>/<project>_optimized.json exists."""
    if not project:
        return False
    return (_RULES_DIR / project / f"{project}_optimized.json").exists()


def _has_thermal_optimized_preset(project: str) -> bool:
    """Return True when assets/project_rule/<project>/<project>_optimized_4_thermal.json exists."""
    if not project:
        return False
    return (_RULES_DIR / project / f"{project}_optimized_4_thermal.json").exists()


def _load_optimized_hint(project: str) -> str:
    """
    Load per-project optimized preset hint from <project>/<project>_optimized_hint.md.
    Returns the file contents when it exists, otherwise the generic Hitatori-style fallback.
    """
    if project:
        path = _RULES_DIR / project / f"{project}_optimized_hint.md"
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            pass
        except OSError as exc:
            _log.warning("[LLM] could not read optimized hint %s: %s", path, exc)
    # Generic fallback (original Hitatori wording)
    return (
        "PASSIVE THERMAL PRESET AVAILABLE: yes\n"
        "Pre-computed preset = Step A (see project rules): heatsink material swap — Heatsink-1 and Heatsink-2 change to Aluminum 6061.\n"
        "TRIGGER: respond with {'action': 'apply_optimized'} AND 'mode_switch': 'Thermal Simulation' when the user "
        "confirms a material swap recommendation (e.g. says 'Apply', 'Commit', 'Yes', 'Do it' after a heatsink "
        "material change has been suggested in the conversation). "
        "Do NOT apply on a general question — only on an explicit confirmation.\n"
        "Do NOT generate move_sequence steps.\n"
        "Explain in your response with this recommendation style:\n"
        "**Recommendation:**\n"
        "- Swap Heatsink-1 and Heatsink-2 from Copper C1100 to Aluminum 6061.\n"
        "- Aluminum's lower density reduces thermal mass, improving the heatsink's ability to respond to rapid temperature changes from the CPU.\n"
        "- This change typically yields a ~6°C reduction in CPU hotspot temperature during peak 30-second loads, with no loss in overall cooling under steady airflow."
    )


def _format_memory_context(memory_list: list[dict]) -> str:
    """Format session memory facts into a compact context block for the prompt."""
    if not memory_list:
        return ""
    
    lines = []
    for mem in memory_list:
        ts = mem.get("timestamp", "")[:16].replace("T", " ") if mem.get("timestamp") else ""
        fact = mem.get("fact", "")
        source = mem.get("source", "")
        lines.append(f"  • [{ts}] {fact} (source: {source})")
    
    return "\n".join(lines) if lines else ""


# ── Geometry context builder ───────────────────────────────────────────────────

def _geometry_context(positions: dict, workspace_mode: str) -> str:
    """Format current component positions into a compact text block for the prompt."""
    board = positions.get("_board", {})
    board_w = board.get("w", "?")
    board_h = board.get("h", "?")

    lines: list[str] = []
    for name, comp in positions.items():
        if name == "_board":
            continue
        x = round(float(comp.get("x", 0)), 1)
        y = round(float(comp.get("y", 0)), 1)
        w = round(float(comp.get("w", 0)), 1)
        h = round(float(comp.get("h", 0)), 1)
        label = comp.get("label", name)
        rotated = comp.get("rotated", False)
        material = comp.get("material", "metal")
        lines.append(
            f"  {label}: centre=({x}, {y}) mm, size={w}×{h} mm, material={material}"
            + (" [rotated]" if rotated else "")
        )

    comps = "\n".join(lines) if lines else "  (no components loaded)"
    return (
        f"CURRENT WORKSPACE MODE: {workspace_mode}\n"
        f"BOARD SIZE: {board_w} × {board_h} mm\n"
        f"COMPONENT POSITIONS:\n{comps}"
    )


# ── API call ───────────────────────────────────────────────────────────────────

def call_llm(
    user_message: str,
    chat_history: list[dict],
    positions: dict,
    workspace_mode: str,
    project: str = "",
    session_cfg: dict | None = None,
    preset_stage: str = "initial",
) -> tuple[str, Optional[dict | str]]:
    """
    Call the Ollama Chat API and return (response_text, action).

    Parameters
    ----------
    user_message   : Current user input.
    chat_history   : Full history list including the current user turn
                     (each item: {"role": "user"|"assistant", "text": str}).
    positions      : Current component_positions dict from session state.
    workspace_mode : Current workspace mode string.
    project        : Active project name; used to load project_rule/<project>.md.
    session_cfg    : Optional OpenClaw session config dict (contains memory facts).
    preset_stage   : Current applied preset stage — "initial" | "optimized" | "optimized_thermal".
    """

    # ── Build message list ─────────────────────────────────────────────────────
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Geometry snapshot + project rules as a primed exchange
    geo_ctx = _geometry_context(positions, workspace_mode)
    rules = _load_project_rules(project)
    ctx_parts = [f"[GEOMETRY CONTEXT — do not respond to this]\n{geo_ctx}"]
    
    # Inject memory facts if available
    if session_cfg and session_cfg.get("memory"):
        memory_ctx = _format_memory_context(session_cfg["memory"])
        if memory_ctx:
            ctx_parts.append(f"[SESSION MEMORY — user advice & facts]\n{memory_ctx}")
    
    if rules:
        ctx_parts.append(f"PROJECT RULES ({project}):\n{rules}")
    if preset_stage == "initial" and _has_optimized_preset(project):
        ctx_parts.append(_load_optimized_hint(project))
    if preset_stage == "optimized" and _has_thermal_optimized_preset(project):
        ctx_parts.append(
            "THERMAL PRESET AVAILABLE: yes\n"
            "Pre-computed preset = Step B: heat pipe enhancement — adds 3× 6 mm sintered copper heat pipes "
            "(Heatsink-HP-1/2/3) bridging CPU cold plate to rear fin stack.\n"
            "CURRENT STATE: Step A (material swap) is already applied. Step B is the only available next action.\n"
            "TRIGGER: when the user says 'Apply', 'Yes', 'Commit', or any explicit confirmation AFTER heat pipe "
            "enhancement has been discussed, output EXACTLY:\n"
            "  'placement': {'action': 'apply_optimized_thermal'}  ← MUST NOT be null\n"
            "  'mode_switch': 'Thermal Simulation'\n"
            "Do NOT set placement to null on confirmation — the system will not apply the preset without it.\n"
            "Do NOT generate move_sequence steps.\n"
            "Explain in your response:\n"
            "**Further Recommendation:**\n"
            "- Integrate 3× 6 mm sintered copper heat pipes (Heatsink-HP-1, 2, 3) bridging the CPU cold-plate directly to the rear fin stack.\n"
            "- These heat pipes will rapidly transfer heat away from the CPU, greatly improving peak and sustained cooling performance.\n"
            "- This enhancement typically brings an additional **~11°C reduction in CPU hotspot**, for a total improvement of ~17°C versus the original baseline."
        )
    messages.append({
        "role": "user",
        "content": "\n\n".join(ctx_parts),
    })
    messages.append({
        "role": "assistant",
        "content": '{"response": "Context received.", "mode_switch": null, "placement": null}',
    })

    # Previous turns (exclude the last entry which is the current user message)
    for msg in chat_history[:-1]:
        messages.append({"role": msg["role"], "content": msg["text"]})

    # Current user turn
    messages.append({"role": "user", "content": user_message})

    payload = {"model": _DEPLOYMENT, "messages": messages, "stream": False}
    _log.info("[LLM] payload: %d messages, ~%d chars",
              len(messages), sum(len(m["content"]) for m in messages))

    # ── HTTP request ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            _API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        _log.warning("[LLM] request timed out after %.1f s", time.perf_counter() - t0)
        return "⚠️ **Request timed out.** The AI service did not respond in time.", None
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        _log.warning("[LLM] HTTP error %s after %.1f s", status, time.perf_counter() - t0)
        return f"⚠️ **API HTTP error {status}.** Check your endpoint.", None
    except requests.exceptions.RequestException as exc:
        _log.warning("[LLM] network error after %.1f s: %s", time.perf_counter() - t0, exc)
        return f"⚠️ **Network error:** {html.escape(str(exc))}", None

    t_http = time.perf_counter() - t0
    _log.info("[LLM] HTTP round-trip: %.2f s  |  response size: %d bytes",
              t_http, len(resp.content))

    # ── Parse ─────────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        data = resp.json()
        raw_text = data.get("message", {}).get("content", "")
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        return (
            f"⚠️ **Unexpected API response format:** {html.escape(str(exc))}",
            None,
        )

    result = _parse_llm_output(raw_text)
    _log.info("[LLM] parse: %.3f s  |  total: %.2f s",
              time.perf_counter() - t1, time.perf_counter() - t0)
    return result


# ── Output parser ──────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> tuple[str, Optional[dict | str]]:
    """
    Parse GPT's JSON reply into (response_text, action).
    Falls back gracefully if the output is not valid JSON.
    """
    stripped = raw.strip()

    # Strip markdown code fences if GPT wraps in ```json ... ```
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        stripped = "\n".join(lines[1:end])

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        # LLM sometimes outputs prose text followed by a JSON blob.
        # Scan forward from each '{' until we find a valid JSON object.
        idx = stripped.find("{")
        obj = None
        while idx != -1:
            try:
                obj = json.loads(stripped[idx:])
                break
            except json.JSONDecodeError:
                idx = stripped.find("{", idx + 1)
        if obj is None:
            # No valid JSON found anywhere — render raw text safely
            return html.escape(raw), None

    response_text: str = obj.get("response", "")
    mode_switch         = obj.get("mode_switch")
    placement           = obj.get("placement")

    # Determine action (same contract as chat_responses.route_message).
    # Placement presets take priority over mode_switch; when both are present
    # a 2-tuple (preset_action, mode_action) is returned so _handle_chat can
    # apply both in sequence.
    action: Optional[dict | str | tuple] = None
    if isinstance(placement, dict) and placement.get("action") == "apply_optimized":
        if mode_switch == "Thermal Simulation":
            action = ("apply_optimized", "switch_thermal")
        elif mode_switch == "3D":
            action = ("apply_optimized", "switch_3d")
        else:
            action = "apply_optimized"
    elif isinstance(placement, dict) and placement.get("action") == "apply_optimized_thermal":
        if mode_switch == "Thermal Simulation":
            action = ("apply_optimized_thermal", "switch_thermal")
        elif mode_switch == "3D":
            action = ("apply_optimized_thermal", "switch_3d")
        else:
            action = "apply_optimized_thermal"
    elif isinstance(placement, dict) and placement.get("action") == "move_sequence":
        action = placement
    elif mode_switch == "3D":
        action = "switch_3d"
    elif mode_switch == "Thermal Simulation":
        action = "switch_thermal"

    return response_text, action
