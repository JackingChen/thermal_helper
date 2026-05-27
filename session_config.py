"""
session_config.py
─────────────────
OpenClaw-style session config manager for the Design Assistant showcase.

Default skills and subagents are loaded from  data/agent_config.yaml.
The live config lives in st.session_state["session_cfg"] — never on disk.

Public API
──────────
    default_config()                              → dict
    log_subagent(cfg, name, trigger, result)      → mutates cfg in-place
    add_memory(cfg, fact, source="user_advice")   → mutates cfg in-place
    as_json(cfg)                                  → str  (indented JSON)
    as_markdown(cfg)                              → str  (skill-style Markdown)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ── Load defaults from data/agent_config.yaml ─────────────────────────────────
_CONFIG_FILE = Path(__file__).parent / "data" / "agent_config.yaml"

def _load_yaml_defaults() -> dict:
    """Load skills, subagents and agent config from YAML. Falls back to empty dicts."""
    if _YAML_AVAILABLE and _CONFIG_FILE.exists():
        with _CONFIG_FILE.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

_YAML_CFG = _load_yaml_defaults()

_DEFAULT_SKILLS: list[dict] = [
    {"name": s["name"], "enabled": s.get("enabled", True), "description": s.get("description", "")}
    for s in _YAML_CFG.get("skills", [])
] or [
    {"name": "thermal_analysis", "enabled": True, "description": "Gaussian-field PCB thermal simulation (100 × 150 grid)"},
    {"name": "placement_optimization", "enabled": True, "description": "Component placement constraint solver and preset loader"},
    {"name": "emi_faq_lookup", "enabled": True, "description": "RF/EMI expert Q&A retrieval from structured FAQ CSV"},
    {"name": "geometry_context", "enabled": True, "description": "PCB coordinate system awareness and component geometry injection"},
    {"name": "project_rules", "enabled": True, "description": "Per-project constraint file loader (assets/project_rule/<project>/)"},
]

_DEFAULT_SUBAGENTS: list[dict] = [
    {"name": s["name"], "description": s.get("description", ""), "invocations": []}
    for s in _YAML_CFG.get("subagents", [])
] or [
    {"name": "thermal_sim_agent", "description": "Runs NumPy thermal field simulation for the active project", "invocations": []},
    {"name": "placement_agent", "description": "Applies preset or delta placement moves to component positions", "invocations": []},
    {"name": "rag_faq_agent", "description": "Retrieves best-match FAQ answer using token-overlap scoring", "invocations": []},
    {"name": "llm_reasoning_agent", "description": "Ollama reasoning agent (gemma:2b) for open-ended queries", "invocations": []},
]

_DEFAULT_AGENT = _YAML_CFG.get("agent", {
    "model": "inference/nvidia/nemotron-3-super-120b-a12b",
    "mode": "local",
})

_AGENT_NAME = _YAML_CFG.get("name", "thermal-design-assistant")
_AGENT_DESC = _YAML_CFG.get("description", "AI copilot for thermal management and PCB layout design.")


# ── Public helpers ─────────────────────────────────────────────────────────────

def default_config() -> dict:
    """Return a freshly initialised OpenClaw session config."""
    return {
        "session_id": uuid.uuid4().hex[:12],
        "started_at": _now(),
        "agent": dict(_DEFAULT_AGENT),
        "skills": [dict(s) for s in _DEFAULT_SKILLS],
        "memory": [],
        "subagents": [{**s, "invocations": []} for s in _DEFAULT_SUBAGENTS],
    }


def log_subagent(cfg: dict, name: str, trigger: str, result: str) -> None:
    """Append an invocation record to the named subagent entry."""
    record = {
        "invoked_at": _now(),
        "trigger": trigger,
        "result": result,
    }
    for sa in cfg.get("subagents", []):
        if sa["name"] == name:
            sa.setdefault("invocations", []).append(record)
            return
    # Ad-hoc subagent not in the default list
    cfg.setdefault("subagents", []).append({
        "name": name,
        "description": "(ad-hoc)",
        "invocations": [record],
    })


def add_memory(cfg: dict, fact: str, source: str = "user_advice") -> None:
    """Append a memory fact to the session config."""
    cfg.setdefault("memory", []).append({
        "timestamp": _now(),
        "fact": fact,
        "source": source,
    })


def as_json(cfg: dict) -> str:
    """Return the config as an indented JSON string (UTF-8 safe)."""
    return json.dumps(cfg, indent=2, ensure_ascii=False)


def as_markdown(cfg: dict) -> str:
    """Render the session config as a skill-style Markdown document."""
    agent = cfg.get("agent", {})
    skills = cfg.get("skills", [])
    memory = cfg.get("memory", [])
    subagents = cfg.get("subagents", [])

    lines: list[str] = [
        f"---",
        f"name: {_AGENT_NAME}",
        f"session_id: {cfg.get('session_id', 'n/a')}",
        f"started_at: {cfg.get('started_at', 'n/a')}",
        f"model: {agent.get('model', 'n/a')}",
        f"mode: {agent.get('mode', 'n/a')}",
        f"---",
        f"",
        f"# {_AGENT_NAME.replace('-', ' ').title()}",
        f"",
        f"## Overview",
        f"{_AGENT_DESC.strip()}",
        f"",
        f"## Skills",
        f"",
        f"| Skill | Status | Description |",
        f"|-------|--------|-------------|" ,
    ]
    for s in skills:
        status = "✅ enabled" if s.get("enabled") else "⬜ disabled"
        lines.append(f"| `{s['name']}` | {status} | {s.get('description', '')} |")

    lines += [
        "",
        "## Memory",
        "",
    ]
    if memory:
        for m in memory:
            ts = m.get("timestamp", "")[:16].replace("T", " ")
            lines.append(f"- **[{ts}]** {m.get('fact', '')}  *(source: {m.get('source', '')})*")
    else:
        lines.append("*No memory facts recorded yet.*")

    lines += [
        "",
        "## Subagents",
        "",
        "| Agent | Invocations |",
        "|-------|-------------|" ,
    ]
    for sa in subagents:
        n_inv = len(sa.get("invocations", []))
        lines.append(f"| `{sa['name']}` | {n_inv} |")

    return "\n".join(lines)


# ── Internal ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
