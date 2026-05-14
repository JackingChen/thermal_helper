"""
session_config.py
─────────────────
OpenClaw-style session config manager for the Design Assistant showcase.

The config lives entirely in st.session_state["session_cfg"] — never on disk.
It demonstrates the OpenClaw concepts of skills, memory, and subagents without
coupling them to the actual system workflow (thermal_sim, llm_backend, etc.).

Public API
──────────
    default_config()                              → dict
    log_subagent(cfg, name, trigger, result)      → mutates cfg in-place
    add_memory(cfg, fact, source="user_advice")   → mutates cfg in-place
    as_json(cfg)                                  → str  (indented JSON)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone


# ── Default skills shipped with every new session ─────────────────────────────
_DEFAULT_SKILLS: list[dict] = [
    {
        "name": "thermal_analysis",
        "enabled": True,
        "description": "Gaussian-field PCB thermal simulation (100 × 150 grid)",
    },
    {
        "name": "placement_optimization",
        "enabled": True,
        "description": "Component placement constraint solver and preset loader",
    },
    {
        "name": "emi_faq_lookup",
        "enabled": True,
        "description": "RF/EMI expert Q&A retrieval from structured FAQ CSV",
    },
    {
        "name": "geometry_context",
        "enabled": True,
        "description": "PCB coordinate system awareness and component geometry injection",
    },
    {
        "name": "project_rules",
        "enabled": True,
        "description": "Per-project constraint file loader (assets/project_rule/<project>/)",
    },
]

# ── Default subagents registered in every session ─────────────────────────────
_DEFAULT_SUBAGENTS: list[dict] = [
    {
        "name": "thermal_sim_agent",
        "description": "Runs NumPy thermal field simulation for the active project",
        "invocations": [],
    },
    {
        "name": "placement_agent",
        "description": "Applies preset or delta placement moves to component positions",
        "invocations": [],
    },
    {
        "name": "rag_faq_agent",
        "description": "Retrieves best-match FAQ answer using token-overlap scoring",
        "invocations": [],
    },
    {
        "name": "llm_reasoning_agent",
        "description": "Azure OpenAI reasoning agent (GPT-4o) for open-ended queries",
        "invocations": [],
    },
]


# ── Public helpers ─────────────────────────────────────────────────────────────

def default_config() -> dict:
    """Return a freshly initialised OpenClaw session config."""
    return {
        "session_id": uuid.uuid4().hex[:12],
        "started_at": _now(),
        "agent": {
            "model": "inference/nvidia/nemotron-3-super-120b-a12b",
            "mode": "local",
        },
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


# ── Internal ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
