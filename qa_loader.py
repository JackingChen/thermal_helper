"""
qa_loader.py
─────────────
Loads the thermal/RF FAQ CSV and provides a keyword-match retriever.

Usage
─────
    from qa_loader import load_qa, find_answer

    qa_data = load_qa()                          # cached on first call
    row     = find_answer("surface temperature") # best-matching dict or None
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_CSV_PATH = (
    _HERE
    / "data"
    / "(AIC 0408) Ai for design_Thermal Expert System_FAQ v.1.csv"
)

# ── Column aliases (normalised) ────────────────────────────────────────────────
# The CSV uses Traditional Chinese headers; we map them to short English keys
# so downstream code stays readable.
_COL_MAP = {
    "分類模組": "category",
    "核心設計問題 (Question)": "question",
    "技術參考背景 (Problem Reference)": "problem_reference",
    "專家建議答案 (Expert Answer)": "expert_answer",
    "答案技術指標/設計準則 (Technical Guideline)": "guideline",
    "參考資料": "reference",
}


@lru_cache(maxsize=1)
def load_qa(path: str | None = None) -> list[dict]:
    """
    Load the FAQ CSV and return a list of normalised row dicts.

    Parameters
    ----------
    path : str | None
        Override the default CSV path (useful for testing).

    Returns
    -------
    list[dict]  — each dict has keys from _COL_MAP plus 'id'.
    """
    csv_path = Path(path) if path else _CSV_PATH

    # Try common CJK encodings in order
    for enc in ("big5", "gbk", "utf-8-sig", "utf-8", "cp950"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=str)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise RuntimeError(f"Cannot decode {csv_path} — tried big5/gbk/utf-8")

    # Rename columns where possible
    rename = {c: _COL_MAP[c] for c in df.columns if c in _COL_MAP}
    df = df.rename(columns=rename)

    # Ensure all expected keys exist (fill missing with empty string)
    for key in _COL_MAP.values():
        if key not in df.columns:
            df[key] = ""

    df = df.fillna("")
    records = df.to_dict(orient="records")

    # Add a stable integer id if the CSV doesn't provide one
    for i, row in enumerate(records, start=1):
        row.setdefault("id", i)

    return records


# ── Retrieval helpers ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenise(text: str) -> set[str]:
    """Split on non-word boundaries; also keep CJK chars as individual tokens."""
    tokens: set[str] = set()
    # ASCII / Latin words
    tokens.update(re.findall(r"[a-z0-9]+", text))
    # CJK characters (each char is a meaningful unit)
    tokens.update(re.findall(r"[\u4e00-\u9fff]", text))
    return tokens


def find_answer(
    query: str,
    qa_data: list[dict] | None = None,
    top_n: int = 1,
) -> Optional[dict]:
    """
    Return the best-matching FAQ row for *query*, or None if no rows exist.

    Scoring: token-overlap count across (question + subcategory + category)
    fields.  Ties broken by question-field weight (×2).

    Parameters
    ----------
    query   : user input string (mixed Chinese / English OK)
    qa_data : pre-loaded list from load_qa(); loaded automatically if None
    top_n   : return only the single best match (future: support a list)
    """
    if qa_data is None:
        qa_data = load_qa()

    if not qa_data:
        return None

    q_tokens = _tokenise(_normalise(query))
    if not q_tokens:
        return None

    best_row: Optional[dict] = None
    best_score = -1

    for row in qa_data:
        q_text = _normalise(row.get("question", ""))
        sub    = _normalise(row.get("subcategory", ""))
        cat    = _normalise(row.get("category", ""))

        q_tok = _tokenise(q_text)
        s_tok = _tokenise(sub)
        c_tok = _tokenise(cat)

        # question matches count double
        score = (
            2 * len(q_tokens & q_tok)
            + len(q_tokens & s_tok)
            + len(q_tokens & c_tok)
        )

        if score > best_score:
            best_score = score
            best_row = row

    return best_row if best_score > 0 else None


def find_answers_by_category(
    category_keyword: str,
    qa_data: list[dict] | None = None,
) -> list[dict]:
    """Return all rows whose 'category' contains *category_keyword*."""
    if qa_data is None:
        qa_data = load_qa()
    kw = _normalise(category_keyword)
    return [
        r for r in qa_data
        if kw in _normalise(r.get("category", ""))
    ]
