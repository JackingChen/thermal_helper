"""
data_loader.py
Loads the FAQ Excel file into a clean DataFrame.
"""
import pandas as pd
from pathlib import Path

XLSX_PATH = Path(__file__).parent / "data" / "(AIC 0408) Ai for design_Thermal Expert System_FAQ v.1.xlsx"

COL_ID = "編號"
COL_CATEGORY = "分類模組"
COL_QUESTION = "核心設計問題 (Question)"
COL_REFERENCE = "技術參考背景 (Problem Reference)"
COL_ANSWER = "專家建議答案 (Expert Answer)"
COL_GUIDELINE = "答案技術指標/設計準則 (Technical Guideline)"
COL_SOURCE = "參考資料"

ALL_COLS = [COL_ID, COL_CATEGORY, COL_QUESTION,
            COL_REFERENCE, COL_ANSWER, COL_GUIDELINE, COL_SOURCE]


def load_faq(path: Path = XLSX_PATH) -> pd.DataFrame:
    """Load FAQ from xlsx, skipping the first 2 header rows."""
    df = pd.read_excel(path, header=2, usecols=range(7))
    df.columns = ALL_COLS
    # Drop rows with no question
    df = df[df[COL_QUESTION].notna()].reset_index(drop=True)
    # Fill NaN with empty string for display
    df[COL_REFERENCE] = df[COL_REFERENCE].fillna("")
    df[COL_ANSWER] = df[COL_ANSWER].fillna("")
    df[COL_GUIDELINE] = df[COL_GUIDELINE].fillna("")
    df[COL_SOURCE] = df[COL_SOURCE].fillna("")
    return df
