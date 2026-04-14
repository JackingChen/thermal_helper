# Thermal Expert QA Agent

A Streamlit demo app that lets engineers query a thermal-design FAQ knowledge base using semantic search. Results are displayed in a chat interface with support for inline visualisation sub-windows.

---

## Layout

```
┌─────────────────┬──────────────────────────────┐
│  Project Files  │       Display window         │
│  (sidebar)      │  chat history + inline figs  │
│                 ├──────────────────────────────┤
│                 │       Prompt window          │
└─────────────────┴──────────────────────────────┘
```

- **Project Files (sidebar)** — browse and filter all 50 FAQ entries by category; click any to pre-fill the prompt.
- **Display window** — scrollable chat history showing user questions and AI answer cards. Graphics (plots, 3-D views) appear as bordered inline sub-windows inside the AI response.
- **Prompt window** — free-text input, Top-K selector, and similarity score toggle.

---

## Data Source

`(AIC 0408) Ai for design_Thermal Expert System_FAQ v.1.xlsx`

| Column | Role |
|---|---|
| 編號 | ID |
| 分類模組 | Category (架構 / 組件 / 風扇 / 模擬 / 流體 / 材料) |
| 核心設計問題 (Question) | **Input** – indexed for retrieval |
| 技術參考背景 (Problem Reference) | **Output** |
| 專家建議答案 (Expert Answer) | **Output** |
| 答案技術指標/設計準則 (Technical Guideline) | **Output** |
| 參考資料 | **Output** |

Row 1–2 are metadata/notes; data starts from row 3 (header) and row 4 onwards.

---

## Architecture

```
qa_agent/
├── app.py             # Streamlit entry point – layout, session state, submission logic
├── data_loader.py     # Reads xlsx, exposes column constants
├── retriever.py       # FAQRetriever – sentence-transformer semantic search
├── display.py         # render_chat_answer() – chat cards + inline sub-windows
├── backends/
│   ├── __init__.py    # BackendResult dataclass + KEYWORD_MAP + resolve_backend()
│   ├── thermal.py     # 🌡️  Thermal computation backend (stub)
│   └── placement.py   # 📦  Component placement backend (stub)
└── requirements.txt
```

### Retrieval

Uses **`paraphrase-multilingual-MiniLM-L12-v2`** (sentence-transformers). All FAQ questions are embedded at startup and stored as normalised vectors. At query time, the user question is embedded and cosine similarity is computed against all question vectors. Top-K results are returned ranked by score.

To swap for a larger model, change `DEFAULT_MODEL` in `retriever.py`.

### Backend system

When the top retrieval result matches a keyword (see `backends/KEYWORD_MAP`), the corresponding backend module is automatically imported and its `run(context)` function is called. The returned `BackendResult` (title + Plotly figure + summary) is rendered as an inline sub-window inside the AI chat bubble.

**Current backends (stub state):**

| Backend | Trigger keywords | Planned output |
|---|---|---|
| `thermal` | 熱, 散熱, heat, thermal, rth, junction | RC-network temperature simulation |
| `placement` | placement, layout, 擺放, 佈局, component, opening | PCB 3-D component layout colour view |

To implement a backend: edit `backends/thermal.py` or `backends/placement.py`, replace the placeholder figure in the `run()` function with real solver output, and return a populated `BackendResult`.

---

## Setup

### Using the native Linux filesystem (recommended – faster I/O)

```bash
cd /home/project

# Step 1 – install CPU-only torch first (avoids pulling CUDA packages)
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch

# Step 2 – install remaining dependencies
.venv/bin/pip install streamlit openpyxl pandas scikit-learn plotly sentence-transformers
```

### Run

```bash
cd /home/project
.venv/bin/streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Future Roadmap

- [ ] Replace `thermal.py` stub with real RC-network solver
- [ ] Replace `placement.py` stub with PCB layout rule-checker / optimiser
- [ ] Add `backends/cfd.py` for CFD airflow visualisation
- [ ] Upgrade retriever to `BAAI/bge-m3` for higher accuracy
- [ ] Add FAISS index for large corpus scaling
- [ ] Stream AI responses token-by-token via `st.write_stream`



# ISSUE
1. 按submit沒有反應