"""
app.py  –  Thermal Expert QA Agent (Streamlit demo)

Layout:
  ┌─────────────────┬──────────────────────────────┐
  │  Project Files  │       Display window         │
  │  (sidebar)      │  chat history + inline figs  │
  │                 ├──────────────────────────────┤
  │                 │       Prompt window          │
  └─────────────────┴──────────────────────────────┘

A single Display window holds the full conversation (user + AI bubbles).
When an answer carries a figure it is rendered inline as a bordered
sub-window directly inside the AI chat bubble.
"""
import streamlit as st

from data_loader import load_faq, COL_ID, COL_CATEGORY, COL_QUESTION
from retriever import FAQRetriever
from display import render_chat_answer
from backends import resolve_backend
import importlib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermal Expert QA",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str, "results": list|None}
    st.session_state.history = []

# ── Load data & build retriever (cached) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading FAQ & building index…")
def get_retriever() -> FAQRetriever:
    df = load_faq()
    return FAQRetriever(df)


retriever = get_retriever()
df = retriever.df

# ── Sidebar – project file browser ───────────────────────────────────────────
with st.sidebar:
    st.title("📁 Project Files")
    st.caption("Thermal Expert System FAQ")

    categories = ["All"] + sorted(df[COL_CATEGORY].unique().tolist())
    selected_cat = st.selectbox("Filter by category", categories)
    filtered = df if selected_cat == "All" else df[df[COL_CATEGORY] == selected_cat]

    st.markdown(f"**{len(filtered)} questions** in `{selected_cat}`")
    st.divider()

    for _, row in filtered.iterrows():
        label = f"[{row[COL_ID]}] {row[COL_QUESTION]}"
        if st.button(label, key=f"q_{row[COL_ID]}", use_container_width=True):
            st.session_state["prefill"] = row[COL_QUESTION]

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.caption("Tip: click any question above to pre-fill the prompt.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🌡️ Thermal Expert QA Agent")

# ── Display window ─────────────────────────────────────────────────────────────
st.subheader("📺 Display")
display_area = st.container(height=520)
with display_area:
    if not st.session_state.history:
        st.info("Ask a question below to get started.")
    for msg in st.session_state.history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                results = msg.get("results", [])
                if results:
                        render_chat_answer(
                            results,
                            show_score=msg.get("show_score", True),
                            backend_result=msg.get("backend_result"),
                        )
st.divider()
st.subheader("✏️ Prompt")

col_input, col_opts = st.columns([4, 1])

with col_opts:
    top_k = st.number_input("Top-K results", min_value=1, max_value=10, value=3)
    show_score = st.checkbox("Show similarity score", value=True)

with col_input:
    prefill = st.session_state.pop("prefill", "")
    query = st.text_area(
        "Your question",
        value=prefill,
        height=90,
        placeholder="e.g. D-case 進風開孔率建議值？",
        label_visibility="collapsed",
    )
    submitted = st.button("🔍 Search", type="primary", use_container_width=True)

# ── Handle submission ─────────────────────────────────────────────────────────
if submitted and query.strip():
    st.write("🔍submitted query: ", query)
    results = retriever.search(query, top_k=top_k)

    st.session_state.history.append({"role": "user", "content": query})

    if results:
        # Check if the top result triggers a backend computation
        top = results[0]
        backend_name = resolve_backend(
            top.get("核心設計問題 (Question)", ""),
            top.get("分類模組", ""),
        )
        backend_result = None
        if backend_name:
            try:
                mod = importlib.import_module(f"backends.{backend_name}")
                backend_result = mod.run(dict(top))
            except Exception as e:
                backend_result = None  # degrade gracefully

        st.session_state.history.append({
            "role": "assistant",
            "content": "",
            "results": results,
            "show_score": show_score,
            "backend_result": backend_result,
        })
    else:
        st.session_state.history.append({
            "role": "assistant",
            "content": "❌ No matching entries found in the FAQ knowledge base.",
            "results": [],
            "backend_result": None,
        })

    st.rerun()
