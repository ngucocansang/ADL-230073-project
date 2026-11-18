"""
Web interface of the CS Major Handbook chatbot (Task 4).

Run:  streamlit run app.py         ->  http://localhost:8501
"""
import html
import time
from dataclasses import replace
from datetime import datetime

import streamlit as st

import config
from rag import NOT_FOUND_MESSAGE, RAGEngine, RAGSettings, ollama_models

st.set_page_config(page_title="CS Handbook Assistant", page_icon="📘", layout="wide")

EXAMPLES = [
    "What are the prerequisites for CS311 Advanced Deep Learning?",
    "How many credits does the CS major require?",
    "What are the eligibility criteria for the Capstone?",
    "What jobs have CS graduates obtained?",
    "Which courses are cross-listed with Computer Science?",
]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.src-meta {color: #6b7280; font-size: 0.82rem; white-space: pre-wrap; line-height: 1.35;}
.chip {display:inline-block; padding:2px 8px; margin:0 4px 4px 0; border-radius:10px;
       background:#eef2ff; color:#3730a3; font-size:0.78rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Engine (cached: the index is loaded / built only once per setting)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the handbook index (first run embeds the PDF)...")
def get_engine(mode: str, top_k: int, history_turns: int) -> RAGEngine:
    s = replace(RAGSettings(), retrieval_mode=mode, top_k=top_k, history_turns=history_turns)
    return RAGEngine(s, verbose=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    models = ollama_models(config.OLLAMA_URL)
    if models:
        st.success(f"Ollama connected · {len(models)} models", icon="🟢")
    else:
        st.error(f"Cannot reach Ollama at {config.OLLAMA_URL}", icon="🔴")
        st.stop()

    llms = [m for m in models if "embed" not in m] or [config.LLM_MODEL]
    default = config.LLM_MODEL if config.LLM_MODEL in llms else llms[0]
    model = st.selectbox("Language model", llms, index=llms.index(default))
    temperature = st.slider("Temperature", 0.0, 1.0, float(config.TEMPERATURE), 0.05)
    top_k = st.slider("Retrieved chunks (top-K)", 1, 10, config.TOP_K)
    mode = st.radio("Retrieval mode", ["hybrid", "vector", "bm25"],
                    index=["hybrid", "vector", "bm25"].index(config.RETRIEVAL_MODE),
                    horizontal=True, help="hybrid = semantic vectors + BM25 keywords (RRF)")
    use_history = st.toggle("Use conversation history", True,
                            help="Lets you ask follow-ups such as 'what are its prerequisites?'")
    show_sources = st.toggle("Show retrieved sources", True)

    engine = get_engine(mode, top_k, config.HISTORY_TURNS if use_history else 0)
    info = engine.index_info
    st.divider()
    st.caption("📄 Knowledge base")
    for f in info.get("files", {}):
        st.markdown(f"<span class='chip'>{f}</span>", unsafe_allow_html=True)
    st.caption(f"{info.get('num_chunks', '?')} chunks · embeddings: {engine.s.embed_model}")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    transcript = "\n\n".join(
        f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}"
        for m in st.session_state.get("messages", []))
    c2.download_button("⬇️ Export", transcript or "(empty)", use_container_width=True,
                       file_name=f"chat_{datetime.now():%Y%m%d_%H%M}.md")


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
st.title("📘 CS Major Handbook Assistant")
st.caption("Ask about program requirements, courses, prerequisites, learning outcomes and "
           "academic policies of the B.S. Computer Science at Fulbright University Vietnam. "
           "Runs 100% locally (Ollama + Chroma).")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources, meta):
    with st.expander(f"📚 Sources ({len(sources)}) · retrieval {meta.get('retrieve_s', 0)} s · "
                     f"generation {meta.get('generate_s', 0)} s · {meta.get('model', '')}"):
        for s in sources:
            page = f"page {s['page']}" if s["page"] else ""
            st.markdown(f"**[{s['rank']}] {s['source']}** · {page} · similarity {s['score']:.2f}")
            snippet = html.escape(s["content"][:600]) + ("…" if len(s["content"]) > 600 else "")
            st.markdown(f"<div class='src-meta'>{snippet}</div>", unsafe_allow_html=True)
            st.write("")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📘"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            render_sources(msg["sources"], msg.get("meta", {}))

# Example questions on an empty chat
clicked = None
if not st.session_state.messages:
    st.markdown("##### Try an example")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            clicked = ex

question = st.chat_input("Ask a question about the CS major handbook…") or clicked

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("Searching the handbook…"):
            prep = engine.prepare(question, history if use_history else None)
        t0 = time.time()
        try:
            answer = st.write_stream(engine.stream(prep, model=model, temperature=temperature))
        except Exception as exc:
            answer = f"⚠️ The language model failed: {exc}"
            st.error(answer)
        meta = {"retrieve_s": prep["retrieve_s"], "generate_s": round(time.time() - t0, 1),
                "model": model}
        if prep["used_fallback"]:
            st.info(f"No passage was relevant enough (best similarity "
                    f"{prep['best_similarity']:.2f} < {engine.s.score_threshold}).")
        if show_sources and not prep["used_fallback"]:
            render_sources(prep["sources"], meta)

    st.session_state.messages.append({
        "role": "assistant", "content": answer if isinstance(answer, str) else NOT_FOUND_MESSAGE,
        "sources": [] if prep["used_fallback"] else prep["sources"], "meta": meta})
    if clicked:
        st.rerun()
