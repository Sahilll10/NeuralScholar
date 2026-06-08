import streamlit as st
import httpx
import json
import time
from typing import List, Dict, Any

# ─── Configuration ──────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"
STREAM_URL = f"{API_BASE}/query/stream"
QUERY_URL = f"{API_BASE}/query"

st.set_page_config(
    page_title="NeuralScholar",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .citation-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 14px;
    }
    .metric-badge {
        background: #e9ecef;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        color: #495057;
    }
    .answer-text {
        line-height: 1.7;
        font-size: 16px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialization ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_citations" not in st.session_state:
    st.session_state.current_citations = []
if "latency_history" not in st.session_state:
    st.session_state.latency_history = []


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 NeuralScholar")
    st.caption("ML Research Literature Intelligence")
    st.divider()

    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Documents to retrieve", 1, 10, 5)
    retrieval_mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"])
    use_hyde = st.toggle("Enable HyDE", value=True)
    use_cache = st.toggle("Use query cache", value=True)
    use_streaming = st.toggle("Stream response", value=True)

    st.divider()
    st.subheader("📚 Ingest Papers")
    ingest_query = st.text_input("ArXiv search query")
    max_results = st.number_input("Max papers", 10, 500, 50)
    if st.button("📥 Ingest from ArXiv", type="secondary"):
        with st.spinner("Ingesting papers..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/ingest",
                    json={"source": "arxiv", "query": ingest_query, "max_results": max_results},
                    timeout=300
                )
                data = resp.json()
                st.success(
                    f"✅ {data['documents_processed']} docs → "
                    f"{data['chunks_created']} chunks"
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    if st.session_state.current_citations:
        st.subheader("📖 Source Papers")
        for i, cit in enumerate(st.session_state.current_citations, 1):
            with st.expander(f"[{i}] {cit.get('title', 'Unknown')[:50]}...", expanded=False):
                st.write(f"**Authors:** {cit.get('authors', 'Unknown')}")
                st.write(f"**Year:** {cit.get('year', 'Unknown')}")
                if cit.get("arxiv_id"):
                    st.write(f"**ArXiv:** {cit.get('arxiv_id')}")
                if cit.get("paper_url"):
                    st.markdown(f"[🔗 View Paper]({cit['paper_url']})")
                st.write(f"**Rerank score:** {cit.get('rerank_score', 'N/A')}")

    if st.session_state.latency_history:
        st.divider()
        st.subheader("⏱️ Latency")
        last = st.session_state.latency_history[-1]
        col1, col2 = st.columns(2)
        col1.metric("Retrieval", f"{last.get('retrieval_ms', 0):.0f}ms")
        col2.metric("Generation", f"{last.get('generation_ms', 0):.0f}ms")


# ─── Main chat area ─────────────────────────────────────────────────────────────
st.title("🔬 NeuralScholar — ML Research Q&A")
st.caption("Ask questions about machine learning research. Powered by semantic RAG + GPT-4o.")

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def stream_query(query: str) -> str:
    """Stream query response from FastAPI SSE endpoint."""
    full_text = ""
    citations = []

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            STREAM_URL,
            json={
                "query": query,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "use_hyde": use_hyde,
                "use_cache": use_cache
            }
        ) as response:
            message_placeholder = st.empty()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "token" in data:
                        full_text += data["token"]
                        message_placeholder.markdown(full_text + "▌")
                    elif "citations" in data:
                        citations = data["citations"]
                except json.JSONDecodeError:
                    pass

    if citations:
        st.session_state.current_citations = citations

    return full_text


def sync_query(query: str) -> Dict[str, Any]:
    """Synchronous query for non-streaming mode."""
    resp = httpx.post(
        QUERY_URL,
        json={
            "query": query,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "use_hyde": use_hyde,
            "use_cache": use_cache
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()


# ─── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about ML research (e.g. 'How does RAG work?' or 'Compare BERT and GPT')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t_start = time.time()
        try:
            if use_streaming:
                answer = stream_query(prompt)
                st.markdown(answer)
            else:
                with st.spinner("Thinking..."):
                    data = sync_query(prompt)
                answer = data["answer"]
                st.markdown(answer)
                if data.get("citations"):
                    st.session_state.current_citations = data["citations"]
                st.session_state.latency_history.append({
                    "retrieval_ms": data.get("retrieval_latency_ms", 0),
                    "generation_ms": data.get("generation_latency_ms", 0)
                })

        except Exception as e:
            answer = f"❌ Error: {str(e)}"
            st.error(answer)

        e2e_ms = (time.time() - t_start) * 1000
        st.caption(f"⏱️ {e2e_ms:.0f}ms | Mode: {retrieval_mode} | HyDE: {use_hyde}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()