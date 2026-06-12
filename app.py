# Streamlit
import streamlit as st

# Standard library
import os
import logging
import time
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# Your RAG pipeline
from rag_engine import RAGPipeline, QueryResult, RetrieverResult
from rag_engine import setup_logging as rag_setup_logging

# Environment
from dotenv import load_dotenv

# Optional (for metrics visualization)
import pandas as pd
import plotly.express as px

# Load environment variables
load_dotenv()

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

APP_TITLE    = "🏁 Regulatory Compliance RAG Assistant"
APP_SUBTITLE = "Advanced Retrieval-Augmented Generation for Financial Compliance Teams"
APP_VERSION  = "1.0.0"
APP_DESCRIPTION = """
Compliance and legal risk teams in large financial institutions must frequently interpret dense, 
complex, and hierarchical regulatory circulars from multiple statutory bodies. This advanced 
RAG system provides semantic precision, hybrid lookup, and bulletproof source verification 
across multi-jurisdictional banking and securities frameworks.
"""

REGULATORY_BODIES = ["All Bodies", "RBI", "Basel Committee", "SEBI"]

LLM_OPTIONS = {
    "ollama": {
        "name": "Ollama (Local - Llama3)",
        "model": "llama3",
        "requires_api_key": False,
        "description": "Local LLM via Ollama - privacy-conscious, no API keys required"
    },
    "openai": {
        "name": "OpenAI (Cloud - GPT-4o-mini)",
        "model": "gpt-4o-mini",
        "requires_api_key": True,
        "description": "Cloud LLM via OpenAI - highest quality, requires a paid OpenAI API key"
    },
    "groq": {
        "name": "Groq (Cloud - Llama 3.3 70B)",
        "model": "llama-3.3-70b-versatile",
        "requires_api_key": True,
        "description": "Cloud LLM via Groq API - ultra-fast inference with Llama 3.3 70B, free-tier developer token friendly"
    }
}

STREAMLIT_PORT   = 8501
STREAMLIT_HOST   = "0.0.0.0"
DEFAULT_TOP_K    = 5
MAX_QUERY_LENGTH = 1000
CHAT_HISTORY_KEY    = "chat_history"
LLM_TYPE_KEY        = "llm_type"
REGULATORY_BODY_KEY = "regulatory_body"

CUSTOM_CSS = """
<style>
    .main h1 { font-size: 2.5rem !important; font-weight: 700 !important; color: #1f77b4 !important; }
    .main .sub-header { font-size: 1.2rem !important; color: #666 !important; margin-bottom: 20px !important; }
    .answer-box {
        background-color: #f0f8ff; border-left: 5px solid #1f77b4;
        padding: 15px; margin: 10px 0; border-radius: 5px;
        color: #111111; font-size: 1.05rem; line-height: 1.6;
    }
    .citation-box {
        background-color: #fff; border: 1px solid #ddd; padding: 10px;
        margin: 5px 0; border-radius: 3px; font-size: 0.9rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .citation-score { color: #28a745; font-weight: 600; font-size: 0.85rem; }
    .body-badge-rbi   { background-color: #e3f2fd; color: #1976d2; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .body-badge-basel { background-color: #f3e5f5; color: #7b1fa2; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .body-badge-sebi  { background-color: #fff3e0; color: #f57c00; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .stButton button  { background-color: #1f77b4 !important; color: white !important; font-weight: 600 !important; }
    .footer { text-align: center; font-size: 0.8rem; color: #888; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }
</style>
"""

# ==============================================================================
# HELPERS
# ==============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("StreamlitApp")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
        logger.addHandler(h)
    return logger


def configure_streamlit_page():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏁", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.markdown(f"<p class='sub-header'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)
    st.markdown(APP_DESCRIPTION)
    st.divider()


@st.cache_resource
def initialize_rag_pipeline(llm_type: str = "ollama") -> Optional[RAGPipeline]:
    """
    Initialize the RAG pipeline.
    """
    logger = setup_logging()
    logger.info(f"Initializing RAG pipeline with {llm_type}...")

    try:
        if llm_type == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ `OPENAI_API_KEY` not found in `.env` file or environment variables.")
                return None
            pipeline = RAGPipeline(
                llm_type="openai",
                openai_model_name="gpt-4o-mini"
            )

        elif llm_type == "groq":
            groq_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not groq_key:
                st.error("❌ Neither `GROQ_API_KEY` nor `OPENAI_API_KEY` found in `.env` file.")
                return None
            os.environ["OPENAI_API_KEY"] = groq_key
            pipeline = RAGPipeline(
                llm_type="openai",
                openai_model_name="llama-3.3-70b-versatile",
                openai_base_url="https://api.groq.com/openai/v1"
            )

        elif llm_type == "ollama":
            import ollama
            try:
                ollama.list()
            except Exception as ollama_err:
                logger.error(f"Ollama health check failed: {ollama_err}")
                st.error("❌ Unable to connect to Ollama. Please verify it is running (`ollama serve`).")
                return None
            pipeline = RAGPipeline(llm_type="ollama", llm_model_name="llama3")

        else:
            st.error(f"❌ Invalid LLM type: {llm_type}")
            return None

        logger.info("✅ RAG pipeline initialized successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Critical error initializing RAG pipeline: {e}")
        st.error(f"Critical initialization error: {e}")
        return None


def initialize_session_state():
    defaults = {
        CHAT_HISTORY_KEY:    [],
        LLM_TYPE_KEY:        "ollama",
        REGULATORY_BODY_KEY: "All Bodies",
        "last_result":       None,
        "is_loading":        False,
        "query_input_val":   ""
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_to_chat_history(query: str, answer: str, result: QueryResult):
    entry = {
        "query": query, "answer": answer,
        "citations_count": result.retrieved_count,
        "processing_time": result.total_processing_time_seconds,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state[CHAT_HISTORY_KEY].append(entry)
    if len(st.session_state[CHAT_HISTORY_KEY]) > 10:
        st.session_state[CHAT_HISTORY_KEY] = st.session_state[CHAT_HISTORY_KEY][-10:]


def clear_chat_history():
    st.session_state[CHAT_HISTORY_KEY] = []
    st.session_state["last_result"]    = None
    st.session_state["query_input_val"] = ""


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_sidebar() -> int:
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        llm_options_list = [
            {"value": "ollama", "label": LLM_OPTIONS["ollama"]["name"]},
            {"value": "openai", "label": LLM_OPTIONS["openai"]["name"]},
            {"value": "groq",   "label": LLM_OPTIONS["groq"]["name"]}
        ]
        
        # FIXED: Changed closing bracket ']' to curly brace '}'
        current_idx = {"ollama": 0, "openai": 1, "groq": 2}.get(st.session_state[LLM_TYPE_KEY], 0)

        llm_selection = st.selectbox(
            label="🤖 LLM Provider",
            options=llm_options_list,
            format_func=lambda x: x["label"],
            index=current_idx
        )
        new_llm_type = llm_selection["value"]
        if new_llm_type != st.session_state[LLM_TYPE_KEY]:
            st.session_state[LLM_TYPE_KEY] = new_llm_type
            st.rerun()

        selected_llm = LLM_OPTIONS[st.session_state[LLM_TYPE_KEY]]
        st.info(selected_llm["description"])

        if selected_llm["requires_api_key"]:
            is_groq   = st.session_state[LLM_TYPE_KEY] == "groq"
            key_label = "Groq API Key"    if is_groq else "OpenAI API Key"
            key_help  = "console.groq.com" if is_groq else "platform.openai.com"
            env_var   = "GROQ_API_KEY"    if is_groq else "OPENAI_API_KEY"

            api_key = st.text_input(key_label, type="password", help=f"Get your key at {key_help}")
            if api_key:
                os.environ[env_var]          = api_key
                os.environ["OPENAI_API_KEY"] = api_key

        selected_body = st.selectbox(
            label="🏛️ Regulatory Body Filter",
            options=REGULATORY_BODIES,
            index=REGULATORY_BODIES.index(st.session_state[REGULATORY_BODY_KEY])
        )
        if selected_body != st.session_state[REGULATORY_BODY_KEY]:
            st.session_state[REGULATORY_BODY_KEY] = selected_body

        top_k = st.slider("📊 Number of Citations (Top K)", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)

        # 📥 INGESTION SYSTEM FOR WEB APP IN CLOUD CONTAINERS
        st.markdown("---")
        st.subheader("📥 Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload Regulatory PDFs (RBI, SEBI, Basel III)", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Directly parse and embed fresh statutory frameworks into ChromaDB"
        )

        if uploaded_files and st.button("🏗️ Index Framework Documents", use_container_width=True):
            os.makedirs("data", exist_ok=True)
            pipeline = initialize_rag_pipeline(st.session_state[LLM_TYPE_KEY])
            
            if pipeline:
                success_count = 0
                for file in uploaded_files:
                    with st.spinner(f"Ingesting {file.name}..."):
                        file_path = os.path.join("data", file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        try:
                            if hasattr(pipeline, 'ingest_document'):
                                pipeline.ingest_document(file_path)
                            elif hasattr(pipeline, 'ingest'):
                                pipeline.ingest(file_path)
                            success_count += 1
                        except Exception as ingest_err:
                            st.error(f"Failed to ingest {file.name}: {ingest_err}")
                
                if success_count > 0:
                    st.success(f"🎯 Successfully indexed {success_count} framework files!")
                    st.session_state["last_result"] = None
                    time.sleep(1.5)
                    st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            clear_chat_history()
            st.rerun()

        if len(st.session_state[CHAT_HISTORY_KEY]) > 0:
            st.download_button(
                label="📄 Download Chat as JSON",
                data=json.dumps(st.session_state[CHAT_HISTORY_KEY], indent=2),
                file_name=f"compliance_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📱 About")
        st.markdown(f"**Version:** {APP_VERSION}")
        st.markdown("**Stack:** Streamlit · ChromaDB · BM25 · Cross-Encoder")
        st.markdown("**Corpus:** 11 Banking & Securities Framework PDFs")

        return top_k


def render_query_input() -> Tuple[Optional[str], bool]:
    with st.container():
        st.header("🔍 Query Regulatory Documents")
        st.markdown("---")

        query_text = st.text_area(
            label="Enter your compliance or regulatory question",
            value=st.session_state["query_input_val"],
            placeholder="Type your question here...\nExample: What is the Liquidity Coverage Ratio (LCR) requirement under Basel III?",
            height=150,
            max_chars=MAX_QUERY_LENGTH,
            key="main_query_text_area"
        )
        if query_text:
            st.caption(f"{len(query_text)}/{MAX_QUERY_LENGTH} characters")

        submit = st.button(
            "🚀 Generate Verified Answer",
            type="primary",
            use_container_width=True,
            disabled=len(query_text.strip()) == 0
        )
        return query_text.strip(), submit


def render_citation(citation: RetrieverResult, index: int):
    body_upper = getattr(citation.metadata, 'regulatory_body', 'SEBI').upper()
    if "RBI" in body_upper:
        badge_class, badge_name = "body-badge-rbi", "RBI"
    elif "BASEL" in body_upper:
        badge_class, badge_name = "body-badge-basel", "Basel Committee"
    else:
        badge_class, badge_name = "body-badge-sebi", "SEBI"

    doc_title = getattr(citation.metadata, 'doc_title',   'Regulatory Document')
    page_num  = getattr(citation.metadata, 'page_number', 'N/A')
    chunk_id  = getattr(citation.metadata, 'chunk_id',    'Unknown')

    st.markdown(f"""
        <div class='citation-box'>
            <div>
                <strong>[{index}] {doc_title}</strong>
                <span style='margin-left:15px;color:#555;'>Page: {page_num}</span>
                <span style='margin-left:15px;'><span class='{badge_class}'>{badge_name}</span></span>
            </div>
            <div class='citation-score'>Relevance: {citation.rerank_score:.4f}</div>
        </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📖 View Cited Text [{index}]"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Document:** {doc_title}")
            st.markdown(f"**Regulatory Body:** {badge_name}")
            st.markdown(f"**Chunk ID:** `{chunk_id}`")
        with c2:
            st.markdown(f"**Cross-Encoder Score:** `{citation.rerank_score:.4f}`")
            st.markdown(f"**Vector Score:** `{getattr(citation, 'vector_score', 0.0):.4f}`")
            st.markdown(f"**BM25 Score:** `{getattr(citation, 'bm25_score', 0.0):.4f}`")
        st.markdown("---")
        st.text_area("Document Segment", value=citation.chunk_text, height=180,
                     disabled=True, key=f"ta_cit_{index}_{chunk_id}")
        if st.button(f"📋 Copy Text [{index}]", key=f"cp_cit_{index}"):
            st.code(citation.chunk_text, language="text")


def render_answer(result: QueryResult):
    with st.container():
        st.markdown("---")
        st.header("📝 Generated Compliance Answer")

        st.markdown(f"<div class='answer-box'>{result.answers}</div>", unsafe_allow_html=True)

        with st.expander("📊 Pipeline Latency Breakdown"):
            c1, c2 = st.columns(2)
            c1.metric("Citations Retrieved", f"{result.retrieved_count} chunks")
            c2.metric("Total Latency",       f"{result.total_processing_time_seconds:.2f}s")

            if getattr(result, 'retrieval_stage_times', None):
                stage_df = pd.DataFrame([
                    {"Stage": s.replace("_", " ").title(), "Latency (s)": float(d)}
                    for s, d in result.retrieval_stage_times.items() if s != "total"
                ])
                if not stage_df.empty:
                    fig = px.bar(stage_df, x="Latency (s)", y="Stage", orientation="h",
                                 title="Per-Stage Latency", color="Latency (s)",
                                 color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("🔗 **Source Citations**")
        if getattr(result, 'citations', None):
            for i, cit in enumerate(result.citations):
                render_citation(cit, i + 1)
        else:
            st.warning("No citations could be isolated for this query.")

        if st.button("📋 Copy Answer", use_container_width=True, key="copy_answer_btn"):
            st.code(result.answers, language="text")


def render_chat_history():
    if not st.session_state[CHAT_HISTORY_KEY]:
        return

    with st.container():
        st.markdown("---")
        st.header("📜 Session History")
        for i, entry in enumerate(reversed(st.session_state[CHAT_HISTORY_KEY])):
            real_idx = len(st.session_state[CHAT_HISTORY_KEY]) - i
            with st.expander(f"💬 Query #{real_idx}: {entry['query'][:80]}..."):
                st.markdown(f"**Query:** {entry['query']}")
                st.markdown(f"**Answer:**\n\n{entry['answer']}")
                st.caption(f"Citations: {entry['citations_count']} | Time: {entry['processing_time']:.2f}s | {entry['timestamp']}")
                if st.button("🔄 Reuse Query", key=f"reuse_{real_idx}"):
                    st.session_state["query_input_val"] = entry['query']
                    st.rerun()


def render_example_queries():
    with st.container():
        st.markdown("---")
        st.header("💡 Example Queries")
        st.markdown("Click any query to load it into the input box:")

        examples = {
            "RBI": [
                "What is the purpose of Master Directions on Relief/Savings Bonds?",
                "What are the liquidity facilities required for Primary Dealers?",
                "How is agency commission calculated for government business?",
                "What are the statutory liabilities for agency banks disbursing pension?"
            ],
            "Basel Framework": [
                "What is the Liquidity Coverage Ratio (LCR) requirement?",
                "What is the Net Stable Funding Ratio (NSFR) percentage?",
                "What are the credit risk capital requirements in Basel III?",
                "How many days of HQLA must banks hold for LCR coverage?"
            ],
            "SEBI": [
                "What is UPSI under SEBI PIT Regulations?",
                "How is 'Connected Person' defined under Insider Trading Regulations?",
                "What are the promoter lock-in periods under SEBI ICDR?",
                "What are the corporate disclosure timelines under SEBI LODR?"
            ]
        }

        for idx, (category, queries) in enumerate(examples.items()):
            with st.expander(f"📂 {category}"):
                for query in queries:
                    if st.button(query, use_container_width=True, key=f"ex_{idx}_{hash(query)}"):
                        st.session_state["query_input_val"] = query
                        st.rerun()


def handle_query_error(error: Exception, query: str):
    logger = setup_logging()
    logger.error(f"Query failure for '{query}': {error}", exc_info=True)

    st.error("❌ A processing error occurred during document retrieval.")
    with st.expander("🔍 Error Details"):
        st.code(f"Type: {type(error).__name__}\nMessage: {error}", language="text")

    err_str = str(error).lower()
    st.warning("💡 Suggested Fix:")
    if "api_key" in err_str or "groq" in err_str:
        st.markdown("- Check your API key in the sidebar or `.env` file.")
    elif "ollama" in err_str:
        st.markdown("- Run `ollama serve` and ensure `llama3` is pulled.")
    elif "chroma" in err_str or "collection" in err_str:
        st.markdown("- Use the sidebar ingestion portal to rebuild the vector index.")
    else:
        st.markdown("- Confirm uploaded PDFs match the targeted indexing formats.")

    if st.button("🔄 Retry", key="retry_btn"):
        st.rerun()


# ==============================================================================
# MAIN ENGINE
# ==============================================================================

def main():
    configure_streamlit_page()
    initialize_session_state()
    top_k = render_sidebar()

    query_text, is_submitted = render_query_input()

    if is_submitted and query_text:
        st.session_state["is_loading"] = True
        current_llm   = st.session_state[LLM_TYPE_KEY]
        filter_body   = st.session_state[REGULATORY_BODY_KEY]
        actual_filter = None if filter_body == "All Bodies" else filter_body

        pipeline = initialize_rag_pipeline(current_llm)
        if pipeline is None:
            st.error("RAG Pipeline failed to initialize.")
            st.session_state["is_loading"] = False
            return

        with st.spinner("🤖 Retrieving and synthesizing regulatory context..."):
            try:
                t0     = time.time()
                result = pipeline.query(query_text=query_text, regulatory_body=actual_filter, final_top_k=top_k)
                elapsed = time.time() - t0

                if not getattr(result, 'total_processing_time_seconds', None):
                    result.total_processing_time_seconds = elapsed

                st.session_state["last_result"]     = result
                st.session_state["is_loading"]      = False
                st.session_state["query_input_val"] = ""
                add_to_chat_history(query_text, result.answers, result)
                st.rerun()

            except Exception as e:
                st.session_state["is_loading"] = False
                handle_query_error(e, query_text)

    if st.session_state["last_result"] is not None:
        render_answer(st.session_state["last_result"])

    render_chat_history()
    render_example_queries()

    st.markdown("---")
    st.markdown(f"""
        <div class='footer'>
            <p><strong>🏁 Regulatory Compliance RAG Assistant</strong> | v{APP_VERSION}</p>
            <p>ChromaDB · BM25 · Cross-Encoder · Streamlit</p>
            <p>© 2026 Financial Risk &amp; Compliance Systems Group.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()