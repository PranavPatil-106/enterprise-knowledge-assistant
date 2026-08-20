"""Streamlit web interface for the Enterprise Knowledge Assistant."""

import streamlit as st
from src.application import run_workflow
from src.config import get_settings

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom enterprise styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    .pipeline-badge {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0F172A;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    .answer-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 1.1rem 1.3rem;
        border-radius: 0 6px 6px 0;
        color: #1E293B;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load application settings
settings = get_settings()

# Main page header
st.markdown(
    """
    <div class="main-title">
        <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="16" height="20" x="4" y="2" rx="2" ry="2"/>
            <path d="M9 22v-4h6v4"/>
            <path d="M8 6h.01"/>
            <path d="M16 6h.01"/>
            <path d="M12 6h.01"/>
            <path d="M12 10h.01"/>
            <path d="M12 14h.01"/>
            <path d="M16 10h.01"/>
            <path d="M16 14h.01"/>
            <path d="M8 10h.01"/>
            <path d="M8 14h.01"/>
        </svg>
        Enterprise Knowledge Assistant
    </div>
    <div class="subtitle">
        Search enterprise policy documents, synthesize grounded answers, and verify accuracy with RAGAS evaluation.
    </div>
    <div class="pipeline-badge">
        Multi-Agent Workflow: Retriever Agent &rarr; Response Agent &rarr; Evaluator Agent
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick sample question buttons
st.markdown("<div class='section-header'>Suggested Queries</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
sample_query = None

with col1:
    if st.button("Remote Work Core Hours", use_container_width=True):
        sample_query = "What are the remote work core hours?"
with col2:
    if st.button("Annual Leave Entitlement", use_container_width=True):
        sample_query = "How many days of annual leave do employees receive per year?"
with col3:
    if st.button("IT Password Security Rules", use_container_width=True):
        sample_query = "What are the IT password complexity and change requirements?"

# User question form
with st.form(key="question_form", clear_on_submit=False):
    user_input = st.text_input(
        "Search policy knowledge base:",
        placeholder="Enter your question (e.g., What are the remote work core hours?)",
    )
    submit_button = st.form_submit_button("Search Knowledge Base", type="primary")

query_to_run = sample_query if sample_query else (user_input if submit_button else None)

if query_to_run:
    if not query_to_run.strip():
        st.warning("Please enter a question to search.")
    else:
        st.markdown(f"**Query:** `{query_to_run.strip()}`")
        with st.spinner("Executing multi-agent workflow and RAGAS evaluation..."):
            try:
                result = run_workflow(query_to_run.strip(), settings=settings)

                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                evaluation = result.get("evaluation", {})

                # Prominent answer display
                st.markdown("<div class='section-header'>Grounded Answer</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)

                # Authoritative source documents expander
                with st.expander(f"Cited Source Documents ({len(sources)})", expanded=False):
                    if sources:
                        for src in sources:
                            st.markdown(f"- `{src}`")
                    else:
                        st.write("No source documents identified.")

                # Quality evaluation display
                st.markdown("<div class='section-header'>Quality Evaluation (RAGAS)</div>", unsafe_allow_html=True)
                if isinstance(evaluation, dict):
                    faithfulness = evaluation.get("faithfulness", 0.0)
                    relevancy = evaluation.get("answer_relevancy", 0.0)
                    interpretation = evaluation.get(
                        "interpretation", "No interpretation provided."
                    )
                else:
                    faithfulness = 0.0
                    relevancy = 0.0
                    interpretation = str(evaluation)

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(
                        label="Faithfulness (Groundedness)",
                        value=f"{faithfulness * 100:.2f}%",
                    )
                with metric_col2:
                    st.metric(
                        label="Answer Relevancy",
                        value=f"{relevancy * 100:.2f}%",
                    )

                st.info(f"**Evaluation Summary:** {interpretation}")

            except ValueError as e:
                st.warning(str(e))
            except FileNotFoundError as e:
                st.error(
                    f"ChromaDB Index or document not found: {e}. Please run `uv run python -m src.ingest` first."
                )
            except Exception as e:
                err_msg = str(e)
                if "not found" in err_msg.lower() and ("ollama" in err_msg.lower() or "404" in err_msg):
                    st.error(
                        f"Ollama model '{settings.llm_model}' not found in local Ollama instance. "
                        f"Please run `ollama pull {settings.llm_model}` in your terminal to download it."
                    )
                elif "429" in err_msg or "resource_exhausted" in err_msg.lower():
                    st.warning(
                        f"Rate limit reached for model '{settings.llm_model}'. "
                        "Please wait ~15 seconds for the free-tier quota window to reset and re-submit."
                    )
                else:
                    st.error(f"Workflow execution error: {e}")
