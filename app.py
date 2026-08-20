"""Streamlit web interface for the Enterprise Knowledge Assistant."""

import streamlit as st
from src.application import run_workflow
from src.config import get_settings

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load application settings
settings = get_settings()

# Custom professional styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0F172A;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.35rem;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 1.25rem;
        border-radius: 4px;
        margin-bottom: 1.25rem;
        font-size: 1rem;
        line-height: 1.6;
        color: #1E293B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page header
st.markdown('<div class="main-title">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Multi-agent RAG system for querying internal policies, '
    'retrieving verified context via Filesystem MCP, and evaluating answer faithfulness using RAGAS.</div>',
    unsafe_allow_html=True,
)

# Sample question buttons
st.markdown("**Sample Inquiries:**")
col1, col2, col3 = st.columns(3)
sample_query = None

with col1:
    if st.button("Remote Work: Core Hours", use_container_width=True):
        sample_query = "What are the remote work core hours?"
with col2:
    if st.button("Leave Policy: Annual Entitlement", use_container_width=True):
        sample_query = "How many days of annual leave do employees receive per year?"
with col3:
    if st.button("IT Security: Password Requirements", use_container_width=True):
        sample_query = "What are the IT password complexity and change requirements?"

# User question form
with st.form(key="question_form", clear_on_submit=False):
    user_input = st.text_input(
        "Policy Search Query:",
        placeholder="Enter your question about enterprise policies...",
    )
    submit_button = st.form_submit_button("Submit Query", type="primary")

query_to_run = sample_query if sample_query else (user_input if submit_button else None)

if query_to_run:
    if not query_to_run.strip():
        st.warning("Please enter a valid query.")
    else:
        st.markdown(f"**Active Query:** {query_to_run.strip()}")
        with st.spinner("Processing through Retriever Agent, Response Agent, and Evaluator Agent..."):
            try:
                result = run_workflow(query_to_run.strip(), settings=settings)

                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                evaluation = result.get("evaluation", {})

                # Prominent answer display
                st.markdown('<div class="section-header">Generated Response</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                # Source documents expander
                with st.expander(f"Authoritative Sources ({len(sources)})", expanded=False):
                    if sources:
                        for src in sources:
                            st.markdown(f"- `{src}`")
                    else:
                        st.write("No authoritative source documents identified.")

                # Quality evaluation display
                st.markdown('<div class="section-header">Evaluation Metrics (RAGAS)</div>', unsafe_allow_html=True)
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
                        label="Faithfulness",
                        value=f"{faithfulness * 100:.2f}%",
                    )
                with metric_col2:
                    st.metric(
                        label="Answer Relevancy",
                        value=f"{relevancy * 100:.2f}%",
                    )

                st.info(f"Summary: {interpretation}")

            except ValueError as e:
                st.warning(f"Validation notice: {e}")
            except FileNotFoundError as e:
                st.error(
                    f"Index or document not found: {e}. Please run `uv run python -m src.ingest` first."
                )
            except Exception as e:
                err_msg = str(e)
                if "not found" in err_msg.lower() and ("ollama" in err_msg.lower() or "404" in err_msg):
                    st.error(
                        f"Ollama model '{settings.llm_model}' not found in local instance. "
                        f"Please run `ollama pull {settings.llm_model}` in your terminal."
                    )
                elif "429" in err_msg or "resource_exhausted" in err_msg.lower():
                    st.warning(
                        f"Rate limit reached for model '{settings.llm_model}'. "
                        "Please wait ~15 seconds for the quota window to reset and re-submit."
                    )
                else:
                    st.error(f"Workflow execution error: {e}")
