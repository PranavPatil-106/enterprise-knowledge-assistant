"""Streamlit web interface for the Enterprise Knowledge Assistant."""

import streamlit as st
from src.application import run_workflow
from src.config import get_settings

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide",
)

# Custom styling for a clean, modern, enterprise aesthetic without casual emojis
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #334155;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 6px;
        background-color: #f1f5f9;
        color: #475569;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load application settings
settings = get_settings()

# Main page header
st.markdown('<div class="main-title">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    "Search internal enterprise policy documents, synthesize grounded answers, "
    "evaluate answer quality with RAGAS, and retrieve authoritative context via Filesystem MCP."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="badge">Pipeline: Retriever Agent &rarr; Response Agent &rarr; Evaluator Agent</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Sample question buttons
st.markdown("##### Quick Reference Queries")
col1, col2, col3 = st.columns(3)
sample_query = None

with col1:
    if st.button("Remote Work Core Hours", use_container_width=True):
        sample_query = "What are the remote work core hours?"
with col2:
    if st.button("Annual Leave Entitlement", use_container_width=True):
        sample_query = "How many days of annual leave do employees receive per year?"
with col3:
    if st.button("IT Password Policy", use_container_width=True):
        sample_query = "What are the IT password complexity and change requirements?"

# User question form
with st.form(key="question_form", clear_on_submit=False):
    user_input = st.text_input(
        "Enter your policy inquiry:",
        placeholder="e.g., What are the remote work core hours?",
    )
    submit_button = st.form_submit_button("Submit Inquiry", type="primary")

query_to_run = sample_query if sample_query else (user_input if submit_button else None)

if query_to_run:
    if not query_to_run.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown(f"**Query:** {query_to_run.strip()}")
        with st.spinner("Processing inquiry through multi-agent pipeline..."):
            try:
                result = run_workflow(query_to_run.strip(), settings=settings)

                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                evaluation = result.get("evaluation", {})

                # Prominent answer display
                st.subheader("Synthesized Response")
                st.markdown(answer)

                # Source documents expander
                with st.expander(f"Authoritative Sources ({len(sources)})", expanded=False):
                    if sources:
                        for src in sources:
                            st.markdown(f"- `{src}`")
                    else:
                        st.write("No source documents identified.")

                # Quality evaluation display
                st.subheader("Quality Metrics (RAGAS)")
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

                st.info(f"**Evaluation Summary:** {interpretation}")

            except ValueError as e:
                st.warning(f"{e}")
            except FileNotFoundError as e:
                st.error(
                    f"Vector index or document not found: {e}. Please run `uv run python -m src.ingest` first."
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
