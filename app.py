import streamlit as st
from src.config import get_settings
from src.graph.workflow import run_workflow


st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #334155;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()

st.markdown('<div class="main-title">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'Autonomous RAG system for internal policy synthesis, authoritative source retrieval via FastMCP, '
    'and automated quality evaluation via RAGAS.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge">Pipeline: Supervisor &rarr; Retriever &rarr; Response &rarr; Evaluator</span>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-header">Sample Inquiries</div>', unsafe_allow_html=True)
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

with st.form(key="question_form", clear_on_submit=False):
    user_input = st.text_input(
        "Enter your question regarding enterprise policies:",
        placeholder="e.g., What are the remote work core hours?",
    )
    submit_button = st.form_submit_button("Submit Question", type="primary")

query_to_run = sample_query if sample_query else (user_input if submit_button else None)

if query_to_run:
    if not query_to_run.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown(f"**Question:** {query_to_run.strip()}")
        with st.spinner("Processing inquiry with Supervisor Agent..."):
            try:
                result = run_workflow(query_to_run.strip(), settings=settings)

                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                evaluation = result.get("evaluation", {})

                st.markdown('<div class="section-header">Synthesized Response</div>', unsafe_allow_html=True)
                st.markdown(answer)

                with st.expander(f"Authoritative Sources ({len(sources)})", expanded=False):
                    if sources:
                        for src in sources:
                            st.markdown(f"- `{src}`")
                    else:
                        st.write("No source documents identified.")

                st.markdown('<div class="section-header">Quality Evaluation (RAGAS)</div>', unsafe_allow_html=True)
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
                st.warning(f"Validation Warning: {e}")
            except FileNotFoundError as e:
                st.error(
                    f"ChromaDB index or document not found: {e}. Please run `uv run python -m src.ingest` first."
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
