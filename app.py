"""Streamlit web interface for the Enterprise Knowledge Assistant."""

import streamlit as st
from src.application import run_workflow
from src.config import get_settings

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
)

# Load application settings
settings = get_settings()

# Sidebar: Safe configuration details only
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    st.markdown("**LLM Provider:**")
    st.code(settings.llm_provider, language="text")
    st.markdown("**LLM Model:**")
    st.code(settings.llm_model, language="text")
    st.markdown("**Embedding Model (Fixed):**")
    st.code("gemini-embedding-001", language="text")
    st.markdown("**LangSmith Tracing:**")
    if settings.langsmith_tracing:
        st.success("Enabled")
    else:
        st.info("Disabled")
    st.caption("Configuration is loaded securely from the environment.")

# Main page header
st.title("🏢 Enterprise Knowledge Assistant")
st.markdown(
    "Search internal enterprise policy documents, generate grounded answers, "
    "evaluate answer quality with RAGAS, and retrieve authoritative document context via Filesystem MCP."
)
st.caption("Workflow: Retriever Agent -> Response Agent -> Evaluator Agent")
st.markdown("---")

# Sample question buttons
st.markdown("##### Quick Sample Questions")
col1, col2, col3 = st.columns(3)
sample_query = None

with col1:
    if st.button("🕒 Remote Work Core Hours", use_container_width=True):
        sample_query = "What are the remote work core hours?"
with col2:
    if st.button("🌴 Annual Leave Entitlement", use_container_width=True):
        sample_query = "How many days of annual leave do employees receive per year?"
with col3:
    if st.button("🔒 IT Password Rules", use_container_width=True):
        sample_query = "What are the IT password complexity and change requirements?"

# User question form
with st.form(key="question_form", clear_on_submit=False):
    user_input = st.text_input(
        "Enter your question about enterprise policies:",
        placeholder="e.g., What are the remote work core hours?",
    )
    submit_button = st.form_submit_button("Submit Question", type="primary")

query_to_run = sample_query if sample_query else (user_input if submit_button else None)

if query_to_run:
    if not query_to_run.strip():
        st.warning("Please enter a question.")
    else:
        st.markdown(f"**Question:** {query_to_run.strip()}")
        with st.spinner("Processing through Retriever Agent, Response Agent, and Evaluator Agent..."):
            try:
                result = run_workflow(query_to_run.strip(), settings=settings)

                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                evaluation = result.get("evaluation", {})

                # Prominent answer display
                st.subheader("💡 Answer")
                st.markdown(answer)

                # Source documents expander
                with st.expander(f"📁 Source Documents ({len(sources)})", expanded=False):
                    if sources:
                        for src in sources:
                            st.markdown(f"- `{src}`")
                    else:
                        st.write("No source documents identified.")

                # Quality evaluation display
                st.subheader("📊 Quality Evaluation (RAGAS)")
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
                st.warning(f"⚠️ {e}")
            except FileNotFoundError as e:
                st.error(
                    f"ChromaDB Index or document not found: {e}. Please run `uv run python -m src.ingest` first."
                )
            except Exception as e:
                st.error(f"Workflow execution error: {e}")
