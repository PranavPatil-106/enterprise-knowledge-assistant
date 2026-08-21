# Enterprise Knowledge Assistant

Enterprise Knowledge Assistant is a multi-agent Retrieval-Augmented Generation (RAG) system built to search, synthesize, and evaluate enterprise policy documents. It addresses the common challenges of enterprise search—hallucinations, untracked context sources, and unverifiable answers—by orchestrating a **Supervisor Agent** workflow in LangGraph that incorporates authoritative file reading via FastMCP, automated answer evaluation via RAGAS, observability via LangSmith, input guardrails with built-in LangChain `PIIMiddleware`, and an interactive Streamlit web interface.

---

## Architecture Overview

The system uses a **Supervisor Agent** pattern in LangGraph to coordinate specialized worker nodes:

```text
               ┌────────────────────────┐
        START ─►    Supervisor Agent    ◄────────┐
               │  (Dynamic State Router)│        │
               └───────────┬────────────┘        │
                           │                     │
           ┌───────────────┼───────────────┐     │
           ▼               ▼               ▼     │
      [Retriever]     [Response]      [Evaluator]│
    (ChromaDB + MCP) (Grounded LLM)  (RAGAS Eval)│
           │               │               │     │
           └───────────────┴───────────────┴─────┘
                    (Reports back to Supervisor)
```

### Execution Pipeline Flow

1. **Input Guardrails & PII Middleware**: Validates query length, strips whitespace, and processes sensitive personal information (Emails, Phone numbers, URLs, IPs, SSNs) via LangChain's built-in `PIIMiddleware`.
2. **Supervisor Agent**: Inspects the workflow state and dynamically routes execution to the appropriate node.
3. **Retriever Node**: Queries ChromaDB vector storage using fixed Gemini embeddings, identifies the top-ranked source file, and invokes the custom FastMCP server (`read_policy_document`) to retrieve authoritative source context.
4. **Response Node**: Synthesizes a factual, strictly grounded answer based exclusively on the assembled context using the configured Chat LLM.
5. **Evaluator Node**: Assesses the generated answer against the retrieved context and user question using RAGAS (*Faithfulness* and *Answer Relevancy*). If faithfulness is below 70%, the Supervisor can trigger a grounded regeneration retry.
6. **Workflow Completion**: Once all tasks are complete, the Supervisor Agent routes to `END` and returns the final answer, source citations, and evaluation metrics.

---

## Guardrails and Privacy Middleware

The application implements guardrails using LangChain's built-in `PIIMiddleware` (`src/guardrails.py`):

- **Input Validation**: Rejects empty queries and restricts question length to a maximum of 500 characters.
- **Built-in PII Strategies**: Supports modular redaction, masking, and blocking policies:
  - **`redact`**: Replaces sensitive data with category placeholders (e.g. `[REDACTED_EMAIL]`, `[REDACTED_URL]`).
  - **`mask`**: Obscures sensitive digits or tokens (e.g. `****-****-****-0002` for credit cards, `user@****.com` for emails, masked SSNs).
  - **`hash`**: Securely hashes sensitive identifiers (e.g. IP addresses).
  - **`block`**: Raises a `ValueError` / `PIIDetectionError` if prohibited personal information is submitted.
- **Sandboxed MCP Tooling**: The FastMCP client is strictly read-only (`read_policy_document`), confined exclusively to `data/raw/`, and validates file path containment prior to execution.

---

## Technology Stack

- **Python**: Core programming language (Python 3.12+).
- **uv / pip**: Package and environment management.
- **FastMCP**: Python framework used to build and run the custom Model Context Protocol (MCP) server over `stdio`.
- **LangGraph**: Multi-agent state graph orchestration and dynamic supervisor routing.
- **LangChain**: Document abstractions, vector store integrations, LLM interfaces, and `PIIMiddleware`.
- **Chat LLMs**: Configurable chat models supporting **Gemini** (default), **OpenAI**, and **Ollama**.
- **Embedding Model**: Fixed **Gemini Embeddings** (`gemini-embedding-001`).
- **ChromaDB**: Persistent vector database for embedding storage and similarity retrieval.
- **RAGAS**: Automated evaluation framework for Faithfulness and Answer Relevancy.
- **Streamlit**: Web application frontend for interactive querying and metric visualization.
- **LangSmith**: Telemetry, execution tracing, and agent debugging.

---

## Setup Instructions

### 1. Clone the Repository
```powershell
git clone <repo-url>
cd "AI Enterprise Project"
```

### 2. Install Dependencies
Using `uv`:
```powershell
uv sync
```
Or using standard `pip`:
```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Configure your credentials in `.env`:

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | Chat model provider (`gemini`, `openai`, `ollama`) | `gemini` |
| `LLM_MODEL` | Chat model identifier | `gemini-2.5-flash` |
| `GOOGLE_API_KEY` | Google AI Studio API key (required for embeddings and Gemini LLM) | `AIzaSy...` |
| `OPENAI_API_KEY` | OpenAI API key (required only if `LLM_PROVIDER=openai`) | `sk-...` |
| `KNOWLEDGE_BASE_PATH` | Directory containing enterprise policy documents | `data/raw` |
| `CHROMA_PERSIST_DIRECTORY`| Directory for ChromaDB vector storage | `data/chroma` |
| `MCP_ALLOWED_DIRECTORY` | Directory allowed for FastMCP access | `data/raw` |
| `LANGSMITH_TRACING` | Enable or disable LangSmith tracing (`true`/`false`) | `true` |
| `LANGSMITH_API_KEY` | LangSmith API key for trace ingestion | `lsv2_pt_...` |
| `LANGSMITH_PROJECT` | LangSmith project name for traces | `enterprise-knowledge-assistant` |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint URL | `https://api.smith.langchain.com` |

---

## Execution Instructions

### 1. Ingest & Index Knowledge Documents
Load policy documents from `data/raw/`, chunk the text, compute Gemini embeddings, and store them in ChromaDB:
```powershell
uv run python -m src.ingest
```

### 2. Run the Multi-Agent Graph via CLI
Execute the full LangGraph pipeline from the command line:
```powershell
uv run python -m src.run_graph "What are the remote work core hours?"
```

#### Sample Output:
```text
Running workflow (gemini/gemini-2.5-flash): What are the remote work core hours?

[Supervisor] Diverting to Retriever...
[Retriever] Fetching relevant documents...
[Supervisor] Diverting to Response...
[Response] Generating answer...
[Supervisor] Diverting to Evaluator...
[Evaluator] Evaluating response with RAGAS...
[Supervisor] Workflow complete.

FINAL ANSWER:
The remote‑work core hours are 10:00 AM to 4:00 PM in the employee’s designated primary time zone.

SOURCES:
  - remote_work_policy.md
  - leave_policy.md

EVALUATION (RAGAS):
  - Faithfulness:      100.00%
  - Answer Relevancy:  86.25%
  - Summary:           Strong quality: Response is highly faithful to context and directly relevant to the question.
```

### 3. Run Direct Query CLI
```powershell
uv run python -m src.query "How many days of annual leave do employees receive per year?"
```

### 4. Launch the Streamlit Web Interface
Start the interactive Streamlit application:
```powershell
uv run streamlit run app.py
```
Open **`http://localhost:8501`** (or displayed port) in your browser.

---

## RAG & Indexing Details

- **Raw Knowledge Source**: Stored in `data/raw/` (`leave_policy.md`, `remote_work_policy.md`, `it_security_policy.md`).
- **Supported Formats**: `.md`, `.txt`, `.pdf` parsed via LangChain document loaders.
- **Chunking Strategy**: `RecursiveCharacterTextSplitter` configured with `chunk_size=800` characters and `chunk_overlap=120` characters.
- **Embedding Model**: Fixed `gemini-embedding-001` embeddings.
- **Vector Storage**: Persisted in `data/chroma/` under collection `enterprise_knowledge`.

---

## Model Context Protocol (MCP) Integration (FastMCP)

The project implements a **custom Python MCP Server** powered by **FastMCP** (`src/mcp_server.py`):

### 1. Server Architecture & Transport
- **Framework**: `fastmcp.FastMCP("EnterpriseKnowledgeServer")`
- **Transport**: Standard JSON-RPC over `stdio` (`python -m src.mcp_server`).
- **Sandbox Security**: Path containment validation guarantees read operations are restricted strictly within `data/raw/`.

### 2. Tools Exposed by FastMCP Server
- `read_policy_document(file_path: str) -> str`: Safely reads the full, authoritative text of any verified enterprise policy file.
- `list_policy_documents() -> list[str]`: Lists all supported policy files (`.md`, `.txt`, `.pdf`) currently in the knowledge base.

### 3. Use Case Implementation in Multi-Agent Pipeline
- In the `retriever_node` of the LangGraph workflow, ChromaDB vector search identifies the top-ranked policy file.
- The Retriever calls `read_policy_document` on the FastMCP server over `stdio`.
- This enriches chunk-level embeddings with complete, authoritative document context, eliminating truncation errors and boosting answer quality.

---

## RAGAS Quality Evaluation

The Evaluator Agent integrates automated metrics via the RAGAS framework:

- **Faithfulness**: Quantifies whether all claims made in the answer are grounded in the retrieved context (detects hallucinations).
- **Answer Relevancy**: Quantifies how directly and concisely the answer addresses the user's prompt.
- **Quality Scale**:
  - **$\ge 80.00\%$**: Strong quality (*Response is highly faithful to context and directly relevant to the question*).
  - **$60.00\% - 79.99\%$**: Acceptable quality (*Response is reasonably grounded and relevant, but could improve*).
  - **$< 60.00\%$**: Needs review (*Low faithfulness or relevancy detected*).

---

## Observability & LangSmith Tracing

The multi-agent workflow is instrumented with **LangSmith**:

- **End-to-End Tracing**: Automatically captures multi-agent trajectories showing `supervisor`, `retriever_node` (including FastMCP tool calls), `response_node` (LLM generation), and `evaluator_node` (RAGAS evaluation).
- **Safe Operation**: No private keys or secrets are exposed in telemetry logs or traces.

---

## Screenshots & Visual Evidence

The `docs/screenshots/` folder contains visual documentation proving application functionality:

| Screenshot File | Description & Verification Proof |
| :--- | :--- |
| [`streamlit_app.jpg`](docs/screenshots/streamlit_app.jpg) | Proves Streamlit UI startup and sample question buttons. |
| [`streamlit_app_withoutput.jpg`](docs/screenshots/streamlit_app_withoutput.jpg) | Proves end-to-end question answering, grounded answer display, source expander, and RAGAS metric cards. |
| [`langsmith_observability.jpg`](docs/screenshots/langsmith_observability.jpg) | Proves LangSmith multi-agent execution trace showing Supervisor, Retriever, Response, and Evaluator nodes. |

---

## Folder Structure

```text
app.py                  # Streamlit web user interface
data/
  raw/                  # Raw enterprise knowledge documents (.md, .txt, .pdf)
    leave_policy.md
    remote_work_policy.md
    it_security_policy.md
  chroma/               # Persistent ChromaDB vector storage
docs/
  screenshots/          # UI screenshots and visual documentation
    streamlit_app.jpg
    streamlit_app_withoutput.jpg
    langsmith_observability.jpg
src/
  __init__.py
  answering.py          # Grounded answering, context formatting, and prompt template
  config.py             # Configuration and environment settings
  evaluation.py         # RAGAS evaluation (Faithfulness & Answer Relevancy)
  guardrails.py         # Input validation and LangChain PIIMiddleware
  ingest.py             # Ingestion & index build CLI
  llm_factory.py        # Chat model factory (Gemini, OpenAI, Ollama)
  mcp_server.py         # Custom FastMCP Server exposing enterprise document tools
  mcp_client.py         # FastMCP client connecting to custom server via stdio
  query.py              # Direct query CLI
  run_graph.py          # LangGraph workflow CLI runner
  graph/
    __init__.py
    nodes.py            # Supervisor, Retriever, Response, and Evaluator nodes
    state.py            # GraphState TypedDict definition
    workflow.py         # StateGraph builder and workflow execution
  rag/
    __init__.py
    loaders.py          # PDF, TXT, MD document loaders
    indexer.py          # Chunking, Gemini embeddings, and ChromaDB indexing
    retriever.py        # Vector similarity search using fixed Gemini embeddings
.env.example            # Sample environment variables template
.gitignore              # Git ignore rules
pyproject.toml          # Project configuration and dependencies
requirements.txt        # Clean top-level dependency manifest
README.md               # Project documentation
```

---

## Submission Checklist

- [x] **Source Code**: Complete multi-agent implementation across `src/` and `app.py`.
- [x] **FastMCP Custom Server**: Python-native FastMCP server (`EnterpriseKnowledgeServer`) with `read_policy_document` and `list_policy_documents` tools.
- [x] **Supervisor Agent**: Dynamic hub-and-spoke state routing to specialized worker nodes.
- [x] **Privacy & Guardrails**: Input validation, length checks, and LangChain `PIIMiddleware`.
- [x] **Documentation**: Full `README.md` with architecture, setup, RAG, FastMCP, RAGAS, and execution guides.
- [x] **Visual Evidence**: Screenshots in `docs/screenshots/` verifying UI startup, results, and LangSmith traces.
- [x] **Secrets Excluded**: `.env`, `.venv/`, `data/chroma/`, and temporary artifacts properly gitignored.
