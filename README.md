# Enterprise Knowledge Assistant

Enterprise Knowledge Assistant is an agentic, multi-agent Retrieval-Augmented Generation (RAG) system built to accurately search, synthesize, and evaluate enterprise policy documents. It addresses the common challenges of enterprise search—hallucinations, untracked context sources, and unverifiable answers—by orchestrating a multi-agent workflow that incorporates authoritative file reading via the Model Context Protocol (MCP), automated answer evaluation via RAGAS, full observability via LangSmith, application-level safety guardrails, and an interactive Streamlit user interface.

---

## Architecture Overview

The system processes queries through an orchestrated multi-agent linear workflow wrapped with safety middleware:

```text
User
  -> Streamlit UI / CLI Runner
  -> Workflow Middleware (Pre-execution Guardrails)
  -> LangGraph Workflow
       -> Retriever Agent (ChromaDB + Filesystem MCP)
       -> Response Agent (Grounded Synthesis)
       -> Evaluator Agent (RAGAS Quality Metrics)
  -> Workflow Middleware (Post-execution Validation)
  -> Final Answer + Source Documents + RAGAS Scores
```

### Execution Pipeline Flow

```text
Retriever Agent -> Response Agent -> Evaluator Agent -> End
```

1. **Pre-Workflow Guardrails**: Validates query length, strips whitespace, and blocks prompt-injection or secret-exfiltration attempts.
2. **Retriever Agent**: Queries ChromaDB vector storage using fixed Gemini embeddings, identifies the top-ranked source file, and invokes the official Filesystem MCP server (`read_text_file`) to retrieve complete, authoritative source context.
3. **Response Agent**: Synthesizes a factual, strictly grounded answer based exclusively on the assembled context using the configured Chat LLM.
4. **Evaluator Agent**: Assesses the generated answer against the retrieved context and user question using RAGAS (*Faithfulness* and *Answer Relevancy*).
5. **Post-Workflow Middleware**: Validates that a non-empty answer was produced and at least one authoritative source document was cited before results are returned.

---

## Guardrails and Safety

The application implements a dedicated safety and validation layer (`src/guardrails.py` and `src/workflow_middleware.py`) to protect system integrity and ensure high answer quality:

- **Input Validation**: Rejects empty queries and restricts question length to a maximum of 500 characters.
- **Prompt-Injection Defense**: Pattern-matching filters block instruction-override and prompt-manipulation attempts (e.g., *"ignore previous instructions"*, *"reveal system prompt"*, *"jailbreak"*).
- **Secret & Credential Protection**: Proactively intercepts and blocks requests attempting to access API keys, `.env` files, credentials, private tokens, or environment variables.
- **Grounded Answer & Source Validation**: Post-execution middleware strictly requires that the response agent generated an answer and that at least one authoritative source document grounded the synthesis.
- **Sandboxed MCP Tooling**: The Filesystem MCP client is strictly read-only (`read_text_file`), confined exclusively to `data/raw/`, and validates file path containment prior to execution.

### Human-in-the-Loop (HITL) Note

> **Note on Human-in-the-Loop**:
> - Human-in-the-loop confirmation is intentionally not required for this workflow because the system operates strictly in a read-only capacity, performs no state modifications, and makes no irreversible external changes.
> - Any future extensions introducing write, delete, file-mutation, or external API execution tools must require explicit human-in-the-loop authorization before execution.

---

## Technology Stack

- **Python**: Core programming language.
- **uv**: Primary package and virtual environment manager.
- **LangGraph**: Multi-agent workflow orchestration and state graph execution.
- **LangChain**: Document abstractions, vector store integrations, and LLM interfaces.
- **Chat LLMs**: Configurable chat models supporting **Gemini** (default), **OpenAI**, and **Ollama**.
- **Embedding Model**: Fixed, non-configurable **Gemini Embeddings** (`gemini-embedding-001`).
- **ChromaDB**: Persistent vector database for embedding storage and similarity retrieval.
- **RAGAS**: Automated evaluation framework for Faithfulness and Answer Relevancy.
- **Filesystem MCP**: Official `@modelcontextprotocol/server-filesystem` server for safe, read-only file reading over `stdio`.
- **Streamlit**: Web application frontend for interactive querying and metric visualization.
- **LangSmith**: Telemetry, execution tracing, and agent debugging.
- **Git / GitHub**: Source control and project lifecycle management.

---

## Dependency Management & `requirements.txt`

- **Primary Tool**: `uv` is the primary dependency and environment manager for this project.
- **Submission Reference**: `requirements.txt` is an exported dependency manifest generated directly from the locked `uv` environment for submission and reference purposes.
- **Regenerating `requirements.txt`**: If dependencies change, regenerate `requirements.txt` with:
  ```powershell
  uv export --format requirements-txt --no-hashes --output-file requirements.txt
  ```

---

## Setup Instructions

### 1. Clone or Open the Repository
```powershell
git clone <repo-url>
cd "AI Enterprise Project"
```

### 2. Install Dependencies
Install all required packages and initialize the virtual environment using `uv`:
```powershell
uv sync
```

### 3. Configure Environment Variables
Copy the sample environment template `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Edit `.env` to configure your credentials and preferences:

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | Chat model provider (`gemini`, `openai`, `ollama`) | `gemini` |
| `LLM_MODEL` | Chat model identifier | `gemini-3.6-flash` |
| `GOOGLE_API_KEY` | Google AI Studio API key (required for embeddings and Gemini LLM) | `AIzaSy...` |
| `OPENAI_API_KEY` | OpenAI API key (required only if `LLM_PROVIDER=openai`) | `sk-...` |
| `OLLAMA_BASE_URL` | Local Ollama endpoint URL (required only if `LLM_PROVIDER=ollama`) | `http://localhost:11434` |
| `KNOWLEDGE_BASE_PATH` | Directory containing enterprise policy documents | `data/raw` |
| `CHROMA_PERSIST_DIRECTORY`| Directory for ChromaDB vector storage | `data/chroma` |
| `MCP_ALLOWED_DIRECTORY` | Directory allowed for Filesystem MCP access | `data/raw` |
| `LANGSMITH_TRACING` | Enable or disable LangSmith tracing (`true`/`false`) | `true` |
| `LANGSMITH_API_KEY` | LangSmith API key for trace ingestion | `lsv2_pt_...` |
| `LANGSMITH_PROJECT` | LangSmith project name for traces | `enterprise-knowledge-assistant` |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint URL | `https://api.smith.langchain.com` |

> **Important Notes on Configuration**:
> - **Fixed Embeddings**: Embeddings **always** use `gemini-embedding-001` via `GOOGLE_API_KEY`. The chat LLM is dynamic and configurable.
> - **Prerequisites for Filesystem MCP**: Node.js and `npx` are required on the host machine to run the official `@modelcontextprotocol/server-filesystem` server.

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
uv run python -m src.run_graph --question "What are the remote work core hours?"
```

#### Sample Output:
```text
============================================================
FINAL ANSWER:
------------------------------------------------------------
Based on the provided documents, the core working hours for remote (and hybrid) employees are 10:00 AM to 4:00 PM in the employee's designated primary time zone. During these hours, employees must be reachable and available for synchronous communication.

------------------------------------------------------------
SOURCES:
  - remote_work_policy.md
  - leave_policy.md

------------------------------------------------------------
EVALUATION (RAGAS):
  - Faithfulness:      100.00%
  - Answer Relevancy:  83.52%
  - Summary:           Strong quality: Response is highly faithful to context and directly relevant to the question.
============================================================
```

### 3. Launch the Streamlit Web Interface
Start the interactive Streamlit application:
```powershell
uv run streamlit run app.py
```
The browser opens automatically at `http://localhost:8501`.

---

## RAG Design & Indexing

- **Raw Knowledge Source**: Stored in `data/raw/` (`leave_policy.md`, `remote_work_policy.md`, `it_security_policy.md`).
- **Supported Formats**: `.md`, `.txt`, `.pdf` parsed via LangChain document loaders.
- **Chunking Strategy**: `RecursiveCharacterTextSplitter` configured with `chunk_size=800` characters and `chunk_overlap=120` characters to maintain coherent sentence boundaries.
- **Embedding Model**: Fixed `gemini-embedding-001` embeddings configured in `Settings.embedding_model`.
- **Vector Storage**: Persisted locally in `data/chroma/` under collection `enterprise_knowledge`.
- **Clean Ingestion**: Re-running ingestion drops and rebuilds the `enterprise_knowledge` collection cleanly.

---

## LangGraph Multi-Agent Workflow

The stateful pipeline connects three specialized agent nodes:

```text
START -> Retriever Agent -> Response Agent -> Evaluator Agent -> END
```

1. **Retriever Agent (`retriever_node`)**:
   - Queries ChromaDB using fixed Gemini embeddings.
   - Validates file paths and queries the official Filesystem MCP server via `read_text_file` to fetch the authoritative complete source document.
   - Populates `documents`, `context`, `mcp_context`, and `sources` in `GraphState`.

2. **Response Agent (`response_node`)**:
   - Accepts the query and enriched context.
   - Prompts the configured Chat LLM to produce a factual, grounded answer with strict anti-hallucination guardrails.
   - Sets `answer` in `GraphState`.

3. **Evaluator Agent (`evaluator_node`)**:
   - Passes the question, synthesized answer, and retrieved contexts to RAGAS.
   - Calculates **Faithfulness** and **Answer Relevancy**.
   - Generates a human-readable quality summary and sets `evaluation` in `GraphState`.

---

## Model Context Protocol (MCP) Integration

The application actively uses the official **Filesystem MCP Server** during graph execution:

- **Official Server**: `@modelcontextprotocol/server-filesystem` executed over `stdio` via `npx`.
- **Tool Invoked**: `read_text_file`.
- **Restricted Sandbox**: Access is strictly constrained to `data/raw` via `MCP_ALLOWED_DIRECTORY`.
- **Retriever Agent Use Case**: Enriches top-ranked chunk retrieval by fetching complete, authoritative policy context directly through standard MCP tooling.
- **Read-Only Safety**: The Python client validates path containment within the sandbox and only executes non-destructive read calls.

---

## RAGAS Quality Evaluation

The Evaluator Agent integrates automated metrics via the RAGAS framework:

- **Faithfulness**: Quantifies whether all claims made in the answer are grounded in the retrieved context (detects hallucinations).
- **Answer Relevancy**: Quantifies how directly and concisely the answer addresses the user's prompt (penalizes evasive or redundant responses).
- **Score Scale**: Ranging from `0.00%` to `100.00%` (higher is better).
- **Quality Thresholds**:
  - **$\ge 80.00\%$**: Strong quality (*Response is highly faithful to context and directly relevant to the question*).
  - **$60.00\% - 79.99\%$**: Acceptable quality (*Response is reasonably grounded and relevant, but could improve*).
  - **$< 60.00\%$**: Needs review (*Low faithfulness or relevancy detected*).
- **Verified Benchmark Result**:
  - **Faithfulness**: `100.00%`
  - **Answer Relevancy**: `~83.00% - 87.00%`
  *(Note: Live scores may vary slightly as evaluation is LLM-based).*

---

## Observability & LangSmith Tracing

The entire multi-agent workflow is instrumented with **LangSmith**:

- **End-to-End Tracing**: Automatically captures full multi-agent trajectories showing `retriever_node` (including Filesystem MCP tool calls), `response_node` (LLM generation), and `evaluator_node` (RAGAS evaluation).
- **Telemetry Flushing**: Traces are flushed deterministically on shutdown via `flush_langsmith()`.
- **Safe Operation**: No secrets or private keys are exposed in telemetry logs or traces.

---

## Screenshots & Visual Evidence

The `docs/screenshots/` folder contains visual documentation proving application functionality:

| Screenshot File | Description & Verification Proof |
| :--- | :--- |
| [`streamlit_startup.png`](docs/screenshots/streamlit_startup.png) | Proves Streamlit UI startup, system configuration sidebar, and sample question buttons. |
| [`streamlit_app.png`](docs/screenshots/streamlit_app.png) | Proves end-to-end question answering, grounded answer display, source expander, and RAGAS metric cards. |
| [`langsmith_trace.png`](docs/screenshots/langsmith_trace.png) | Proves LangSmith multi-agent execution trace showing Retriever, Response, and Evaluator nodes in order. |

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
    streamlit_startup.png
    streamlit_app.png
    langsmith_trace.png
logs/                   # Application logs
src/
  __init__.py
  answering.py          # Grounded answering, context formatting, and prompt template
  application.py        # Shared application service layer for workflow execution
  config.py             # Configuration and environment settings
  evaluation.py         # RAGAS evaluation (Faithfulness & Answer Relevancy)
  guardrails.py         # Input validation, injection prevention, and secret protection
  ingest.py             # Ingestion & index build CLI
  llm_factory.py        # Chat model factory (Gemini, OpenAI, Ollama)
  mcp_client.py         # Official Filesystem MCP client and path validator
  observability.py      # LangSmith tracing setup and flushing
  query.py              # Direct query CLI
  run_graph.py          # LangGraph workflow CLI runner
  workflow_middleware.py# Pre- and post-execution application middleware
  graph/
    __init__.py
    nodes.py            # Retriever, Response, and Evaluator agent nodes
    state.py            # GraphState TypedDict definition
    workflow.py         # StateGraph builder and compilation
    rag/
      __init__.py
      loaders.py        # PDF, TXT, MD document loaders
      indexer.py        # Chunking, Gemini embeddings, and ChromaDB indexing
      retriever.py      # Vector similarity search using fixed Gemini embeddings
tests/                  # Automated test suite
  test_chunking.py      # In-memory chunking unit tests
  test_evaluation.py    # RAGAS evaluation interpretation & reporting unit tests
  test_graph.py         # Graph structure and node delegation unit tests
  test_guardrails.py    # Safety guardrails and workflow middleware unit tests
  test_mcp_client.py    # MCP path validation and text extraction unit tests
  test_observability.py # LangSmith configuration and environment unit tests
  test_query.py         # Context formatting and source extraction unit tests
.env.example            # Sample environment variables template
.gitignore              # Git ignore rules
pyproject.toml          # Project configuration and dependencies
requirements.txt        # Exported dependency list for submission/reference
README.md               # Project documentation
```

---

## Submission Checklist

- [x] **Source Code**: Complete multi-agent implementation across `src/`, `app.py`, and `tests/`.
- [x] **Safety Guardrails**: Input validation, prompt injection blocking, secret exfiltration defense, and source verification.
- [x] **Documentation**: Full `README.md` with architecture, setup, RAG, MCP, RAGAS, and execution guides.
- [x] **Visual Evidence**: Screenshots in `docs/screenshots/` verifying UI startup, results, and LangSmith traces.
- [x] **Secrets & Local Artifacts Excluded**: `.env`, `.venv/`, `data/chroma/`, and `logs/` properly gitignored.
- [x] **Package Manifests**: `pyproject.toml`, `uv.lock`, and exported `requirements.txt` included.
