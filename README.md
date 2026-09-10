# Enterprise Knowledge Assistant

Enterprise Knowledge Assistant is a multi-agent Retrieval-Augmented Generation (RAG) system built to search, synthesize, and evaluate enterprise policy documents. It addresses the common challenges of enterprise search—hallucinations, untracked context sources, and unverifiable answers—by orchestrating a linear multi-agent workflow in LangGraph that incorporates authoritative file reading via the official Filesystem MCP server, automated answer evaluation via RAGAS, observability via LangSmith, input guardrails with built-in LangChain `PIIMiddleware`, and post-workflow output guardrails.

---

## Architecture Overview

The system uses a clean linear multi-agent pipeline in LangGraph:

```text
START ──► [Retriever Agent] ──► [Response Agent] ──► [Evaluator Agent] ──► END
          (ChromaDB + MCP)      (Grounded LLM)       (RAGAS Eval)
                                                           │
                                                           ▼
                                                  [Output Guardrails]
                                                           │
                                                           ▼
                                                     Final Result
```

### Execution Pipeline Flow

1. **Input Guardrails & PII Middleware**: Validates query length, strips whitespace, and processes sensitive personal information (Emails, Phone numbers, URLs, IPs, SSNs) via LangChain's built-in `PIIMiddleware`.
2. **Retriever Node**: Queries ChromaDB vector storage using fixed Gemini embeddings, identifies the top-ranked source file, and invokes the official Filesystem MCP server (`read_text_file`) to retrieve authoritative source context.
3. **Response Node**: Synthesizes a factual, strictly grounded answer based exclusively on the assembled context using the configured Chat LLM.
4. **Evaluator Node**: Assesses the generated answer against the retrieved context and user question using RAGAS (*Faithfulness* and *Answer Relevancy*).
5. **Output Guardrails**: Post-graph application middleware verifying that answers are grounded with supporting documents, meet RAGAS quality thresholds, filter secret credentials, and protect against prompt injection or internal prompt disclosure.

---

## Guardrails and Privacy Middleware

### Input Guardrails (`src/guardrails.py`)
The application validates and sanitizes incoming user queries before graph execution:
- **Input Validation**: Rejects empty queries and restricts question length to a maximum of 500 characters.
- **Built-in PII Strategies**: Powered by LangChain's built-in `PIIMiddleware`:
  - **`redact`**: Replaces sensitive data with category placeholders (e.g. `[REDACTED_EMAIL]`, `[REDACTED_URL]`).
  - **`mask`**: Obscures sensitive digits or tokens (e.g. `****-****-****-0002` for credit cards, `user@****.com` for emails, masked SSNs).
  - **`hash`**: Securely hashes sensitive identifiers (e.g. IP addresses).
  - **`block`**: Raises a `ValueError` / `PIIDetectionError` if prohibited personal information is submitted.
- **Sandboxed MCP Tooling**: The FastMCP client is strictly read-only (`read_policy_document`), confined exclusively to `data/raw/`, and validates file path containment prior to execution.

### Output Guardrails (`src/output_guardrails.py` & `src/workflow_middleware.py`)
Post-workflow safety validation occurs after the Evaluator Agent assesses the answer (`Retriever Agent -> Response Agent -> Evaluator Agent -> Output Guardrails -> Final Result`):
- **Supporting Sources Requirement**: Non-empty answers require verified, supporting source documents. Answers without supporting sources are blocked.
- **RAGAS Quality Thresholds**: Enforces minimum quality standards:
  - **Faithfulness**: Must be $\ge 0.70$ (detects hallucinations and unsupported claims).
  - **Answer Relevancy**: Must be $\ge 0.60$ (ensures concise alignment with user query).
  - Missing, invalid, or sub-threshold scores immediately halt output delivery.
- **Secret-Pattern Filtering**: Detects and blocks accidental disclosure of sensitive credentials:
  - Google AI / Cloud API keys (`AIza...`)
  - OpenAI API keys (`sk-...`)
  - LangSmith tokens (`lsv2_pt_...`)
  - GitHub tokens (`ghp_...` and `github_pat_...`)
- **Internal-Prompt & Prompt-Injection Filtering**: Blocks responses containing system prompt reveals or prompt-injection instructions (e.g., "ignore previous instructions", "reveal system prompt", "show hidden instructions"). Normal enterprise policies concerning passwords, credentials, and API security guidelines remain unblocked.
- **Safe User Messaging**: Unsafe or ungrounded answers are hidden and replaced with a friendly, safe notice (`"Unable to show a verified answer for this question. Please try a more specific policy question."`) without exposing raw model outputs, stack traces, or internal prompt details.

---

## Contact Redirection

The system includes deterministic contact-based redirection for unsupported queries, ensuring employees always receive verified guidance without LLM hallucination:

- **Relevance-Aware Routing**: Documents retrieved from ChromaDB are evaluated against `MIN_RETRIEVAL_SCORE` (default `0.45`). When no retrieved documents meet this threshold, the system flags the inquiry as lacking verified policy context.
- **Deterministic Topic Classification**: A lightweight, keyword-based classifier in `src/contact_directory.py` assigns inquiries to the appropriate department:
  - **IT & Security** (`it_security`): Matches terms such as *password*, *MFA*, *security*, *device*, *laptop*, *access*, *encryption*, *VPN*.
  - **People Operations** (`people_operations`): Matches terms such as *leave*, *PTO*, *sick leave*, *remote work*, *work from home*, *holiday*.
  - **General Support** (`general_support`): Handles all other unmapped inquiries.
- **Safe Redirection Message**: When no verified context exists, the Response Agent bypasses LLM generation and produces a standardized redirection message pointing the user to the designated team contact.
- **Evaluation Bypass**: RAGAS evaluation is safely skipped for redirected inquiries since there is no retrieved ground-truth context to assess, setting the evaluation status to *"Not evaluated because no verified policy context was found."*
- **Contact Footer on Supported Answers**: For normal verified answers, a standardized contact footer (`For more information, contact <Name>, <Position>, at <Email>.`) is displayed after RAGAS evaluation and output guardrail validation.
- **Configurable Directory**: Contact information is maintained in `data/contact_directory.json` with safe placeholder contacts. Prior to enterprise deployment, these entries should be updated with authorized organizational contacts.

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
| `GITHUB_MCP_URL` | GitHub remote read-only MCP server endpoint | `https://api.githubcopilot.com/mcp/readonly` |
| `GITHUB_PAT` | GitHub Personal Access Token with read-only repository permissions | `ghp_...` |
| `GITHUB_REPOSITORY_OWNER` | GitHub repository owner/organization | `PranavPatil-106` |
| `GITHUB_REPOSITORY_NAME` | GitHub repository name | `enterprise-knowledge-assistant` |

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

## External GitHub MCP Integration (Streamable HTTP)

The project incorporates GitHub’s official remote read-only Model Context Protocol (MCP) server (`https://api.githubcopilot.com/mcp/readonly`) to answer live repository inquiries without modifying the core three-node LangGraph pipeline.

### 1. Connection & Transport
- **Client Implementation**: [`src/github_mcp_client.py`](file:///e:/L3%20Project/AI%20Enterprise%20Project/src/github_mcp_client.py)
- **Protocol & Transport**: Official `mcp` SDK using `streamable_http_client` and `ClientSession` over Streamable HTTP (`POST` with event stream processing).
- **Authentication**: `Authorization: Bearer <GITHUB_PAT>` configured via local `.env`.
- **Target Repository**: Configured via `GITHUB_REPOSITORY_OWNER` and `GITHUB_REPOSITORY_NAME` (e.g. `PranavPatil-106/enterprise-knowledge-assistant`).

### 2. Read-Only Protection & Allowed Tools
- **Read-Only Server Guarantee**: The connection explicitly specifies `X-MCP-Readonly: true` on every HTTP transport handshake.
- **Allowed Tool Header**: Constrained strictly to `X-MCP-Tools: list_commits,get_file_contents`.
- **Client-Side Whitelist Enforcement**: `ALLOWED_GITHUB_MCP_TOOLS = {"list_commits", "get_file_contents"}` is enforced programmatically in Python before opening any network transport. Any attempt to invoke write tools (such as creating issues, pushing commits, opening PRs, or deleting branches) raises a `ValueError` immediately.
- **Credential Protection**: Personal Access Tokens (`GITHUB_PAT`) are never logged, printed, or exposed in telemetry traces or error messages.

### 3. Pipeline Integration (Linear Flow Preserved)
- **Zero New Nodes**: Integration lives entirely inside the existing `Retriever Agent` (`retriever_node`). The LangGraph flow remains strictly linear:
  ```text
  START -> Retriever Agent -> Response Agent -> Evaluator Agent -> END
  ```
- **Deterministic Routing**: Questions referencing repositories, commits, project files, or readmes are routed to GitHub MCP without LLM overhead.
- **Context Synthesis**: Live commit metadata (commit SHA, author, date, message) or file contents (`README.md`) are converted into a verified `Document` and passed to the Response Agent.
- **Evaluator & UI Display**: Responses are evaluated via RAGAS and flagged in Streamlit with an `External source: GitHub MCP (read-only)` indicator.

### 4. Human-in-the-Loop (HITL) Note for Future Write Operations
While the current integration is strictly read-only, if future write tools (such as opening pull requests or creating issues) were ever added, they **must** require human confirmation (Human-in-the-Loop) through interactive approval prompts and fine-grained, dedicated write tokens rather than autonomous execution.

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
  contact_directory.json # Departmental contact directory for query redirection
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
  application.py        # Application service layer coordinating workflow and guardrails
  config.py             # Configuration and environment settings
  contact_directory.py  # Contact loading, keyword topic classification, and message formatters
  evaluation.py         # RAGAS evaluation (Faithfulness & Answer Relevancy)
  guardrails.py         # Input validation and LangChain PIIMiddleware
  ingest.py             # Ingestion & index build CLI
  llm_factory.py        # Chat model factory (Gemini, OpenAI, Ollama)
  mcp_server.py         # Custom FastMCP Server exposing enterprise document tools
  mcp_client.py         # FastMCP client connecting to custom server via stdio
  github_mcp_client.py  # GitHub remote read-only MCP client over Streamable HTTP
  output_guardrails.py  # Post-workflow output validation and secret/injection filtering
  query.py              # Direct query CLI
  run_graph.py          # LangGraph workflow CLI runner
  workflow_middleware.py # Safety middleware hooks before and after graph execution
  graph/
    __init__.py
    nodes.py            # Retriever, Response, and Evaluator nodes
    state.py            # GraphState TypedDict definition
    workflow.py         # StateGraph builder and workflow execution
  rag/
    __init__.py
    loaders.py          # PDF, TXT, MD document loaders
    indexer.py          # Chunking, Gemini embeddings, and ChromaDB indexing
    retriever.py        # Vector similarity search using fixed Gemini embeddings
tests/                  # Comprehensive unit and integration test suite
  test_chunking.py
  test_contact_directory.py
  test_evaluation.py
  test_github_mcp_client.py
  test_graph.py
  test_guardrails.py
  test_mcp_client.py
  test_output_guardrails.py
  test_query.py
.env.example            # Sample environment variables template
.gitignore              # Git ignore rules
pyproject.toml          # Project configuration and dependencies
requirements.txt        # Clean top-level dependency manifest
README.md               # Project documentation
```

---

## Submission Checklist

- [x] **Source Code**: Complete multi-agent implementation across `src/` and `app.py`.
- [x] **Filesystem MCP Integration**: Official Filesystem MCP client (`read_text_file`) retrieving authoritative document content.
- [x] **External GitHub MCP Integration**: GitHub Copilot remote read-only MCP server over Streamable HTTP with strict read-only header and whitelist enforcement.
- [x] **Linear Agent Pipeline**: Strict linear workflow (`START -> Retriever -> Response -> Evaluator -> END`).
- [x] **Contact Redirection**: Relevance-aware routing, topic classification, safe redirection, and contact footers.
- [x] **Privacy & Guardrails**: Input validation (`PIIMiddleware`) and output guardrails with RAGAS thresholding and credential filtering.
- [x] **Documentation**: Full `README.md` with architecture, setup, RAG, MCP, RAGAS, and execution guides.
- [x] **Visual Evidence**: Screenshots in `docs/screenshots/` verifying UI startup, results, and LangSmith traces.
- [x] **Secrets Excluded**: `.env`, `.venv/`, `data/chroma/`, and temporary artifacts properly gitignored.
