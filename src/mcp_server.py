from pathlib import Path
from fastmcp import FastMCP
from src.config import get_settings

mcp = FastMCP("EnterpriseKnowledgeServer")


@mcp.tool()
def read_policy_document(file_path: str) -> str:
    settings = get_settings()
    allowed_dir = settings.mcp_allowed_directory.resolve()
    target_path = Path(file_path).resolve()

    try:
        target_path.relative_to(allowed_dir)
    except ValueError:
        raise ValueError(f"Access denied: '{target_path}' is outside allowed directory '{allowed_dir}'.")

    if not target_path.exists():
        raise FileNotFoundError(f"Policy document '{target_path}' does not exist.")

    return target_path.read_text(encoding="utf-8")


@mcp.tool()
def list_policy_documents() -> list[str]:
    settings = get_settings()
    allowed_dir = settings.mcp_allowed_directory.resolve()
    if not allowed_dir.exists():
        return []
    return sorted(
        [f.name for f in allowed_dir.iterdir() if f.is_file() and f.suffix.lower() in (".md", ".txt", ".pdf")]
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
