"""Client for the official GitHub Remote Model Context Protocol (MCP) server over Streamable HTTP."""

import asyncio
import concurrent.futures
import json
import re
from typing import Any

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.config import Settings, get_settings
from src.mcp_client import extract_text_from_mcp_result

# Strictly enforce read-only tool access; write tools are prohibited
ALLOWED_GITHUB_MCP_TOOLS: set[str] = {"list_commits", "get_file_contents"}

# Regex pattern for deterministic GitHub inquiry detection
GITHUB_QUESTION_PATTERN = re.compile(
    r"\b(?:github|repository|repo|commits?|readme|project\s+files?|source\s+code)\b",
    re.IGNORECASE,
)


def is_github_question(question: str) -> bool:
    """
    Determine whether a question pertains to the GitHub repository, commits, or source files.

    Deterministic keyword matching without calling any LLM.
    """
    if not question or not isinstance(question, str):
        return False
    return bool(GITHUB_QUESTION_PATTERN.search(question))


def get_github_mcp_headers(settings: Settings | None = None) -> dict[str, str]:
    """
    Prepare headers for connecting to GitHub's official remote read-only MCP server.

    Includes required authentication, read-only constraint, and tool limitation headers.
    Raises ValueError if GITHUB_PAT is not configured.
    """
    cfg = settings or get_settings()
    pat = cfg.github_pat.strip()
    if not pat:
        raise ValueError(
            "GitHub Personal Access Token (GITHUB_PAT) is missing. "
            "Please configure GITHUB_PAT in your local .env file."
        )

    return {
        "Authorization": f"Bearer {pat}",
        "X-MCP-Readonly": "true",
        "X-MCP-Tools": "list_commits,get_file_contents",
    }


def format_commit_info(raw_text: str, owner: str, repo: str) -> str:
    """Extract and format essential commit details (message, short SHA, author, date, URL)."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, list) and data:
            latest = data[0]
        elif isinstance(data, dict):
            latest = data.get("commits", [data])[0] if isinstance(data.get("commits"), list) else data
        else:
            return raw_text

        sha = latest.get("sha", "")
        short_sha = sha[:7] if sha else "latest"
        commit_obj = latest.get("commit", {})
        msg = commit_obj.get("message", latest.get("message", "No commit message provided")).strip()
        author_info = commit_obj.get("author", latest.get("author", {}))
        author_name = (
            author_info.get("name", "Unknown author")
            if isinstance(author_info, dict)
            else str(author_info)
        )
        date = author_info.get("date", "") if isinstance(author_info, dict) else ""
        html_url = latest.get("html_url") or f"https://github.com/{owner}/{repo}/commit/{short_sha}"

        parts = [
            f"Repository: {owner}/{repo}",
            f"Latest Commit: {short_sha}",
            f"Commit Message: {msg}",
            f"Author: {author_name}",
        ]
        if date:
            parts.append(f"Date: {date}")
        if html_url:
            parts.append(f"Commit URL: {html_url}")

        return "\n".join(parts)
    except Exception:
        return raw_text.strip()


def format_file_content(raw_text: str, file_path: str, owner: str, repo: str) -> str:
    """Format file content returned by GitHub MCP."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            content = data.get("content") or data.get("text") or raw_text
            return f"=== GitHub Repository File ({owner}/{repo}:{file_path}) ===\n{content}"
    except Exception:
        pass
    return f"=== GitHub Repository File ({owner}/{repo}:{file_path}) ===\n{raw_text.strip()}"


async def _async_call_github_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    settings: Settings | None = None,
) -> str:
    """Connect to GitHub's remote MCP server over Streamable HTTP and call a read-only tool."""
    cfg = settings or get_settings()

    if tool_name not in ALLOWED_GITHUB_MCP_TOOLS:
        raise ValueError(
            f"Unauthorized GitHub MCP tool: '{tool_name}' is not permitted. "
            f"Only read-only tools {sorted(ALLOWED_GITHUB_MCP_TOOLS)} are allowed."
        )

    headers = get_github_mcp_headers(cfg)
    url = cfg.github_mcp_url.strip()

    timeout = httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=15.0)
    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
            async with streamable_http_client(url=url, http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)
                    return extract_text_from_mcp_result(result)
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if status_code == 401:
            raise RuntimeError(
                "GitHub MCP authentication failed (401 Unauthorized). "
                "Please verify that your GITHUB_PAT is valid and has read-only repository permissions."
            ) from None
        elif status_code == 403:
            raise RuntimeError(
                "GitHub MCP access forbidden (403 Forbidden). "
                "Your GITHUB_PAT may lack permissions for this repository or rate limits were exceeded."
            ) from None
        else:
            raise RuntimeError(
                f"GitHub remote MCP server returned HTTP status {status_code}."
            ) from None
    except httpx.RequestError:
        raise RuntimeError(
            f"Failed to connect to GitHub remote MCP server at '{url}'. "
            "Please check your internet connection and verify GITHUB_MCP_URL."
        ) from None
    except Exception as e:
        err_msg = str(e)
        if "PAT" in err_msg or "token" in err_msg.lower():
            raise RuntimeError("GitHub MCP authentication error: invalid credentials.") from None
        raise RuntimeError(f"GitHub MCP execution error: {e}") from None


def call_github_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    settings: Settings | None = None,
) -> str:
    """
    Synchronous entry point to execute a read-only GitHub MCP tool over Streamable HTTP.
    Safely dispatches across new or existing event loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run, _async_call_github_mcp_tool(tool_name, arguments, settings)
            ).result()

    return asyncio.run(_async_call_github_mcp_tool(tool_name, arguments, settings))


def get_latest_commit_from_github(settings: Settings | None = None) -> str:
    """Retrieve the latest commit from the configured GitHub repository via GitHub MCP list_commits."""
    cfg = settings or get_settings()
    owner = cfg.github_repository_owner
    repo = cfg.github_repository_name
    raw_result = call_github_mcp_tool(
        tool_name="list_commits",
        arguments={"owner": owner, "repo": repo, "perPage": 1},
        settings=cfg,
    )
    return format_commit_info(raw_result, owner, repo)


def get_readme_from_github(settings: Settings | None = None) -> str:
    """Retrieve the README.md content from the configured GitHub repository via GitHub MCP get_file_contents."""
    cfg = settings or get_settings()
    owner = cfg.github_repository_owner
    repo = cfg.github_repository_name
    raw_result = call_github_mcp_tool(
        tool_name="get_file_contents",
        arguments={"owner": owner, "repo": repo, "path": "README.md"},
        settings=cfg,
    )
    return format_file_content(raw_result, "README.md", owner, repo)


def fetch_github_context_for_question(
    question: str,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """
    Fetch repository context for a GitHub-related question.

    Returns:
        tuple[str, str]: (context_text, source_label)
    """
    cfg = settings or get_settings()
    owner = cfg.github_repository_owner
    repo = cfg.github_repository_name
    source_label = f"GitHub MCP: {owner}/{repo}"

    q_lower = question.lower()
    if "readme" in q_lower or "project files" in q_lower:
        context = get_readme_from_github(cfg)
    else:
        # Default for repository/commit/source code queries: fetch latest commit
        context = get_latest_commit_from_github(cfg)

    return context, source_label
