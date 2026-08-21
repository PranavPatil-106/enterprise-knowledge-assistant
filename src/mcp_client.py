import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def validate_file_path(file_path: Path | str, allowed_directory: Path | str) -> Path:
    allowed_dir = Path(allowed_directory).resolve()
    target_file = Path(file_path).resolve()

    try:
        target_file.relative_to(allowed_dir)
    except ValueError:
        raise ValueError(f"Access denied: '{target_file}' is outside '{allowed_dir}'.")

    if not target_file.exists():
        raise FileNotFoundError(f"File not found: '{target_file}'")

    return target_file


def extract_text_from_mcp_result(result: Any) -> str:
    if not result:
        raise ValueError("Invalid MCP response: Result is None.")

    if getattr(result, "is_error", False):
        raise RuntimeError(f"MCP tool reported an error: {result}")

    content = getattr(result, "content", None)
    if not content:
        raise ValueError("MCP tool returned empty content.")

    parts = [getattr(item, "text", str(item.get("text", item) if isinstance(item, dict) else item)) for item in content]
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("MCP tool returned empty text content.")

    return text


def build_server_parameters(allowed_directory: Path) -> StdioServerParameters:
    env = dict(os.environ)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        env=env,
    )


async def _async_read_text_file(validated_file: Path, allowed_directory: Path) -> str:
    server_params = build_server_parameters(allowed_directory)

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "read_policy_document",
                    arguments={"file_path": str(validated_file)},
                )
                return extract_text_from_mcp_result(result)
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError, FileNotFoundError)):
            raise
        raise RuntimeError(f"FastMCP server session error: {e}") from e


def read_text_file_via_mcp(file_path: Path | str, allowed_directory: Path | str) -> str:
    validated_file = validate_file_path(file_path, allowed_directory)
    allowed_dir = Path(allowed_directory).resolve()
    return asyncio.run(_async_read_text_file(validated_file, allowed_dir))


async def _async_list_policy_documents(allowed_directory: Path) -> list[str]:
    server_params = build_server_parameters(allowed_directory)

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("list_policy_documents", arguments={})
                text = extract_text_from_mcp_result(result)
                import json
                try:
                    return json.loads(text)
                except Exception:
                    return [line.strip("- ") for line in text.splitlines() if line.strip()]
    except Exception as e:
        raise RuntimeError(f"FastMCP list error: {e}") from e


def list_policy_documents_via_mcp(allowed_directory: Path | str) -> list[str]:
    allowed_dir = Path(allowed_directory).resolve()
    return asyncio.run(_async_list_policy_documents(allowed_dir))
