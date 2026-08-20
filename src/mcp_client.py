"""Client for the official Model Context Protocol (MCP) Filesystem server."""

import asyncio
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def validate_file_path(file_path: Path | str, allowed_directory: Path | str) -> Path:
    """
    Validate that a requested file path resolves safely inside the allowed directory.

    Raises:
        ValueError: If the file path resolves outside the allowed directory.
        FileNotFoundError: If the file does not exist.
    """
    allowed_dir = Path(allowed_directory).resolve()
    target_file = Path(file_path).resolve()

    try:
        target_file.relative_to(allowed_dir)
    except ValueError:
        raise ValueError(
            f"Access denied: Requested file '{target_file}' is outside allowed directory '{allowed_dir}'."
        )

    if not target_file.exists():
        raise FileNotFoundError(f"File not found: '{target_file}'")

    return target_file


def extract_text_from_mcp_result(result: Any) -> str:
    """Extract plain text from an MCP tool call result."""
    if result is None:
        raise ValueError("Invalid MCP response: Result is None.")

    is_error = getattr(result, "is_error", False)
    if is_error:
        raise RuntimeError(f"MCP tool reported an error during execution: {result}")

    content_list = getattr(result, "content", None)
    if not content_list:
        raise ValueError("MCP tool returned an empty content list.")

    extracted_parts: list[str] = []
    for item in content_list:
        if hasattr(item, "text"):
            extracted_parts.append(item.text)
        elif isinstance(item, dict) and "text" in item:
            extracted_parts.append(str(item["text"]))
        else:
            extracted_parts.append(str(item))

    text = "\n".join(extracted_parts).strip()
    if not text:
        raise ValueError("MCP tool returned empty text content.")

    return text


def build_server_parameters(allowed_directory: Path) -> StdioServerParameters:
    """Build stdio server parameters for the official Filesystem MCP server."""
    resolved_dir = allowed_directory.resolve()
    env = dict(os.environ)

    # Ensure Node.js standard directory is available in PATH on Windows
    node_dir = r"C:\Program Files\nodejs"
    current_path = env.get("PATH", "")
    if node_dir not in current_path and os.path.exists(node_dir):
        env["PATH"] = f"{node_dir};{current_path}"

    if os.name == "nt":
        command = "cmd"
        args = ["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", str(resolved_dir)]
    else:
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", str(resolved_dir)]

    return StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )


async def _async_read_text_file(validated_file: Path, allowed_directory: Path) -> str:
    """Connect to Filesystem MCP server via stdio and read file content."""
    server_params = build_server_parameters(allowed_directory)

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "read_text_file",
                    arguments={"path": str(validated_file)},
                )
                return extract_text_from_mcp_result(result)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to launch Filesystem MCP server: 'npx' or 'cmd' executable not found. "
            "Please ensure Node.js is installed and available in PATH."
        ) from e
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Filesystem MCP session error: {e}") from e


def read_text_file_via_mcp(file_path: Path | str, allowed_directory: Path | str) -> str:
    """
    Synchronous entry point to read a text file using the official Filesystem MCP server.

    Validates that the file resides in the allowed directory, starts the MCP server over stdio,
    calls the `read_text_file` tool, and returns the extracted text content.
    """
    validated_file = validate_file_path(file_path, allowed_directory)
    allowed_dir = Path(allowed_directory).resolve()

    return asyncio.run(_async_read_text_file(validated_file, allowed_dir))
