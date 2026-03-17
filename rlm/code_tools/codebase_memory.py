import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


class CodebaseMemoryClient:
    """Thin adapter around the `codebase-memory-mcp` CLI mode."""

    DEFAULT_COMMAND = "codebase-memory-mcp"
    SUPPORTED_TOOLS = (
        "index_repository",
        "search_graph",
        "search_code",
        "trace_call_path",
        "get_code_snippet",
        "get_architecture",
        "list_projects",
        "index_status",
    )

    def __init__(
        self,
        command: Optional[str] = None,
        repo_path: Optional[str] = None,
        timeout: int = 120,
    ):
        # Look for the binary in the local bin/ folder first
        local_bin_path = Path(__file__).parent.parent.parent / "bin" / self.DEFAULT_COMMAND
        default_cmd = str(local_bin_path) if local_bin_path.exists() else self.DEFAULT_COMMAND
        
        self.command = command or os.getenv("CODEBASE_MEMORY_MCP_CMD") or default_cmd
        self.repo_path = str(Path(repo_path).expanduser().resolve()) if repo_path else None
        self.timeout = timeout

    def is_available(self) -> bool:
        return shutil.which(self.command) is not None

    def tool_help(self) -> dict[str, Any]:
        return {
            "available": self.is_available(),
            "command": self.command,
            "repo_path": self.repo_path,
            "supported_tools": list(self.SUPPORTED_TOOLS),
        }

    def index_repository(self, repo_path: Optional[str] = None, mode: str = "full") -> dict[str, Any]:
        target_repo = self._resolve_repo_path(repo_path)
        if target_repo is None:
            return self._error("repo_path is required to index a repository.")
        return self.call_tool("index_repository", repo_path=target_repo, mode=mode)

    def search_graph(self, **kwargs) -> dict[str, Any]:
        return self.call_tool("search_graph", **kwargs)

    def search_code(self, **kwargs) -> dict[str, Any]:
        return self.call_tool("search_code", **kwargs)

    def trace_call_path(self, **kwargs) -> dict[str, Any]:
        return self.call_tool("trace_call_path", **kwargs)

    def get_code_snippet(self, **kwargs) -> dict[str, Any]:
        return self.call_tool("get_code_snippet", **kwargs)

    def get_architecture(self, **kwargs) -> dict[str, Any]:
        return self.call_tool("get_architecture", **kwargs)

    def list_projects(self) -> dict[str, Any]:
        return self.call_tool("list_projects")

    def index_status(self, project: Optional[str] = None) -> dict[str, Any]:
        payload = {}
        if project:
            payload["project"] = project
        return self.call_tool("index_status", **payload)

    def call_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        if tool_name not in self.SUPPORTED_TOOLS:
            return self._error(f"Unsupported codebase-memory tool: {tool_name}")
        if not self.is_available():
            return self._error(
                "codebase-memory-mcp is not installed or not on PATH.",
                hint=(
                    "Install the binary and ensure it is on PATH, or set "
                    "CODEBASE_MEMORY_MCP_CMD to the executable path."
                ),
            )

        payload = {key: value for key, value in kwargs.items() if value is not None}
        if tool_name == "index_repository" and "repo_path" not in payload and self.repo_path:
            payload["repo_path"] = self.repo_path

        command = [self.command, "cli", tool_name, json.dumps(payload)]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except Exception as exc:
            return self._error(f"Failed to invoke codebase-memory-mcp: {exc}")

        if result.returncode != 0:
            return self._error(
                f"codebase-memory-mcp exited with status {result.returncode}",
                stderr=result.stderr.strip() or None,
                stdout=result.stdout.strip() or None,
            )

        output = result.stdout.strip()
        if not output:
            return {"ok": True, "tool": tool_name, "result": None}

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = output

        return {
            "ok": True,
            "tool": tool_name,
            "command": self.command,
            "result": parsed,
        }

    def _resolve_repo_path(self, repo_path: Optional[str]) -> Optional[str]:
        target_repo = repo_path or self.repo_path
        if target_repo is None:
            return None
        return str(Path(target_repo).expanduser().resolve())

    def _error(self, message: str, **extra) -> dict[str, Any]:
        payload = {"ok": False, "error": message}
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        return payload
