from types import SimpleNamespace

from rlm.code_tools.codebase_memory import CodebaseMemoryClient


def test_missing_binary_returns_structured_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda command: None)
    client = CodebaseMemoryClient(repo_path=".")

    result = client.search_graph(name_pattern=".*main.*")

    assert result["ok"] is False
    assert "error" in result
    assert "hint" in result


def test_successful_cli_response_is_parsed(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/codebase-memory-mcp")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{"items": 3}', stderr=""),
    )
    client = CodebaseMemoryClient(repo_path=".")

    result = client.search_graph(name_pattern=".*main.*")

    assert result["ok"] is True
    assert result["result"] == {"items": 3}


def test_unsupported_tool_name_returns_adapter_error():
    client = CodebaseMemoryClient(repo_path=".")

    result = client.call_tool("detect_changes")

    assert result["ok"] is False
    assert "Unsupported" in result["error"]
