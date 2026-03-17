import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RLMSettings:
    provider: str = os.getenv("RLM_PROVIDER", "openai")
    model: str = os.getenv("RLM_MODEL", "gpt-5")
    recursive_model: str = os.getenv("RLM_RECURSIVE_MODEL", "gpt-5")
    max_iterations: int = _int_env("RLM_MAX_ITERATIONS", 20)
    max_sub_queries: int = _int_env("RLM_MAX_SUB_QUERIES", 50)
    repl_truncate_len: int = _int_env("RLM_REPL_TRUNCATE_LEN", 2000)
    context_warn_chars: int = _int_env("RLM_CONTEXT_WARN_CHARS", 100000)
    codebase_memory_command: str | None = os.getenv("CODEBASE_MEMORY_MCP_CMD")
    state_dir: Path = Path(os.getenv("RLM_STATE_DIR", "~/.rlm")).expanduser()


def load_settings() -> RLMSettings:
    return RLMSettings()

