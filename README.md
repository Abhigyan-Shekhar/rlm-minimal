# Recursive Language Models CLI

A practical, Python-first implementation of **Recursive Language Models (RLMs)** with:

- a persistent Python REPL runtime,
- recursive sub-queries over large contexts,
- an interactive CLI,
- optional repo-aware code exploration via [`codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp),
- lightweight trajectory logging for debugging and paper experiments.

This repo started from the minimal public RLM implementation and has been extended into a more usable local research tool.

- Original paper: [Recursive Language Models](https://arxiv.org/abs/2512.24601v1)
- Original minimal repo: [alexzhang13/rlm-minimal](https://github.com/alexzhang13/rlm-minimal)
- Original blog post: [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/)

![teaser](media/rlm.png)

## Core Principle: Efficient Repository Understanding

The primary goal of this implementation is to pair **Recursive Language Models (RLMs)** with **Repo-Aware Indexing (`codebase-memory-mcp`)**.

- **Improved Understanding**: By providing the model with structural tools (like `search_graph` and `get_architecture`), it can navigate complex codebases with surgical precision rather than trying to digest the entire repository at once.
- **Token Efficiency**: Instead of sending tens of thousands of lines of code to the LLM (which is expensive and often exceeds context limits), the model uses the REPL to fetch only the most relevant snippets.
- **Hallucination Reduction**: The model's answers are grounded in the actual output of the code exploration tools. It doesn't have to "guess" where a function is defined; it verifies it in the REPL before formulating a response.

## What This Repo Is Good For

This implementation is best suited for:

- large single-document QA and summarization,
- long-context decomposition via recursive sub-queries,
- codebase understanding when paired with `codebase-memory-mcp`,
- controlled benchmark runs where you want transparent trajectories and failure modes.

It is **not** yet a full agent framework, long-horizon conversational memory system, or benchmark suite.

## Features

- `RLM_REPL(...).completion(context, query)` as the main programmatic API
- interactive CLI and single-shot CLI mode
- structured failure messages for model-limit and provider errors
- sub-query guardrails to avoid unbounded recursive loops
- repo-aware mode using `codebase-memory-mcp`
- per-run trajectory logging under `~/.rlm/trajectories/`
- optional `rich`-based REPL visualization when installed

## Installation

### Requirements

- Python 3.11+
- one provider API key for the models you want to run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Optional dependency for richer REPL logging

```bash
pip install rich
```

### Environment variables

Supported runtime configuration:

- `RLM_PROVIDER` (default: `openai`)
- `RLM_MODEL` (default: `gpt-5` or `gemini-2.0-flash`)
- `RLM_RECURSIVE_MODEL` (default: `gpt-5` or `gemini-2.0-flash`)
- `RLM_MAX_ITERATIONS`
- `RLM_MAX_SUB_QUERIES`
- `RLM_REPL_TRUNCATE_LEN`
- `RLM_CONTEXT_WARN_CHARS`
- `RLM_STATE_DIR`
- `CODEBASE_MEMORY_MCP_CMD` (absolute path to `codebase-memory-mcp` binary)

Provider keys are loaded through normal environment variables and `.env`:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

## Quick Start

### Interactive mode

```bash
python3 main.py
```

Available commands:

- `/file <path>`: load a plain-text file as context
- `/repo <path>`: switch to repo-aware mode
- `/clear`: clear current file/repo context
- `/help`: show commands
- `/quit`: exit

Any non-command input is treated as the query.

### Single-shot mode

```bash
python3 main.py --file README.md "Summarize this document"
python3 main.py --repo /absolute/path/to/repo "What are the main entrypoints?"
python3 main.py --repo /absolute/path/to/repo --verbose "Trace the payment flow"
```

In single-shot mode:

- final answer goes to `stdout`
- verbose progress goes to `stderr`
- exit code is non-zero on `MODEL_FAILURE`

## Python API

### Plain text mode

```python
from rlm.rlm_repl import RLM_REPL

rlm = RLM_REPL(
    provider="openai",
    model="gpt-5",
    recursive_model="gpt-5-nano",
    max_iterations=10,
)

answer = rlm.completion(
    context="Your long document goes here...",
    query="Summarize the main argument and list the risks.",
)

print(answer)
```

### Repo-aware mode

```python
from rlm.rlm_repl import RLM_REPL

rlm = RLM_REPL(
    provider="gemini",
    model="gemini-2.0-flash",
    recursive_model="gemini-2.0-flash",
    repo_path="/absolute/path/to/repo",
)

answer = rlm.completion(
    context={"repo_path": "/absolute/path/to/repo"},
    query="What are the main entrypoints and who calls the payment flow?",
)

print(answer)
```

## Repo-Aware Code Exploration

This repo can optionally use [`codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp) as an external local code-intelligence backend.

Install that tool separately and either:

- put `codebase-memory-mcp` on your `PATH`, or
- set `CODEBASE_MEMORY_MCP_CMD=/path/to/codebase-memory-mcp`

When repo mode is active, the REPL exposes:

- `codebase_tool_help()`
- `index_repository()`
- `search_graph(...)`
- `search_code(...)`
- `trace_call_path(...)`
- `get_code_snippet(...)`
- `get_architecture(...)`
- `list_indexed_projects()`
- `index_status(...)`

If `codebase-memory-mcp` is missing, the tool layer fails gracefully with structured error payloads. When available, it drastically reduces token usage by enabling the model to search and index code rather than ingesting full contexts.

### Robust Gemini Support
The `GeminiClient` has been enhanced to:
- Correctly handle "thinking" models (e.g., `gemini-2.5-flash`) by safely extracting multi-part content.
- Map `system` messages to the native `system_instruction` API.
- Automatically merge consecutive user/model messages to satisfy Gemini's strict alternating-role requirements.

## Guardrails and Failure Behavior

### Model failures

Root model failures are returned as strings, not uncaught exceptions. Examples:

- `MODEL_FAILURE: The model limit was reached ...`
- `MODEL_FAILURE: The model request failed ...`

For plain-text contexts, limit failures include a hint to try repo mode or a smaller input.

### Sub-query limit

Recursive `llm_query(...)` calls are capped by `RLM_MAX_SUB_QUERIES`.

When the cap is hit:

- sub-queries return an in-band error string telling the model to produce its best final answer
- if the model continues for two full root iterations without converging, the run ends with:
  - `MODEL_FAILURE: Sub-query limit reached and the model did not converge to a final answer.`

### Large plain-text files

This repo does **not** auto-truncate or auto-chunk your loaded file before the root call.

Instead:

- a warning is emitted when the file exceeds `RLM_CONTEXT_WARN_CHARS`
- the run proceeds
- if the provider rejects the context, you get `MODEL_FAILURE`

For codebases, `/repo` is the recommended path.

## Trajectory Logging

Every run writes one JSON trajectory file for debugging/research.

Default location:

```text
~/.rlm/trajectories/
```

Override with:

```bash
export RLM_STATE_DIR=/custom/state/dir
```

Each trajectory contains:

- timestamp
- query
- mode (`text` or `repo`)
- provider/model info
- final result
- final status
- iteration count
- context metadata

It intentionally does **not** store the full raw context by default.

## Project Structure & File Overview

```text
.
├── main.py                # CLI entrypoint (orchestrates single-shot and interactive runs)
├── analyze.py             # Dedicated analysis script for repo-wide insights
├── rlm/
│   ├── cli.py             # CLI implementations (handles /repo and /file commands)
│   ├── rlm_repl.py        # Core RLM Logic: Manages the recursive loop and state
│   ├── repl.py            # Persistent Python REPL: Exposes MCP tools to the model
│   ├── code_tools/
│   │   └── codebase_memory.py # MCP Adapter: The bridge to codebase-memory-mcp terminal
│   ├── utils/
│   │   ├── llm.py         # LLM Clients: Robust Gemini/OpenAI drivers (role merging, text extraction)
│   │   ├── prompts.py     # System Prompts: Instructs the model on MCP tool usage
│   │   └── utils.py       # Helpers: Code block extraction and result formatting
│   ├── settings.py        # Settings: Environment variable management (LLM keys, MCP paths)
│   └── trajectory.py      # Logging: Detailed JSON traces of every model interaction
└── tests/                 # Test suite covering model-limit handling and guardrails
```

### Key Component Roles

- **`rlm_repl.py` (The Coordinator)**: This is the brain of the recursive process. It initializes the `CodebaseMemoryClient` and ensures the LLM is aware of the repository's context.
- **`code_tools/codebase_memory.py` (The Bridge)**: Acts as a lightweight adapter that turns `codebase-memory-mcp` CLI commands into interactive Python tools. This is the key to **reducing tokens**, as it offloads the heavy-lifting of code search to a specialized local indexer.
- **`repl.py` (The Sandbox)**: Exposes the `codebase_tool_*` suite directly into the LLM's workspace. By executing these tools in a real Python environment, the model significantly **reduces hallucinations** because it sees the actual truth of the filesystem.
- **`utils/llm.py` (The Engine)**: Recent enhancements ensure that Gemini "thinking" models remain robust when processing multi-step repository queries, handling complex role requirements automatically.

## Testing

Run the full local test suite with:

```bash
pytest -q tests
```

Current tests cover:

- model-limit failure handling
- adapter error shape and parsing
- CLI behavior
- sub-query guardrails
- `FINAL_VAR(...)` regression behavior
- trajectory file creation

## Research Notes

If you are using this repo for a paper, the strongest current evaluation story is:

- long-context understanding,
- evidence-based multi-hop QA with provided context,
- reasoning controls,
- transparent failure/trajectory analysis.

The current implementation is a good fit for benchmarks that reduce cleanly to:

- `context`
- `query`
- `gold answer`
- scalar metric like exact match, F1, or ROUGE

## Limitations

- no built-in benchmark harness yet
- no multi-turn conversational memory benchmark support
- no automatic backend installation for `codebase-memory-mcp`
- no directory-ingestion mode in the CLI
- repo-aware mode depends on an external locally installed backend

## License

This repo inherits from the minimal public RLM implementation and keeps the upstream license structure. See [LICENSE](LICENSE).
