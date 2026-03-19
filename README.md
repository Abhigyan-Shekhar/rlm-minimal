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
- Interactive CLI and single-shot CLI mode with `/repo` and `/file` commands
- Robust **Gemini API Integration** with automatic role-alternating and system instruction support
- **Repo-aware mode** using `codebase-memory-mcp` for surgical codebase exploration
- Integrated **LongBench v2** evaluation harness for measuring long-context performance
- Transparent trajectory logging for every model interaction

## Installation

### Requirements

- Python 3.11+
- [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) installed and on your PATH (for repo mode)
- Gemini or OpenAI API key

### Install dependencies

```bash
pip install -r requirements.txt
```

### Optional: Benchmarking with LongBench

To run the LongBench v2 benchmark using RLM:

```bash
# Clone the LongBench repo and set up its environment (done automatically in this repo)
# Then run the specialized prediction script:
./LongBench/venv/bin/python rlm_pred.py --limit 10
```

> [!NOTE]
> Testing on LongBench v2 (especially with free-tier Gemini keys) frequently results in `429 RESOURCE_EXHAUSTED` due to the extremely large context sizes (>100k tokens per sample). A high-quota API key is recommended for full evaluation.

## Python API

### Plain text mode

```python
from rlm.rlm_repl import RLM_REPL

rlm = RLM_REPL(
    provider="gemini",
    model="gemini-2.0-flash",
    recursive_model="gemini-2.0-flash",
    max_iterations=10,
)

answer = rlm.completion(
    context="Your long document...",
    query="Summarize the key points.",
)
```

### Repo-aware mode

```python
rlm = RLM_REPL(
    provider="gemini",
    repo_path="/path/to/repo",
)

answer = rlm.completion(
    context={"repo_path": "/path/to/repo"},
    query="Explain the authentication flow.",
)
```

## Repo-Aware Code Exploration

This project utilizes `codebase-memory-mcp` to provide the model with a "memory" of the codebase. Instead of sending the entire repository to the LLM, RLM uses a persistent REPL to query a local index. This results in:
- **Massive Token Savings**: Only relevant snippets are fetched.
- **Zero-Hallucination Grounding**: The model verifies its understanding against the actual code.

## Robust Gemini Support

The `GeminiClient` in `rlm/utils/llm.py` has been specifically optimized for the latest Gemini 2.0 models:
- **Role Merging**: Automatically handles Gemini's strict user-model-user alternating requirement.
- **System Instructions**: Correctly routes system prompts to the native Gemini API.
- **Thinking Models**: Robustly handles multi-part content from "thinking" or reasoning models.

## Benchmarking & Evaluation

The repository now includes a dedicated harness for [LongBench v2](https://github.com/THUDM/LongBench):
- `rlm_pred.py`: A wrapper that passes LongBench samples through the recursive RLM loop.
- Supports evaluating RLM's reasoning and retrieval capabilities on massive contexts.

## Limitations

- repo-aware mode depends on an external locally installed `codebase-memory-mcp` backend
- performance on free-tier APIs is limited by aggressive rate limits on large contexts

## License

This repo inherits from the minimal public RLM implementation and keeps the upstream license structure. See [LICENSE](LICENSE).
