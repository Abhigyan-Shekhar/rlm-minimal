import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rlm.rlm_repl import RLM_REPL
from rlm.settings import load_settings
from rlm.trajectory import write_trajectory


class RLMCLI:
    def __init__(self, args):
        self.args = args
        self.settings = load_settings()
        self.file_context: Optional[str] = None
        self.file_source_path: Optional[str] = None
        self.repo_path: Optional[str] = None

    def run(self) -> int:
        if self.args.query or self.args.file or self.args.repo:
            return self._run_single_shot()
        return self._run_interactive()

    def _build_rlm(self, progress_callback=None) -> RLM_REPL:
        provider = self.args.provider or self.settings.provider
        model = self.args.model or self.settings.model
        recursive_model = self.args.recursive_model or self.settings.recursive_model
        max_iterations = self.args.max_iterations or self.settings.max_iterations
        return RLM_REPL(
            provider=provider,
            model=model,
            recursive_model=recursive_model,
            max_iterations=max_iterations,
            repo_path=self.repo_path or self.args.repo,
            codebase_memory_command=self.settings.codebase_memory_command,
            max_sub_queries=self.settings.max_sub_queries,
            repl_truncate_len=self.settings.repl_truncate_len,
            progress_callback=progress_callback,
        )

    def _run_single_shot(self) -> int:
        if self.args.file:
            self._load_file(self.args.file)
        if self.args.repo:
            self._set_repo(self.args.repo)
        query = " ".join(self.args.query).strip()
        if not query:
            print("A query is required in single-shot mode.", file=sys.stderr)
            return 2
        progress = self._make_progress_callback(self.args.verbose)
        answer, trajectory_path = self._execute_query(query, progress_callback=progress)
        if self.args.verbose:
            print(f"Trajectory: {trajectory_path}", file=sys.stderr)
        print(answer)
        return 1 if answer.startswith("MODEL_FAILURE:") else 0

    def _run_interactive(self) -> int:
        print("RLM interactive mode. Type /help for commands.")
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not raw:
                continue
            if raw.startswith("/"):
                command_result = self._handle_command(raw)
                if command_result is not None:
                    return command_result
                continue
            answer, trajectory_path = self._execute_query(
                raw,
                progress_callback=self._make_progress_callback(self.args.verbose),
            )
            print(answer)
            if self.args.verbose:
                print(f"[trajectory] {trajectory_path}", file=sys.stderr)

    def _handle_command(self, raw: str) -> Optional[int]:
        parts = shlex.split(raw)
        command = parts[0]
        if command == "/help":
            print("/file <path>  load a plain-text file")
            print("/repo <path>  set repo-aware mode")
            print("/clear        clear current file/repo context")
            print("/quit         exit")
            return None
        if command == "/quit":
            return 0
        if command == "/clear":
            self.file_context = None
            self.file_source_path = None
            self.repo_path = None
            print("Cleared file and repo context.")
            return None
        if len(parts) != 2:
            print("Expected exactly one path argument.", file=sys.stderr)
            return None
        if command == "/file":
            self._load_file(parts[1])
            print(f"Loaded file context: {self.file_source_path}")
            return None
        if command == "/repo":
            self._set_repo(parts[1])
            print(f"Set repo context: {self.repo_path}")
            return None
        print(f"Unknown command: {command}", file=sys.stderr)
        return None

    def _load_file(self, path_str: str) -> None:
        path = Path(path_str).expanduser().resolve()
        self.file_context = path.read_text(encoding="utf-8")
        self.file_source_path = str(path)
        self.repo_path = None

    def _set_repo(self, path_str: str) -> None:
        path = Path(path_str).expanduser().resolve()
        self.repo_path = str(path)
        self.file_context = None
        self.file_source_path = None

    def _execute_query(self, query: str, progress_callback=None) -> tuple[str, Path]:
        rlm = self._build_rlm(progress_callback=progress_callback)
        context = self._build_context()
        if self.file_context is not None and len(self.file_context) > self.settings.context_warn_chars:
            print(
                "Warning: plain-text context is large and may exceed model limits. "
                "Consider repo mode for codebases.",
                file=sys.stderr,
            )
        answer = rlm.completion(context=context, query=query)
        trajectory_path = write_trajectory(
            self.settings.state_dir,
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "mode": "repo" if self.repo_path else "text",
                "provider": rlm.provider,
                "model": rlm.model,
                "recursive_model": rlm.recursive_model,
                "final_result": answer,
                "final_status": "model_failure" if answer.startswith("MODEL_FAILURE:") else "success",
                "total_iterations": rlm.last_run_iterations,
                "context_metadata": self._build_context_metadata(),
            },
        )
        return answer, trajectory_path

    def _build_context(self):
        if self.repo_path:
            return {"repo_path": self.repo_path}
        return self.file_context or ""

    def _build_context_metadata(self) -> dict:
        if self.repo_path:
            return {"repo_path": self.repo_path}
        return {"source_path": self.file_source_path, "char_count": len(self.file_context or "")}

    def _make_progress_callback(self, verbose: bool):
        if not verbose:
            return None

        def callback(payload: dict) -> None:
            iteration = payload.get("iteration", "?")
            event = payload.get("event", "event")
            if event == "code_execution":
                code_blocks = payload.get("code_blocks", 0)
                print(f"[iteration {iteration}] executed {code_blocks} code block(s)", file=sys.stderr)
            else:
                print(f"[iteration {iteration}] {event}", file=sys.stderr)

        return callback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recursive Language Models CLI")
    parser.add_argument("--file", help="Load a plain-text file as context")
    parser.add_argument("--repo", help="Use a local repository in repo-aware mode")
    parser.add_argument("--provider", help="LLM provider override")
    parser.add_argument("--model", help="Root model override")
    parser.add_argument("--recursive-model", help="Recursive model override")
    parser.add_argument("--max-iterations", type=int, help="Maximum root iterations")
    parser.add_argument("--verbose", action="store_true", help="Print progress/debug info to stderr")
    parser.add_argument("query", nargs="*", help="Query to run in single-shot mode")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return RLMCLI(args).run()
