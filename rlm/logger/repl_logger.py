from dataclasses import dataclass
from typing import List, Optional

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.rule import Rule

    RICH_AVAILABLE = True
except ImportError:
    Console = None
    Syntax = None
    Panel = None
    Text = None
    box = None
    Rule = None
    RICH_AVAILABLE = False


@dataclass
class CodeExecution:
    code: str
    stdout: str
    stderr: str
    execution_number: int
    execution_time: Optional[float] = None


class REPLEnvLogger:
    def __init__(self, max_output_length: int = 2000, enabled: bool = True):
        self.enabled = enabled and RICH_AVAILABLE
        self.console = Console() if self.enabled else None
        self.executions: List[CodeExecution] = []
        self.execution_count = 0
        self.max_output_length = max_output_length

    def _truncate_output(self, text: str) -> str:
        """Truncate text output to prevent overwhelming console output."""
        if len(text) <= self.max_output_length:
            return text

        half_length = self.max_output_length // 2
        first_part = text[:half_length]
        last_part = text[-half_length:]
        truncated_chars = len(text) - self.max_output_length
        return f"{first_part}\n\n... [TRUNCATED {truncated_chars} characters] ...\n\n{last_part}"

    def log_execution(self, code: str, stdout: str, stderr: str = "", execution_time: Optional[float] = None) -> None:
        self.execution_count += 1
        execution = CodeExecution(
            code=code,
            stdout=stdout,
            stderr=stderr,
            execution_number=self.execution_count,
            execution_time=execution_time,
        )
        self.executions.append(execution)

    def display_last(self) -> None:
        if not self.enabled:
            return
        if self.executions:
            self._display_single_execution(self.executions[-1])

    def display_all(self) -> None:
        if not self.enabled:
            return
        for i, execution in enumerate(self.executions):
            self._display_single_execution(execution)
            if i < len(self.executions) - 1:
                self.console.print(Rule(style="dim", characters="─"))
                self.console.print()

    def _display_single_execution(self, execution: CodeExecution) -> None:
        if not self.enabled:
            return
        timing_panel = None
        display_code = self._truncate_output(execution.code)
        input_panel = Panel(
            Syntax(display_code, "python", theme="monokai", line_numbers=True),
            title=f"[bold blue]In [{execution.execution_number}]:[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
        self.console.print(input_panel)

        if execution.stderr:
            display_stderr = self._truncate_output(execution.stderr)
            error_text = Text(display_stderr, style="bold red")
            output_panel = Panel(
                error_text,
                title=f"[bold red]Error in [{execution.execution_number}]:[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        elif execution.stdout:
            display_stdout = self._truncate_output(execution.stdout)
            output_text = Text(display_stdout, style="white")
            output_panel = Panel(
                output_text,
                title=f"[bold green]Out [{execution.execution_number}]:[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
            if execution.execution_time is not None:
                timing_panel = Panel(
                    Text(f"Execution time: {execution.execution_time:.4f}s", style="bright_black"),
                    border_style="grey37",
                    box=box.ROUNDED,
                    title=f"[bold grey37]Timing [{execution.execution_number}]:[/bold grey37]",
                )
        else:
            body = Text(
                f"Execution time: {execution.execution_time:.4f}s" if execution.execution_time is not None else "No output",
                style="dim",
            )
            output_panel = Panel(
                body,
                title=f"[bold dim]Out [{execution.execution_number}]:[/bold dim]",
                border_style="dim",
                box=box.ROUNDED,
            )

        self.console.print(output_panel)
        if timing_panel:
            self.console.print(timing_panel)

    def clear(self) -> None:
        self.executions.clear()
        self.execution_count = 0
