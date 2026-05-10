from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.rule import Rule
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rlm.rlm import LLMResult

@dataclass
class CodeExecution:
    code: str
    stdout: str
    stderr: str
    execution_number: int
    execution_time: Optional[float] = None
    llm_results: Optional[List[LLMResult]] = None

class REPLEnvLogger:
    def __init__(
        self,
        max_output_length: int = 2000,
        enabled: bool = True,
        log_file: Optional[str] = None,
    ):
        self.enabled = enabled
        self.console = Console()
        self.executions: List[CodeExecution] = []
        self.execution_count = 0
        self.max_output_length = max_output_length
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_file(self, text: str = ""):
        if not self.log_file:
            return
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    
    def _truncate_output(self, text: str) -> str:
        """Truncate text output to prevent overwhelming console output."""
        if len(text) <= self.max_output_length:
            return text
        
        # Show first half, then ellipsis, then last half
        half_length = self.max_output_length // 2
        first_part = text[:half_length]
        last_part = text[-half_length:]
        truncated_chars = len(text) - self.max_output_length
        
        return f"{first_part}\n\n... [TRUNCATED {truncated_chars} characters] ...\n\n{last_part}"
    
    def log_execution(
        self,
        code: str,
        stdout: str,
        stderr: str = "",
        execution_time: Optional[float] = None,
        llm_results: Optional[List[LLMResult]] = None,
    ) -> None:
        """Log a code execution with its output"""
        self.execution_count += 1
        execution = CodeExecution(
            code=code,
            stdout=stdout,
            stderr=stderr,
            execution_number=self.execution_count,
            execution_time=execution_time,
            llm_results=llm_results or [],
        )
        self.executions.append(execution)
        self._write_execution_to_file(execution)

    def _format_llm_metadata(self, result: LLMResult, call_number: int) -> str:
        usage = result.usage
        lines = [
            f"LLM call {call_number}:",
            f"  Provider: {result.provider or 'unknown'}",
            f"  Model: {result.model or 'unknown'}",
        ]
        if result.response_id:
            lines.append(f"  Response ID: {result.response_id}")
        if usage:
            lines.extend(
                [
                    f"  Input tokens: {usage.input_tokens}",
                    f"  Output tokens: {usage.output_tokens}",
                    f"  Total tokens: {usage.total_tokens}",
                    f"  Cached input tokens: {usage.cached_input_tokens}",
                    f"  Cache miss input tokens: {usage.cache_miss_input_tokens}",
                    f"  Cache write tokens: {usage.cache_write_tokens}",
                    f"  Reasoning tokens: {usage.reasoning_tokens}",
                    f"  Audio tokens: {usage.audio_tokens}",
                ]
            )
        else:
            lines.append("  Usage: unavailable")
        if result.cost is None:
            lines.append("  Cost: unavailable")
        else:
            cost_source = f" ({result.cost_source})" if result.cost_source else ""
            lines.append(f"  Cost: ${result.cost:.8f}{cost_source}")
        if result.upstream_cost is not None:
            lines.append(f"  Upstream cost: ${result.upstream_cost:.8f}")
        if result.reasoning_content:
            lines.extend(
                [
                    "  Reasoning content:",
                    "  <think>",
                    result.reasoning_content,
                    "  </think>",
                ]
            )
        return "\n".join(lines)

    def _write_execution_to_file(self, execution: CodeExecution) -> None:
        self._write_file(f"In [{execution.execution_number}]:")
        self._write_file(execution.code)
        if execution.stderr:
            self._write_file(f"Error in [{execution.execution_number}]:")
            self._write_file(self._truncate_output(execution.stderr))
        elif execution.stdout:
            self._write_file(f"Out [{execution.execution_number}]:")
            self._write_file(self._truncate_output(execution.stdout))
        else:
            self._write_file(f"Out [{execution.execution_number}]: No output")
        if execution.execution_time is not None:
            self._write_file(f"Timing [{execution.execution_number}]: {execution.execution_time:.4f}s")
        if execution.llm_results:
            self._write_file("LLM metadata:")
            for i, result in enumerate(execution.llm_results, start=1):
                self._write_file(self._format_llm_metadata(result, i))
        self._write_file()
    
    def display_last(self) -> None:
        """Display the last logged execution"""
        if not self.enabled:
            return
        if self.executions:
            self._display_single_execution(self.executions[-1])
    
    def display_all(self) -> None:
        """Display all logged executions in Jupyter-like format"""
        if not self.enabled:
            return
        for i, execution in enumerate(self.executions):
            self._display_single_execution(execution)
            # Add divider between cells (but not after the last one)
            if i < len(self.executions) - 1:
                self.console.print(Rule(style="dim", characters="─"))
                self.console.print()
    
    def _display_single_execution(self, execution: CodeExecution) -> None:
        """Display a single code execution like a Jupyter cell"""
        if not self.enabled:
            return
        # Input cell (code) - also truncate if too long
        timing_panel = None
        display_code = self._truncate_output(execution.code)
        input_panel = Panel(
            Syntax(display_code, "python", theme="monokai", line_numbers=True),
            title=f"[bold blue]In [{execution.execution_number}]:[/bold blue]",
            border_style="blue",
            box=box.ROUNDED
        )
        self.console.print(input_panel)
        
        # Output cell
        if execution.stderr:
            # Error output
            display_stderr = self._truncate_output(execution.stderr)
            error_text = Text(display_stderr, style="bold red")
            output_panel = Panel(
                error_text,
                title=f"[bold red]Error in [{execution.execution_number}]:[/bold red]",
                border_style="red",
                box=box.ROUNDED
            )
        elif execution.stdout:
            # Normal output with separate timing panel if available
            display_stdout = self._truncate_output(execution.stdout)
            output_text = Text(display_stdout, style="white")
            
            output_panel = Panel(
                output_text,
                title=f"[bold green]Out [{execution.execution_number}]:[/bold green]",
                border_style="green",
                box=box.ROUNDED
            )
            # Show timing as a separate panel for reliable rendering
            if execution.execution_time is not None:
                timing_panel = Panel(
                    Text(f"Execution time: {execution.execution_time:.4f}s", style="bright_black"),
                    border_style="grey37",
                    box=box.ROUNDED,
                    title=f"[bold grey37]Timing [{execution.execution_number}]:[/bold grey37]"
                )
        else:
            # No output but still show timing if available
            if execution.execution_time is not None:
                timing_text = Text(f"Execution time: {execution.execution_time:.4f}s", style="dim")
                output_panel = Panel(
                    timing_text,
                    title=f"[bold dim]Out [{execution.execution_number}]:[/bold dim]",
                    border_style="dim",
                    box=box.ROUNDED
                )
                timing_panel = Panel(
                    Text(f"Execution time: {execution.execution_time:.4f}s", style="bright_black"),
                    border_style="grey37",
                    box=box.ROUNDED,
                    title=f"[bold grey37]Timing [{execution.execution_number}]:[/bold grey37]"
                )
            else:
                output_panel = Panel(
                    Text("No output", style="dim"),
                    title=f"[bold dim]Out [{execution.execution_number}]:[/bold dim]",
                    border_style="dim",
                    box=box.ROUNDED
                )
        
        self.console.print(output_panel)
        if timing_panel:
            self.console.print(timing_panel)
        if execution.llm_results:
            metadata_text = "\n\n".join(
                self._format_llm_metadata(result, i)
                for i, result in enumerate(execution.llm_results, start=1)
            )
            metadata_panel = Panel(
                Text(metadata_text, style="bright_black"),
                title=f"[bold grey37]LLM Metadata [{execution.execution_number}]:[/bold grey37]",
                border_style="grey37",
                box=box.ROUNDED
            )
            self.console.print(metadata_panel)
    
    def clear(self) -> None:
        """Clear all logged executions"""
        self.executions.clear()
        self.execution_count = 0
