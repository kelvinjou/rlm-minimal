"""
Root (colorful) logger for RLM client that tracks model outputs and message changes.
"""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class ColorfulLogger:
    """
    A colorful logger that tracks RLM client interactions with the model.
    """
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        'BG_RED': '\033[41m',
        'BG_GREEN': '\033[42m',
        'BG_YELLOW': '\033[43m',
        'BG_BLUE': '\033[44m',
        'BG_MAGENTA': '\033[45m',
        'BG_CYAN': '\033[46m',
    }
    
    def __init__(
        self,
        enabled: bool = True,
        log_file: Optional[str] = None,
        write_file_header: bool = True,
    ):
        """
        Initialize the colorful logger.
        
        Args:
            enabled: Whether console logging is enabled
            log_file: Optional path to a plain-text log transcript.
            write_file_header: Whether to initialize the file with a header.
        """
        self.enabled = enabled
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            if write_file_header:
                self.log_file.write_text(
                    f"RLM log started at {datetime.now().isoformat(timespec='seconds')}\n\n"
                )
        self.conversation_step = 0
        self.last_messages_length = 0
        self.current_query = ""
        self.session_start_time = None
        self.current_depth = 0

    def _write_file(self, text: str = ""):
        if not self.log_file:
            return
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _emit(self, text: str = "", color: Optional[str] = None):
        if self.enabled:
            print(self._colorize(text, color) if color else text)
        self._write_file(text)
        
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if logging is enabled."""
        if not self.enabled:
            return text
        return f"{self.COLORS[color]}{text}{self.COLORS['RESET']}"
    
    def _print_separator(self, char: str = "=", color: str = "CYAN"):
        """Print a colored separator line."""
        separator = char * 80
        self._emit(separator, color)
    
    def log_query_start(self, query: str):
        """Log the start of a new query."""
        if not self.enabled and not self.log_file:
            return
            
        self.current_query = query
        self.conversation_step = 0
        self.last_messages_length = 0
        self.session_start_time = datetime.now()
        self.current_depth = 0
        
        self._print_separator("=", "GREEN")
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.enabled:
            print(self._colorize("STARTING NEW QUERY", "BOLD") + self._colorize(" | ", "DIM") + 
                  self._colorize(timestamp, "DIM"))
        self._write_file(f"STARTING NEW QUERY | {timestamp}")
        self._print_separator("=", "GREEN")
        
        if self.enabled:
            print(self._colorize("QUERY:", "BOLD") + f" {query}")
            print()
        self._write_file(f"QUERY: {query}")
        self._write_file()
    
    def log_initial_messages(self, messages: List[Dict[str, str]]):
        """Log the initial messages setup."""
        if not self.enabled and not self.log_file:
            return
            
        self._emit("INITIAL MESSAGES SETUP:", "BOLD")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Truncate very long content for readability
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            role_color = "BLUE" if role == "user" else "MAGENTA" if role == "assistant" else "YELLOW"
            if self.enabled:
                print(f"  {self._colorize(f'[{i+1}] {role.upper()}:', role_color)} {content}")
            self._write_file(f"  [{i+1}] {role.upper()}: {content}")
        
        if self.enabled:
            print()
        self._write_file()
        self.last_messages_length = len(messages)
    
    def log_model_response(self, response: str, has_tool_calls: bool):
        """Log the model's response."""
        if not self.enabled and not self.log_file:
            return
            
        self.conversation_step += 1
        
        self._emit(f"MODEL RESPONSE (Step {self.conversation_step}):", "BOLD")
        
        # Truncate very long responses for readability
        display_response = response
        if len(response) > 500:
            display_response = response[:500] + "..."
        
        if self.enabled:
            print(f"  {self._colorize('Response:', 'CYAN')} {display_response}")
        self._write_file(f"  Response: {display_response}")
        
        if has_tool_calls:
            self._emit("  Contains tool calls - will execute them", "YELLOW")
        else:
            self._emit("  No tool calls - final response", "GREEN")
        
        if self.enabled:
            print()
        self._write_file()
    
    def log_tool_execution(self, tool_call_str: str, tool_result: str):
        """Log tool execution and result."""
        if not self.enabled and not self.log_file:
            return
            
        self._emit("TOOL EXECUTION:", "BOLD")
        if self.enabled:
            print(f"  {self._colorize('Call:', 'YELLOW')} {tool_call_str}")
        self._write_file(f"  Call: {tool_call_str}")
        
        # Truncate very long results for readability
        display_result = tool_result
        if len(tool_result) > 300:
            display_result = tool_result[:300] + "..."
        
        if self.enabled:
            print(f"  {self._colorize('Result:', 'GREEN')} {display_result}")
            print()
        self._write_file(f"  Result: {display_result}")
        self._write_file()
    
    def log_final_response(self, response: str):
        """Log the final response from the model."""
        if not self.enabled and not self.log_file:
            return
            
        self._print_separator("=", "GREEN")
        self._emit("FINAL RESPONSE:", "BOLD")
        self._print_separator("=", "GREEN")
        self._emit(response)
        self._print_separator("=", "GREEN")
        if self.enabled:
            print() 
        self._write_file()
