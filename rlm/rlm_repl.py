"""
Simple Recursive Language Model (RLM) with REPL environment.
"""

from typing import Dict, List, Optional, Any 

from rlm import RLM
from rlm.rlm import LLMResult
from rlm.repl import REPLEnv
from rlm.utils.llm import create_llm_client
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    LLM Client that can handle long contexts by recursively calling itself.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 client_backend: Optional[str] = None,
                 recursive_api_key: Optional[str] = None,
                 recursive_base_url: Optional[str] = None,
                 recursive_client_backend: Optional[str] = None,
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5",
                 max_iterations: int = 20,
                 depth: int = 0,
                 enable_logging: bool = False,
                 log_to_file: bool = False,
                 log_dir: str = "logs",
                 log_file: Optional[str] = None,
                 ):
        import os
        from datetime import datetime
        self.client_backend = client_backend or os.getenv("RLM_CLIENT_BACKEND", "openai")
        self.recursive_client_backend = (
            recursive_client_backend
            or os.getenv("RLM_RECURSIVE_CLIENT_BACKEND")
            or self.client_backend
        )
        recursive_uses_root_backend = self.recursive_client_backend == self.client_backend

        self.api_key = api_key
        self.base_url = base_url
        self.call_history: list[LLMResult] = []
        self.recursive_api_key = (
            recursive_api_key
            if recursive_api_key is not None
            else api_key if recursive_uses_root_backend else None
        )
        self.recursive_base_url = (
            recursive_base_url
            if recursive_base_url is not None
            else base_url if recursive_uses_root_backend else None
        )
        self.model = model
        self.recursive_model = recursive_model
        self.llm = create_llm_client(
            api_key=self.api_key,
            model=model,
            base_url=self.base_url,
            provider=self.client_backend,
            call_history=self.call_history,
        )
        
        # Track recursive call depth to prevent infinite loops
        self.repl_env = None
        self.depth = depth # Unused in this version.
        self._max_iterations = max_iterations
        self.log_file = log_file
        if log_to_file and self.log_file is None:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(log_dir, f"rlm_{timestamp}.txt")
        
        # Initialize colorful logger
        self.logger = ColorfulLogger(enabled=enable_logging, log_file=self.log_file)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging, log_file=self.log_file)
        
        self.messages = [] # Initialize messages list
        self.query = None
    
    def setup_context(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None):
        """
        Setup the context for the RLMClient.

        Args:
            context: The large context to analyze in the form of a list of messages, string, or Dict
            query: The user's question
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt(query)
        self.logger.log_initial_messages(self.messages)
        
        # Initialize REPL environment with context data
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model=self.recursive_model,
            recursive_client_backend=self.recursive_client_backend,
            recursive_api_key=self.recursive_api_key,
            recursive_base_url=self.recursive_base_url,
            call_history=self.call_history,
        )
        
        return self.messages

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> str:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.
        """
        self.messages = self.setup_context(context, query)
        
        # Main loop runs for fixed # of root LM iterations
        for iteration in range(self._max_iterations):
            
            # Query root LM to interact with REPL environment
            llm_result = self._completion_with_metadata(
                self.messages + [next_action_prompt(query, iteration)]
            )
            response = llm_result.content
            
            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)
            self.logger.log_llm_call("ROOT LLM METADATA:", llm_result)
            
            # Process code execution or add assistant message
            if code_blocks is not None:
                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env, 
                    self.repl_env_logger, self.logger
                )
            else:
                # Add assistant message when there are no code blocks
                assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
                self.messages.append(assistant_message)
            
            # Check that model produced a final answer
            final_answer = utils.check_for_final_answer(
                response, self.repl_env, self.logger,
            )

            # In practice, you may need some guardrails here.
            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

            
        # If we reach here, no final answer was found in any iteration
        print("No final answer found in any iteration")
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        llm_result = self._completion_with_metadata(self.messages)
        final_answer = llm_result.content
        self.logger.log_llm_call("ROOT LLM METADATA:", llm_result)
        self.logger.log_final_response(final_answer)

        return final_answer

    def _completion_with_metadata(self, messages) -> LLMResult:
        if hasattr(self.llm, "completion_with_metadata"):
            return self.llm.completion_with_metadata(messages)
        return LLMResult(
            content=self.llm.completion(messages),
            provider=getattr(self.llm, "provider", self.client_backend),
            model=getattr(self.llm, "model", self.model),
        )
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        summary: Dict[str, Any] = {
            "num_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "cache_miss_input_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "cost": 0.0,
            "known_cost_calls": 0,
            "unknown_cost_calls": 0,
            "by_provider": {},
            "by_model": {},
        }

        def add_to_bucket(bucket: dict, result: LLMResult):
            usage = result.usage
            bucket["num_calls"] = bucket.get("num_calls", 0) + 1
            if usage:
                bucket["input_tokens"] = bucket.get("input_tokens", 0) + usage.input_tokens
                bucket["output_tokens"] = bucket.get("output_tokens", 0) + usage.output_tokens
                bucket["total_tokens"] = bucket.get("total_tokens", 0) + usage.total_tokens
                bucket["cached_input_tokens"] = bucket.get("cached_input_tokens", 0) + usage.cached_input_tokens
                bucket["cache_miss_input_tokens"] = bucket.get("cache_miss_input_tokens", 0) + usage.cache_miss_input_tokens
                bucket["cache_write_tokens"] = bucket.get("cache_write_tokens", 0) + usage.cache_write_tokens
                bucket["reasoning_tokens"] = bucket.get("reasoning_tokens", 0) + usage.reasoning_tokens
                bucket["audio_tokens"] = bucket.get("audio_tokens", 0) + usage.audio_tokens
            if result.cost is not None:
                bucket["cost"] = bucket.get("cost", 0.0) + result.cost
                bucket["known_cost_calls"] = bucket.get("known_cost_calls", 0) + 1
            else:
                bucket["unknown_cost_calls"] = bucket.get("unknown_cost_calls", 0) + 1

        for result in self.call_history:
            add_to_bucket(summary, result)

            provider_key = result.provider or "unknown"
            provider_bucket = summary["by_provider"].setdefault(provider_key, {})
            add_to_bucket(provider_bucket, result)

            model_key = result.model or "unknown"
            model_bucket = summary["by_model"].setdefault(model_key, {})
            add_to_bucket(model_bucket, result)

        if summary["known_cost_calls"] == 0:
            summary["cost"] = None

        return summary

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = None
        self.messages = []
        self.query = None
        self.call_history.clear()


if __name__ == "__main__":
    pass
