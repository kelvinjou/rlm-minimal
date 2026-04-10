"""
Simple Recursive Language Model (RLM) with REPL environment.
"""

from typing import Dict, List, Optional, Any 

from rlm import RLM
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
        self.messages = build_system_prompt()
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
            response = self.llm.completion(self.messages + [next_action_prompt(query, iteration)])
            
            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)
            
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
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)

        return final_answer
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        raise NotImplementedError("Cost tracking not implemented for RLM REPL.")

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv()
        self.messages = []
        self.query = None


if __name__ == "__main__":
    pass
