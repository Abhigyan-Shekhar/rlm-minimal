"""
Simple Recursive Language Model (RLM) with REPL environment.
"""

from typing import Dict, List, Optional, Any 

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.settings import load_settings
from rlm.utils.llm import get_llm_client
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
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5",
                 provider: str = "openai",
                 max_iterations: int = 20,
                 depth: int = 0,
                 enable_logging: bool = False,
                 repo_path: Optional[str] = None,
                 codebase_memory_command: Optional[str] = None,
                 max_sub_queries: Optional[int] = None,
                 repl_truncate_len: Optional[int] = None,
                 progress_callback=None,
                 ):
        settings = load_settings()
        self.api_key = api_key
        self.model = model or settings.model
        self.recursive_model = recursive_model or settings.recursive_model
        self.provider = provider or settings.provider
        self.llm = get_llm_client(provider=self.provider, api_key=api_key, model=self.model)
        
        # Track recursive call depth to prevent infinite loops
        self.repl_env = None
        self.depth = depth # Unused in this version.
        self._max_iterations = max_iterations or settings.max_iterations
        self.max_sub_queries = max_sub_queries or settings.max_sub_queries
        self.repl_truncate_len = repl_truncate_len or settings.repl_truncate_len
        self.progress_callback = progress_callback
        
        # Initialize colorful logger
        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(
            max_output_length=self.repl_truncate_len,
            enabled=enable_logging,
        )
        self.repo_path = repo_path
        self.codebase_memory_command = codebase_memory_command
        self._active_repo_path = repo_path
        
        self.messages = [] # Initialize messages list
        self.query = None
        self.last_run_iterations = 0
        self.last_run_status = "unknown"
        self.last_context_metadata: Dict[str, Any] = {}

    def _handle_model_failure(self, exc: Exception) -> str:
        """Return a user-visible failure message for model-limit errors."""
        error_text = str(exc).strip()
        lowered = error_text.lower()

        limit_markers = (
            "rate limit",
            "quota",
            "context length",
            "maximum context length",
            "max context",
            "too many tokens",
            "token limit",
            "request too large",
        )
        if any(marker in lowered for marker in limit_markers):
            hint = ""
            if self._active_repo_path is None:
                hint = " Hint: use repo mode for codebases or try a smaller plain-text input."
            return (
                "MODEL_FAILURE: The model limit was reached while processing this request. "
                f"Details: {error_text}{hint}"
            )

        return f"MODEL_FAILURE: The model request failed. Details: {error_text}"

    def _safe_completion(self, messages) -> str:
        """Wrap model calls so limit errors return a clear failure string."""
        try:
            return self.llm.completion(messages)
        except Exception as exc:
            failure_message = self._handle_model_failure(exc)
            self.logger.log_final_response(failure_message)
            return failure_message
    
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
        self.last_run_iterations = 0
        self.last_run_status = "running"

        repo_path = self.repo_path
        if isinstance(context, dict) and isinstance(context.get("repo_path"), str):
            repo_path = context["repo_path"]
        self._active_repo_path = repo_path
        if self._active_repo_path:
            self.last_context_metadata = {"repo_path": self._active_repo_path}
        elif isinstance(context, str):
            self.last_context_metadata = {"source_path": None, "char_count": len(context)}
        else:
            self.last_context_metadata = {"source_path": None}

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt(enable_codebase_tools=bool(self._active_repo_path))
        self.logger.log_initial_messages(self.messages)
        
        # Initialize REPL environment with context data
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
            provider=self.provider,
            repo_path=self._active_repo_path,
            codebase_memory_command=self.codebase_memory_command,
            max_sub_queries=self.max_sub_queries,
            max_output_length=self.repl_truncate_len,
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
            self.last_run_iterations = iteration + 1
            if hasattr(self.repl_env, "set_current_root_iteration"):
                self.repl_env.set_current_root_iteration(iteration)
            self._emit_progress({"iteration": iteration + 1, "event": "root_iteration_start"})
            
            # Query root LM to interact with REPL environment
            response = self._safe_completion(self.messages + [next_action_prompt(query, iteration)])
            if response.startswith("MODEL_FAILURE:"):
                self.last_run_status = "model_failure"
                return response
            
            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)
            
            # Process code execution or add assistant message
            if code_blocks is not None:
                self._emit_progress({"iteration": iteration + 1, "event": "code_execution", "code_blocks": len(code_blocks)})
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
                self.last_run_status = "success"
                return final_answer

            if hasattr(self.repl_env, "should_force_stop_for_sub_query_limit") and self.repl_env.should_force_stop_for_sub_query_limit(iteration):
                failure_message = (
                    "MODEL_FAILURE: Sub-query limit reached and the model did not converge "
                    "to a final answer."
                )
                self.logger.log_final_response(failure_message)
                self.last_run_status = "model_failure"
                return failure_message

            
        # If we reach here, no final answer was found in any iteration
        print("No final answer found in any iteration")
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self._safe_completion(self.messages)
        self.logger.log_final_response(final_answer)
        self.last_run_status = "model_failure" if final_answer.startswith("MODEL_FAILURE:") else "success"

        return final_answer
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        raise NotImplementedError("Cost tracking not implemented for RLM REPL.")

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv(provider=self.provider)
        self.messages = []
        self.query = None
        self._active_repo_path = self.repo_path
        self.last_run_iterations = 0
        self.last_run_status = "unknown"

    def _emit_progress(self, payload: Dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)


if __name__ == "__main__":
    pass
