from types import SimpleNamespace

from rlm.repl import REPLEnv, SUB_QUERY_LIMIT_MESSAGE
from rlm.rlm_repl import RLM_REPL


class DummyClient:
    api_key = "test"

    def completion(self, messages, **kwargs):
        return "sub-response"


def test_repl_sub_query_limit_returns_in_band_message(monkeypatch):
    monkeypatch.setattr("rlm.utils.llm.get_llm_client", lambda **kwargs: DummyClient())
    repl = REPLEnv(max_sub_queries=1)
    repl.set_current_root_iteration(0)

    first = repl.globals["llm_query"]("first prompt")
    second = repl.globals["llm_query"]("second prompt")

    assert first == "sub-response"
    assert second == SUB_QUERY_LIMIT_MESSAGE


class LimitIgnoringLLM:
    def __init__(self):
        self.calls = 0

    def completion(self, messages):
        self.calls += 1
        return "```repl\nprint(llm_query('chunk'))\n```"


class ControlledEnv:
    def __init__(self, *args, **kwargs):
        self.locals = {}
        self.current_root_iteration = 0
        self.sub_query_limit_hit_iteration = None

    def set_current_root_iteration(self, iteration: int):
        self.current_root_iteration = iteration

    def should_force_stop_for_sub_query_limit(self, iteration: int) -> bool:
        return self.sub_query_limit_hit_iteration is not None and iteration - self.sub_query_limit_hit_iteration >= 2

    def code_execution(self, code):
        if self.sub_query_limit_hit_iteration is None:
            self.sub_query_limit_hit_iteration = self.current_root_iteration
        return SimpleNamespace(stdout=SUB_QUERY_LIMIT_MESSAGE, stderr="", locals=self.locals, execution_time=0.01)


def test_root_loop_forces_failure_after_two_iterations_past_sub_query_limit(monkeypatch):
    monkeypatch.setattr("rlm.rlm_repl.get_llm_client", lambda **kwargs: LimitIgnoringLLM())
    monkeypatch.setattr("rlm.rlm_repl.REPLEnv", ControlledEnv)

    rlm = RLM_REPL(api_key="test-key", provider="openai", model="gpt-5", max_iterations=5)
    result = rlm.completion(context="hello world", query="say hi")

    assert "Sub-query limit reached" in result


class FinalVarLLM:
    def __init__(self):
        self.calls = 0

    def completion(self, messages):
        self.calls += 1
        if self.calls == 1:
            return "```repl\nresult = 'done'\n```"
        return "FINAL_VAR(result)"


class FinalVarEnv:
    def __init__(self, *args, **kwargs):
        self.locals = {}

    def set_current_root_iteration(self, iteration: int):
        return None

    def should_force_stop_for_sub_query_limit(self, iteration: int) -> bool:
        return False

    def code_execution(self, code):
        self.locals["result"] = "done"
        return SimpleNamespace(stdout="", stderr="", locals=self.locals, execution_time=0.01)


def test_final_var_resolution_still_works(monkeypatch):
    monkeypatch.setattr("rlm.rlm_repl.get_llm_client", lambda **kwargs: FinalVarLLM())
    monkeypatch.setattr("rlm.rlm_repl.REPLEnv", FinalVarEnv)

    rlm = RLM_REPL(api_key="test-key", provider="openai", model="gpt-5", max_iterations=5)
    result = rlm.completion(context="hello world", query="say hi")

    assert result == "done"
