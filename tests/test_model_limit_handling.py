from rlm.rlm_repl import RLM_REPL


class FailingLLMClient:
    def completion(self, messages):
        raise RuntimeError("Error generating completion: Rate limit exceeded for gpt-5")


class DummyREPLEnv:
    def __init__(self, *args, **kwargs):
        self.locals = {}


def test_completion_returns_clear_failure_when_model_limit_is_reached(monkeypatch):
    monkeypatch.setattr("rlm.rlm_repl.get_llm_client", lambda **kwargs: FailingLLMClient())
    monkeypatch.setattr("rlm.rlm_repl.REPLEnv", DummyREPLEnv)

    rlm = RLM_REPL(api_key="test-key", provider="openai", model="gpt-5")

    result = rlm.completion(context="hello world", query="say hi")

    assert result.startswith("MODEL_FAILURE:")
    assert "model limit was reached" in result.lower()
    assert "rate limit exceeded" in result.lower()
