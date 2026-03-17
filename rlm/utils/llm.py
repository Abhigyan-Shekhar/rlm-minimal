"""
LLM Client wrappers for OpenAI and Gemini models.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.

    def completion(
        self,
        messages: "list[dict[str, str]] | str",
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


class GeminiClient:
    """
    Google Gemini client with the same interface as OpenAIClient.

    Requires:
        - ``google-generativeai`` package  (pip install google-generativeai)
        - ``GEMINI_API_KEY`` environment variable  (or pass api_key directly)

    Supported models (examples):
        ``gemini-2.5-flash``, ``gemini-1.5-pro``, ``gemini-2.0-flash``
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        genai.configure(api_key=self.api_key)
        self._genai = genai
        self.client = genai.GenerativeModel(model)

    def _to_gemini_messages(self, messages: "list[dict[str, str]] | str") -> tuple:
        """
        Convert OpenAI-style messages to a Gemini history + final user prompt pair.

        Returns:
            (history, user_prompt) where history is a list of Content dicts
            and user_prompt is the last user message string.
        """
        if isinstance(messages, str):
            return [], messages
        if isinstance(messages, dict):
            messages = [messages]

        history = []
        user_prompt = ""

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map OpenAI roles → Gemini roles
            gemini_role = "model" if role == "assistant" else "user"

            # The last message is sent as the prompt; everything before is history
            if i < len(messages) - 1:
                history.append({"role": gemini_role, "parts": [content]})
            else:
                user_prompt = content

        return history, user_prompt

    def completion(
        self,
        messages: "list[dict[str, str]] | str",
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            history, user_prompt = self._to_gemini_messages(messages)

            generation_config = {}
            if max_tokens is not None:
                generation_config["max_output_tokens"] = max_tokens

            if history:
                chat = self.client.start_chat(history=history)
                response = chat.send_message(
                    user_prompt,
                    generation_config=generation_config or None,
                )
            else:
                response = self.client.generate_content(
                    user_prompt,
                    generation_config=generation_config or None,
                )

            return response.text

        except Exception as e:
            raise RuntimeError(f"Error generating Gemini completion: {str(e)}")


def get_llm_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Factory function — returns the right LLM client for the given provider.

    Args:
        provider: ``"openai"`` or ``"gemini"``
        api_key:  API key (falls back to env vars ``OPENAI_API_KEY`` / ``GEMINI_API_KEY``)
        model:    Model name. Defaults to ``"gpt-5"`` for OpenAI,
                  ``"gemini-2.5-flash"`` for Gemini.

    Returns:
        An :class:`OpenAIClient` or :class:`GeminiClient` instance.

    Example::

        client = get_llm_client("gemini", model="gemini-2.5-flash")
        print(client.completion("Hello!"))
    """
    provider = provider.lower().strip()

    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or "gpt-5")
    elif provider in ("gemini", "google"):
        return GeminiClient(api_key=api_key, model=model or "gemini-2.5-flash")
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported providers: 'openai', 'gemini'."
        )