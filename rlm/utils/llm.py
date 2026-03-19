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
        - ``google-genai`` package  (pip install google-genai)
        - ``GEMINI_API_KEY`` environment variable  (or pass api_key directly)

    Supported models (examples):
        ``gemini-2.0-flash``, ``gemini-1.5-pro``
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        try:
            import google.genai
        except ImportError:
            raise ImportError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self._client = google.genai.Client(api_key=self.api_key)

    def _to_gemini_messages(self, messages: "list[dict[str, str]] | str") -> tuple:
        """
        Convert OpenAI-style messages to a Gemini history + final user prompt pair + system instruction.

        Returns:
            (history, user_prompt, system_instruction)
        """
        if isinstance(messages, str):
            return [], messages, None
        if isinstance(messages, dict):
            messages = [messages]

        system_instructions = []
        processed_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instructions.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if processed_messages and processed_messages[-1]["role"] == gemini_role:
                processed_messages[-1]["parts"][0] += "\n" + content
            else:
                processed_messages.append({"role": gemini_role, "parts": [content]})

        if not processed_messages:
            return [], "", "\n".join(system_instructions) if system_instructions else None

        last_msg = processed_messages.pop()
        user_prompt = last_msg["parts"][0]
        
        if last_msg["role"] == "model":
            processed_messages.append(last_msg)
            user_prompt = ""

        return processed_messages, user_prompt, "\n".join(system_instructions) if system_instructions else None

    def _extract_text(self, response) -> str:
        """
        Safely extract text from the response, handling thinking models and multi-part content.
        """
        try:
            return response.text
        except ValueError:
            text_parts = []
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
            return "".join(text_parts)

    def completion(
        self,
        messages: "list[dict[str, str]] | str",
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                # Simple string input
                content = messages
            elif isinstance(messages, dict):
                content = messages.get("content", "")
            else:
                # List of messages - extract user content
                content_parts = []
                for msg in messages:
                    role = msg.get("role", "").lower()
                    msg_content = msg.get("content", "")
                    if msg_content:
                        content_parts.append(msg_content)
                content = "\n".join(content_parts) if content_parts else ""

            if not content:
                raise ValueError("No content to send to model")

            # Prepare generation config
            config_dict = {}
            if max_tokens is not None:
                config_dict["max_output_tokens"] = max_tokens

            # Call the API with google-genai (simplified approach)
            response = self._client.models.generate_content(
                model=f"models/{self.model}",
                contents=content,
                config=config_dict if config_dict else None,
            )

            return self._extract_text(response)

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
                  ``"gemini-2.0-flash"`` for Gemini.

    Returns:
        An :class:`OpenAIClient` or :class:`GeminiClient` instance.

    Example::

        client = get_llm_client("gemini", model="gemini-2.0-flash")
        print(client.completion("Hello!"))
    """
    provider = provider.lower().strip()

    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or "gpt-5")
    elif provider in ("gemini", "google"):
        return GeminiClient(api_key=api_key, model=model or "gemini-2.0-flash")
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported providers: 'openai', 'gemini'."
        )