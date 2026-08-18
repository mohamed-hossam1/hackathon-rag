from abc import ABC, abstractmethod
from typing import Optional


class LLMService(ABC):
    """Abstract base class for Large Language Model generation services."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generates text from an LLM given a user prompt and optional system prompt.

        Args:
            prompt: User message / query prompt text.
            system_prompt: Optional system instruction prompt.
            temperature: Sampling temperature (0.0 for deterministic output).
            max_tokens: Optional token generation limit.

        Returns:
            Generated text string response from the LLM.
        """
        pass
