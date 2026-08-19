import logging
from typing import Optional
from openai import OpenAI, OpenAIError, APIConnectionError, RateLimitError

from src.config import get_config
from src.llm.base import LLMService

logger = logging.getLogger("medical_rag.llm")


class OpenAILikeLLMService(LLMService):
    """Generic LLM service for any OpenAI-compatible provider (Groq, OpenRouter, OpenAI, vLLM, LM Studio, Ollama, etc.).

    Automatically reads configuration from .env:
      - LLM_API_KEY
      - LLM_BASE_URL
      - LLM_MODEL
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        config = get_config()
        self.api_key = api_key or config.LLM_API_KEY
        self.model_name = model_name or config.LLM_MODEL
        self.base_url = base_url or config.LLM_BASE_URL

        if not self.api_key:
            logger.warning("LLM_API_KEY is empty in .env. LLM generation calls will fail until configured.")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "missing_key"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generates a text completion using configured LL_BASE_URL and LLM_MODEL.

        Args:
            prompt: User prompt text message.
            system_prompt: Optional system prompt instructions.
            temperature: Sampling temperature (0.0 for deterministic answers).
            max_tokens: Maximum response tokens limit.

        Returns:
            Generated text string response.
        """
        if not prompt.strip():
            return ""

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        try:
            logger.info(f"Calling LLM provider at '{self.base_url}' with model '{self.model_name}' (temp={temperature})")
            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                completion_kwargs["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**completion_kwargs)
            choice = response.choices[0]
            answer_text = choice.message.content or ""
            logger.info("LLM response generated successfully")
            return answer_text.strip()
        except RateLimitError as rle:
            logger.error(f"LLM API rate limit exceeded: {rle}")
            raise RuntimeError("LLM service rate limit exceeded. Please try again in a few moments.") from rle
        except APIConnectionError as ace:
            logger.error(f"Failed to connect to LLM API endpoint '{self.base_url}': {ace}")
            raise RuntimeError("LLM service connection failed. Check your network connection or LLM_BASE_URL.") from ace
        except OpenAIError as oae:
            logger.error(f"LLM API error: {oae}")
            raise RuntimeError(f"LLM generation failed: {str(oae)}") from oae
        except Exception as err:
            logger.error(f"Unexpected error during LLM generation: {err}")
            raise RuntimeError(f"Unexpected LLM error: {str(err)}") from err
