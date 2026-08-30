import logging
from typing import Optional

from src.llm.llm_service import OpenAILikeLLMService

logger = logging.getLogger("medical_rag.services.title")

TITLE_SYSTEM_PROMPT = (
    "You generate short chat titles for a medical Q&A assistant. Given the user's "
    "first message, respond with ONLY a concise, descriptive title: 3-6 words, "
    "Title Case, no quotes, no trailing punctuation. Do not answer the question "
    "itself — only produce a title that summarizes what it's about."
)


class ChatTitleService:
    """Generates a concise chat title from a user's first message using the LLM."""

    def __init__(self, llm_service: Optional[OpenAILikeLLMService] = None):
        self.llm_service = llm_service or OpenAILikeLLMService()

    def generate_title(self, first_message: str, max_len: int = 60) -> str:
        first_message = (first_message or "").strip()
        if not first_message:
            return "Untitled Chat"

        try:
            raw_title = "".join(
                self.llm_service.generate_stream(
                    prompt=f"User's first message:\n{first_message}",
                    system_prompt=TITLE_SYSTEM_PROMPT,
                    temperature=0.3,
                )
            ).strip()

            title = raw_title.strip().strip('"').strip("'")
            title = " ".join(title.split())  # collapse newlines/extra whitespace

            if not title:
                return self._fallback_title(first_message, max_len)

            if len(title) > max_len:
                title = title[:max_len].rsplit(" ", 1)[0].strip() + "…"

            return title
        except Exception as exc:
            logger.warning(f"LLM title generation failed, falling back to truncation: {exc}")
            return self._fallback_title(first_message, max_len)

    @staticmethod
    def _fallback_title(message: str, max_len: int) -> str:
        text = " ".join(message.split())
        if len(text) <= max_len:
            return text
        return text[:max_len].rsplit(" ", 1)[0].strip() + "…"