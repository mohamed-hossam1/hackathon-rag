import json
import logging
from typing import Optional, Tuple
from src.llm.base import LLMService
from src.llm.llm_service import OpenAILikeLLMService

logger = logging.getLogger("medical_rag.services.memory_detector")

DETECTION_SYSTEM_PROMPT = """You are a medical personal information analyzer.
Your job is to analyze a user's question/input to detect if the user is sharing personal medical history or user-specific facts (such as age, gender, chronic diseases like hypertension/diabetes, allergies, current medications, past surgeries, pregnancy status, or lifestyle factors).

IMPORTANT:
- If the user is asking a general medical question (e.g., "what is hypertension?", "ماهو علاج الضغط", "how to treat flu"), return has_personal_info: false.
- If the user is sharing their own medical condition or profile (e.g., "أنا مريض ضغط وعندي حساسية من البنسلين", "I am 45 years old and take Lisinopril", "أنا حامل في الشهر الثالث"), return has_personal_info: true.

Respond strictly in JSON format with no additional text or markdown formatting:
{
  "has_personal_info": true | false,
  "extracted_info": "concise medical summary of the user's personal facts in Arabic or English",
  "memory_prompt": "friendly popup message asking if user wants to save this info for future chat sessions"
}"""


class PersonalMemoryDetector:
    """Detects and extracts personal medical history/data from user queries."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or OpenAILikeLLMService()

    def detect(self, query_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Analyzes query text for personal medical context.

        Returns:
            Tuple of (has_personal_info, extracted_info, memory_prompt)
        """
        if not query_text or len(query_text.strip()) < 5:
            return False, None, None

        # Basic keyword pre-filter to avoid unnecessary LLM calls for obvious general queries
        personal_keywords = [
            "أنا", "عندي", "بعاني", "مصاب", "عمري", "حامل", "باخد", "علاجي", "دوايا", "حساسية",
            "i am", "i have", "my age", "allergic", "taking", "diagnosed", "my doctor", "my pressure", "my blood"
        ]
        query_lower = query_text.lower()
        has_keyword = any(kw in query_lower for kw in personal_keywords)

        # Exclude hypothetical scenarios, third-person questions, or general case studies
        hypothetical_keywords = [
            "imagine", "suppose", "scenario", "a patient", "this patient", "for a patient", "if a patient", "case study",
            "افترض", "تخيل", "سيناريو", "مريض", "للمريض", "حالة مرضية"
        ]
        if any(hk in query_lower for hk in hypothetical_keywords):
            return False, None, None

        if not has_keyword:
            return False, None, None

        try:
            prompt = f"User Input:\n{query_text}"
            llm_response = "".join(
                self.llm_service.generate_stream(
                    prompt=prompt,
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    temperature=0.0
                )
            )

            # Strip markdown codeblocks if present
            cleaned_json = llm_response.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            cleaned_json = cleaned_json.strip()

            parsed = json.loads(cleaned_json)
            has_info = bool(parsed.get("has_personal_info", False))
            extracted_info = parsed.get("extracted_info") if has_info else None
            memory_prompt = parsed.get("memory_prompt") if has_info else None

            if has_info and not memory_prompt:
                memory_prompt = f"هل ترغب في حفظ هذه المعلومة الطبية ({extracted_info}) لإتاحتها في جلسات الشات القادمة؟"

            return has_info, extracted_info, memory_prompt

        except Exception as err:
            logger.error(f"Error in PersonalMemoryDetector: {err}")
            return False, None, None
