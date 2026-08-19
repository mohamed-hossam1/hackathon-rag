import json
import logging
import re
from typing import List, Optional

from src.llm.base import LLMService
from src.llm.llm_service import OpenAILikeLLMService
from src.models.chunk import Chunk
from src.models.evaluation import EvaluationResult

logger = logging.getLogger("medical_rag.evaluation.llm_judge")

LLM_BATCH_JUDGE_SYSTEM_PROMPT = """You are an expert medical relevance auditor.
Your job is to evaluate a batch of retrieved chunks from a specific retrieval strategy against a user query.

For each chunk in the input array, judge if the chunk contains information relevant to answering the query.

Respond ONLY with a valid JSON array of objects with the exact schema:
[
  {
    "chunk_id": "exact chunk_id from input",
    "relevant": true or false,
    "reason": "Concise 1-sentence evidence-based explanation"
  }
]

Do not include any extra text, markdown wrappers, or explanations outside the raw JSON array."""


class LLMJudge:
    """LLM-as-a-Judge for evaluating retrieved chunk relevance in method-isolated batches."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or OpenAILikeLLMService()

    def judge_batch_by_method(
        self,
        query: str,
        chunks: List[Chunk],
        retrieval_method: str
    ) -> List[EvaluationResult]:
        """Evaluates a batch of candidate chunks from ONE retrieval strategy in a single LLM call.

        Args:
            query: User search query text.
            chunks: List of Chunk objects retrieved by this specific method.
            retrieval_method: Strategy name ('semantic', 'recursive', 'bm25', 'reranker').

        Returns:
            List of EvaluationResult objects corresponding to each input chunk.
        """
        if not chunks:
            return []

        logger.info(
            f"Batch evaluating {len(chunks)} chunks for retrieval_method='{retrieval_method}'"
        )

        batch_input = [
            {"chunk_id": chunk.chunk_id, "chunk_text": chunk.text}
            for chunk in chunks
        ]

        user_prompt = (
            f"Retrieval Method Strategy: {retrieval_method}\n"
            f"Query:\n\"{query}\"\n\n"
            f"Retrieved Chunks:\n{json.dumps(batch_input, indent=2)}"
        )

        results: List[EvaluationResult] = []

        try:
            raw_response = self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=LLM_BATCH_JUDGE_SYSTEM_PROMPT,
                temperature=0.0
            )

            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE)
            batch_output = json.loads(clean_json)

            output_by_id = {
                item.get("chunk_id"): item
                for item in batch_output
                if isinstance(item, dict)
            }

            for chunk in chunks:
                out = output_by_id.get(chunk.chunk_id, {})
                relevant = bool(out.get("relevant", False))
                reason = str(out.get("reason", "Batch LLM relevance judgment."))

                results.append(
                    EvaluationResult(
                        query=query,
                        chunk_id=chunk.chunk_id,
                        chunk_text=chunk.text,
                        retrieval_method=retrieval_method,
                        relevant=relevant,
                        reason=reason
                    )
                )

        except Exception as err:
            logger.error(
                f"Batch LLM judge failed for retrieval_method='{retrieval_method}': {err}"
            )
            # Fallback heuristic: word overlap check per chunk
            query_words = set(re.findall(r"\w+", query.lower()))
            for chunk in chunks:
                chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
                overlap = len(query_words.intersection(chunk_words)) / max(1, len(query_words))
                relevant = overlap >= 0.4

                results.append(
                    EvaluationResult(
                        query=query,
                        chunk_id=chunk.chunk_id,
                        chunk_text=chunk.text,
                        retrieval_method=retrieval_method,
                        relevant=relevant,
                        reason=f"Fallback heuristic check (word overlap={overlap:.2f})."
                    )
                )

        return results
