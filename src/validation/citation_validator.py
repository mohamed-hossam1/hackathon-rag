import json
import logging
import re
from typing import Dict, List, Optional

from src.llm.base import LLMService
from src.llm.llm_service import OpenAILikeLLMService
from src.models.chunk import Chunk
from src.models.citation import Citation, CitationValidationResult
from src.models.response import ConfidenceLabel

logger = logging.getLogger("medical_rag.validation")

HIGH_RISK_PATTERNS = [
    r"\b\d+(\.\d+)?\s*(mg|g|ml|mcg|iu|unit|units|dose|doses|mg/kg|mg/dl|mmol/l)\b",
    r"\b(recommend|recommendation|recommended|prescribe|prescribed|first-line|second-line|treatment|therapy|dosage)\b",
    r"\b(contraindicat|interaction|interact|adverse|toxicity|toxic|poisoning|overdose|side effect)\b",
    r"\b(mortality|death|fatal|fatality|survival|curable|cure)\b",
    r"\b\d+(\.\d+)?\s*(%|percent|percentage)\b",
    r"\b(pediatric|infant|children|child|pregnant|pregnancy|geriatric)\b",
]

BATCH_VALIDATION_SYSTEM_PROMPT = """You are a conservative medical evidence auditor.
Your job is to strictly verify whether source chunks support claims made in a medical response.

For each item in the input array, evaluate the relationship between the claim and the source chunk:
- "supports": The source chunk text explicitly and logically proves the claim.
- "not_supported": The source chunk text does not contain sufficient evidence to prove the claim.
- "contradicts": The source chunk text directly conflicts with or contradicts the claim.

Respond ONLY with a valid JSON array matching this exact format:
[
  {
    "id": 0,
    "status": "supports" | "not_supported" | "contradicts",
    "reason": "Concise 1-sentence evidence-based explanation"
  }
]

Do not output any markdown wrappers, conversational text, or formatting outside the raw JSON array."""


def classify_claim_risk(claim: str) -> str:
    """Classifies a claim as 'high' or 'standard' risk based on medical sensitivity patterns."""
    claim_lower = claim.lower()
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, claim_lower):
            return "high"
    return "standard"


class CitationValidator:
    """Batch-oriented, safety-first citation validator with risk weighting and contradiction detection."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or OpenAILikeLLMService()

    def validate_citations(
        self,
        citations: List[Citation],
        chunks_map: Dict[str, Chunk]
    ) -> List[CitationValidationResult]:
        """Validates a list of citations in a single batched LLM request with deterministic pre-checks.

        Args:
            citations: List of Citation references from answer generation.
            chunks_map: Dictionary mapping chunk_id to Chunk domain models.

        Returns:
            List of CitationValidationResult objects with risk classification and support status.
        """
        if not citations:
            return []

        results_map: Dict[int, CitationValidationResult] = {}
        batch_items_to_validate = []

        # Step 1 & 2: Deterministic Pre-checks & Risk Classification
        for idx, citation in enumerate(citations):
            risk_level = classify_claim_risk(citation.claim)
            chunk = chunks_map.get(citation.chunk_id)

            # Deterministic Check 1: Chunk existence in retrieved context
            if not chunk:
                logger.warning(f"Citation #{idx} chunk_id='{citation.chunk_id}' not found in retrieved context")
                results_map[idx] = CitationValidationResult(
                    claim=citation.claim,
                    chunk_id=citation.chunk_id,
                    chunk_text="",
                    supported=False,
                    status="not_supported",
                    risk_level=risk_level,
                    reason="Referenced chunk ID not found in retrieved context."
                )
                continue

            # Deterministic Check 2: Non-empty chunk text
            if not chunk.text.strip():
                logger.warning(f"Citation #{idx} chunk_id='{citation.chunk_id}' has empty text")
                results_map[idx] = CitationValidationResult(
                    claim=citation.claim,
                    chunk_id=citation.chunk_id,
                    chunk_text="",
                    supported=False,
                    status="not_supported",
                    risk_level=risk_level,
                    reason="Referenced chunk contains no text content."
                )
                continue

            # Add to batch for LLM semantic validation
            batch_items_to_validate.append({
                "batch_index": idx,
                "id": len(batch_items_to_validate),
                "claim": citation.claim,
                "chunk_id": citation.chunk_id,
                "chunk_text": chunk.text,
                "risk_level": risk_level
            })

        # Step 3: Batch LLM Validation
        if batch_items_to_validate:
            self._process_llm_validation(batch_items_to_validate, results_map)

        # Assemble final results in original citation order
        final_results = [results_map[i] for i in range(len(citations))]
        supported_count = sum(1 for r in final_results if r.supported)
        contradicted_count = sum(1 for r in final_results if r.status == "contradicts")
        logger.info(
            f"Citation validation complete: total={len(citations)}, "
            f"supported={supported_count}, contradicted={contradicted_count}"
        )
        return final_results

    def _process_llm_validation(
        self,
        batch_items: List[dict],
        results_map: Dict[int, CitationValidationResult]
    ) -> None:
        """Sends a batched validation request to the LLM and populates results_map."""
        llm_payload = [
            {
                "id": item["id"],
                "claim": item["claim"],
                "source_chunk": item["chunk_text"]
            }
            for item in batch_items
        ]

        user_prompt = f"Validate the following claim-chunk pairs:\n{json.dumps(llm_payload, indent=2)}"

        try:
            raw_response = self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=BATCH_VALIDATION_SYSTEM_PROMPT,
                temperature=0.0
            )

            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE)
            batch_outputs = json.loads(clean_json)

            output_by_id = {item.get("id"): item for item in batch_outputs if isinstance(item, dict)}

            for item in batch_items:
                batch_idx = item["batch_index"]
                out = output_by_id.get(item["id"], {})

                status = str(out.get("status", "not_supported")).lower()
                if status not in ("supports", "not_supported", "contradicts"):
                    status = "not_supported"

                supported = (status == "supports")
                reason = str(out.get("reason", "Validation completed by LLM auditor."))

                results_map[batch_idx] = CitationValidationResult(
                    claim=item["claim"],
                    chunk_id=item["chunk_id"],
                    chunk_text=item["chunk_text"],
                    supported=supported,
                    status=status,
                    risk_level=item["risk_level"],
                    reason=reason
                )

        except Exception as err:
            logger.error(f"Batch LLM citation validation failed or produced invalid JSON: {err}")
            # Safety rule: Do NOT use lexical overlap to declare unsupported claims as supported upon LLM failure!
            for item in batch_items:
                batch_idx = item["batch_index"]
                results_map[batch_idx] = CitationValidationResult(
                    claim=item["claim"],
                    chunk_id=item["chunk_id"],
                    chunk_text=item["chunk_text"],
                    supported=False,
                    status="not_supported",
                    risk_level=item["risk_level"],
                    reason="LLM validation failed; claim treated conservatively as unverified."
                )

    @staticmethod
    def compute_evidence_score(validations: List[CitationValidationResult]) -> float:
        """Computes a risk-weighted evidence score (0.0 to 1.0) incorporating claim risk levels and contradiction penalties."""
        if not validations:
            return 0.0

        total_weight = 0.0
        earned_score = 0.0

        for val in validations:
            weight = 2.0 if val.risk_level == "high" else 1.0
            total_weight += weight

            if val.status == "supports":
                earned_score += weight
            elif val.status == "contradicts":
                # Contradiction incurs a severe penalty on earned score
                earned_score -= (weight * 1.5)
            # 'not_supported' yields 0.0 added score

        if total_weight <= 0.0:
            return 0.0

        ratio = earned_score / total_weight
        bounded_score = max(0.0, min(1.0, ratio))
        return round(bounded_score, 2)

    @staticmethod
    def get_confidence_label(
        evidence_score: float,
        validations: Optional[List[CitationValidationResult]] = None
    ) -> ConfidenceLabel:
        """Determines conservative confidence label considering evidence score, high-risk failures, and contradictions."""
        if validations:
            # Rule 1: Any high-risk claim contradicted by evidence drops confidence to INSUFFICIENT
            has_high_risk_contradiction = any(v.risk_level == "high" and v.status == "contradicts" for v in validations)
            if has_high_risk_contradiction:
                logger.warning("Confidence set to INSUFFICIENT due to high-risk claim contradiction")
                return ConfidenceLabel.INSUFFICIENT

            # Rule 2: Any standard claim contradicted caps confidence at LOW
            has_any_contradiction = any(v.status == "contradicts" for v in validations)
            if has_any_contradiction and evidence_score < 0.7:
                return ConfidenceLabel.LOW

            # Rule 3: Any unsupported high-risk claim caps confidence at MEDIUM (cannot be HIGH)
            has_unsupported_high_risk = any(v.risk_level == "high" and not v.supported for v in validations)
            if has_unsupported_high_risk and evidence_score >= 0.7:
                logger.info("Capping confidence to MEDIUM due to unsupported high-risk claim")
                return ConfidenceLabel.MEDIUM

        if evidence_score >= 0.85:
            return ConfidenceLabel.HIGH
        elif evidence_score >= 0.50:
            return ConfidenceLabel.MEDIUM
        elif evidence_score >= 0.25:
            return ConfidenceLabel.LOW
        else:
            return ConfidenceLabel.INSUFFICIENT
