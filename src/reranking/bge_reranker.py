import logging
from typing import Any, List, Optional

import torch
from sentence_transformers import CrossEncoder

from src.config import get_config
from src.models.retrieval import RerankResult, RetrievalResult

logger = logging.getLogger("medical_rag.reranking")


class BGEReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-base model."""

    def __init__(self, model_name: Optional[str] = None):
        config = get_config()
        self.model_name = model_name or config.RERANKER_MODEL
        self.top_k_default = config.RERANKER_TOP_K
        self._model: Optional[CrossEncoder] = None
        self._device: Optional[str] = None

    def _get_device(self) -> str:
        """Return a device supported by the current PyTorch installation."""

        if not torch.cuda.is_available():
            logger.info("CUDA is not available. Using CPU for reranker.")
            return "cpu"

        gpu_name = torch.cuda.get_device_name(0)
        gpu_capability = torch.cuda.get_device_capability(0)
        gpu_arch = f"sm_{gpu_capability[0]}{gpu_capability[1]}"

        supported_architectures = torch.cuda.get_arch_list()

        if gpu_arch not in supported_architectures:
            logger.warning(
                "GPU %s (%s) is not supported by the installed PyTorch build. "
                "Supported architectures: %s. Falling back to CPU.",
                gpu_name,
                gpu_arch,
                supported_architectures,
            )
            return "cpu"

        logger.info(
            "GPU %s (%s) is supported. Using CUDA for reranker.",
            gpu_name,
            gpu_arch,
        )
        return "cuda"

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(
                f"Loading CrossEncoder reranker model lazily: {self.model_name}"
            )

            self._device = self._get_device()

            try:
                self._model = CrossEncoder(
                    self.model_name,
                    device=self._device,
                )

                logger.info(
                    "Reranker loaded successfully on %s",
                    self._device,
                )

            except Exception as e:  # pragma: no cover
                if self._device == "cuda":
                    logger.warning(
                        "Failed to load CrossEncoder on CUDA. "
                        "Falling back to CPU: %s",
                        str(e),
                    )

                    self._device = "cpu"

                    self._model = CrossEncoder(
                        self.model_name,
                        device="cpu",
                    )

                    logger.info("Reranker loaded on CPU")
                else:
                    raise

        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Reranks candidate retrieval results against a query string.

        Args:
            query: User search query text.
            candidates: List of RetrievalResult objects to rerank.
            top_k: Optional top-K override. Defaults to config.RERANKER_TOP_K.

        Returns:
            List of RerankResult objects sorted by rerank_score descending.
        """
        if not query.strip() or not candidates:
            return []

        limit = top_k if top_k is not None else self.top_k_default

        # Construct query-chunk text pairs for CrossEncoder scoring
        pairs: List[Any] = [
            [query, candidate.chunk.text]
            for candidate in candidates
        ]

        logger.info(
            f"Reranking {len(pairs)} query-chunk pairs "
            f"with model {self.model_name}"
        )

        try:
            scores = self.model.predict(pairs)  # type: ignore

        except Exception as e:  # pragma: no cover
            # CUDA can successfully load the model but fail during
            # the first actual inference operation. This is particularly
            # relevant for older GPUs that are visible to CUDA but are not
            # supported by the installed PyTorch build.
            if self._device == "cuda":
                logger.warning(
                    "CUDA inference failed for reranker. "
                    "Falling back to CPU and retrying: %s",
                    str(e),
                )

                self._device = "cpu"

                self._model = CrossEncoder(
                    self.model_name,
                    device="cpu",
                )

                logger.info("Retrying reranker inference on CPU")

                scores = self._model.predict(pairs)  # type: ignore
            else:
                raise

        rerank_results: List[RerankResult] = []

        for candidate, score in zip(candidates, scores):
            rerank_results.append(
                RerankResult(
                    chunk=candidate.chunk,
                    rerank_score=float(score),
                    original_retrieval_method=candidate.retrieval_method,
                    original_score=candidate.score
                )
            )

        # Sort by rerank_score descending
        rerank_results.sort(
            key=lambda r: r.rerank_score,
            reverse=True
        )

        final_results = rerank_results[:limit]

        logger.info(
            f"Reranker returned top {len(final_results)} reranked results"
        )

        return final_results
