"""
Hybrid RAG Service

Orchestrates GPT-4, Claude, and Nemotron for optimal paper improvement.
Routes tasks intelligently and combines results through ensemble scoring.
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from src.services.nemotron_llm import NemotronLLM, ChatMessage
from src.services.nemo_retriever import (
    NeMoEmbedder,
    NeMoReranker,
    NeMoRetrievalPipeline,
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task classification for routing."""
    EVALUATION = "evaluation"  # GPT-4 + Claude
    SUMMARIZATION = "summarization"  # Nemotron
    EXTRACTION = "extraction"  # Nemotron
    CLASSIFICATION = "classification"  # Nemotron
    RETRIEVAL = "retrieval"  # NeMo Retriever


class ModelProvider(Enum):
    """Available model providers."""
    GPT4 = "gpt4"
    CLAUDE = "claude"
    NEMOTRON = "nemotron"


@dataclass
class EvaluationResult:
    """Result from paper evaluation."""
    overall_quality: float
    novelty: float
    methodology: float
    clarity: float
    significance: float
    feedback: str
    provider: ModelProvider
    confidence: float
    latency_ms: float


@dataclass
class EnsembleResult:
    """Ensemble evaluation combining multiple models."""
    overall_quality: float
    novelty: float
    methodology: float
    clarity: float
    significance: float
    feedback: str
    provider_scores: Dict[str, EvaluationResult]
    ensemble_confidence: float
    total_latency_ms: float


class HybridRAGService:
    """
    Hybrid RAG Service combining GPT-4, Claude, and Nemotron.

    Task Routing:
    - Evaluation: GPT-4 + Claude (ensemble)
    - Summarization: Nemotron (fast, cost-effective)
    - Extraction: Nemotron (fast, cost-effective)
    - Retrieval: NeMo Retriever (embedding + reranking)
    """

    def __init__(
        self,
        # Nemotron services
        nemotron_llm: Optional[NemotronLLM] = None,
        nemo_embedder: Optional[NeMoEmbedder] = None,
        nemo_reranker: Optional[NeMoReranker] = None,

        # External LLM APIs
        openai_client: Optional[AsyncOpenAI] = None,
        anthropic_client: Optional[AsyncAnthropic] = None,

        # Configuration
        hybrid_mode: bool = None,
        use_gpt4_for_evaluation: bool = None,
        use_claude_for_evaluation: bool = None,
        use_nemotron_for_summarization: bool = None,
        use_nemotron_for_extraction: bool = None,

        # Ensemble weights
        ensemble_weight_gpt4: float = None,
        ensemble_weight_claude: float = None,
        ensemble_weight_nemotron: float = None,

        # Quality thresholds
        nemotron_confidence_threshold: float = None,
    ):
        """
        Initialize Hybrid RAG Service.

        Args:
            nemotron_llm: Nemotron LLM service
            nemo_embedder: NeMo embedding service
            nemo_reranker: NeMo reranking service
            openai_client: OpenAI API client
            anthropic_client: Anthropic API client
            hybrid_mode: Enable hybrid mode (default: env HYBRID_MODE)
            use_gpt4_for_evaluation: Use GPT-4 for evaluation
            use_claude_for_evaluation: Use Claude for evaluation
            use_nemotron_for_summarization: Use Nemotron for summarization
            use_nemotron_for_extraction: Use Nemotron for extraction
            ensemble_weight_gpt4: GPT-4 ensemble weight
            ensemble_weight_claude: Claude ensemble weight
            ensemble_weight_nemotron: Nemotron ensemble weight
            nemotron_confidence_threshold: Threshold for Nemotron-only tasks
        """
        # Services
        self.nemotron_llm = nemotron_llm or NemotronLLM()
        self.nemo_embedder = nemo_embedder or NeMoEmbedder()
        self.nemo_reranker = nemo_reranker or NeMoReranker()

        # External API clients
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.anthropic_client = anthropic_client or AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Configuration
        self.hybrid_mode = hybrid_mode if hybrid_mode is not None else \
            os.getenv("HYBRID_MODE", "true").lower() == "true"

        self.use_gpt4_for_evaluation = use_gpt4_for_evaluation if use_gpt4_for_evaluation is not None else \
            os.getenv("USE_GPT4_FOR_EVALUATION", "true").lower() == "true"

        self.use_claude_for_evaluation = use_claude_for_evaluation if use_claude_for_evaluation is not None else \
            os.getenv("USE_CLAUDE_FOR_EVALUATION", "true").lower() == "true"

        self.use_nemotron_for_summarization = use_nemotron_for_summarization if use_nemotron_for_summarization is not None else \
            os.getenv("USE_NEMOTRON_FOR_SUMMARIZATION", "true").lower() == "true"

        self.use_nemotron_for_extraction = use_nemotron_for_extraction if use_nemotron_for_extraction is not None else \
            os.getenv("USE_NEMOTRON_FOR_EXTRACTION", "true").lower() == "true"

        # Ensemble weights
        self.ensemble_weight_gpt4 = ensemble_weight_gpt4 if ensemble_weight_gpt4 is not None else \
            float(os.getenv("ENSEMBLE_WEIGHT_GPT4", "0.40"))

        self.ensemble_weight_claude = ensemble_weight_claude if ensemble_weight_claude is not None else \
            float(os.getenv("ENSEMBLE_WEIGHT_CLAUDE", "0.30"))

        self.ensemble_weight_nemotron = ensemble_weight_nemotron if ensemble_weight_nemotron is not None else \
            float(os.getenv("ENSEMBLE_WEIGHT_NEMOTRON", "0.30"))

        # Quality threshold
        self.nemotron_confidence_threshold = nemotron_confidence_threshold if nemotron_confidence_threshold is not None else \
            float(os.getenv("NEMOTRON_CONFIDENCE_THRESHOLD", "0.75"))

        # Validate ensemble weights
        total_weight = self.ensemble_weight_gpt4 + self.ensemble_weight_claude + self.ensemble_weight_nemotron
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Ensemble weights sum to {total_weight:.2f}, normalizing to 1.0")
            self.ensemble_weight_gpt4 /= total_weight
            self.ensemble_weight_claude /= total_weight
            self.ensemble_weight_nemotron /= total_weight

        # Retrieval pipeline
        self.retrieval_pipeline = NeMoRetrievalPipeline(
            embedder=self.nemo_embedder,
            reranker=self.nemo_reranker,
        )

        logger.info(
            f"Initialized Hybrid RAG Service: hybrid_mode={self.hybrid_mode}, "
            f"weights=(GPT4:{self.ensemble_weight_gpt4:.2f}, "
            f"Claude:{self.ensemble_weight_claude:.2f}, "
            f"Nemotron:{self.ensemble_weight_nemotron:.2f})"
        )

    async def evaluate_paper(
        self,
        paper_text: str,
        section: str = "full",
        use_ensemble: bool = True,
    ) -> EnsembleResult:
        """
        Evaluate paper quality using ensemble of models.

        Args:
            paper_text: Paper content to evaluate
            section: Section being evaluated
            use_ensemble: If True, use ensemble scoring; else use best available model

        Returns:
            EnsembleResult with scores and feedback
        """
        if not self.hybrid_mode or not use_ensemble:
            # Single model evaluation (fallback)
            if self.use_gpt4_for_evaluation:
                result = await self._evaluate_with_gpt4(paper_text, section)
                return self._convert_to_ensemble(result)
            elif self.use_claude_for_evaluation:
                result = await self._evaluate_with_claude(paper_text, section)
                return self._convert_to_ensemble(result)
            else:
                result = await self._evaluate_with_nemotron(paper_text, section)
                return self._convert_to_ensemble(result)

        # Ensemble evaluation with parallel execution
        tasks = []
        providers = []

        if self.use_gpt4_for_evaluation:
            tasks.append(self._evaluate_with_gpt4(paper_text, section))
            providers.append(ModelProvider.GPT4)

        if self.use_claude_for_evaluation:
            tasks.append(self._evaluate_with_claude(paper_text, section))
            providers.append(ModelProvider.CLAUDE)

        # Optionally include Nemotron in ensemble
        if self.ensemble_weight_nemotron > 0:
            tasks.append(self._evaluate_with_nemotron(paper_text, section))
            providers.append(ModelProvider.NEMOTRON)

        # Execute evaluations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        provider_scores = {}
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                logger.error(f"{provider.value} evaluation failed: {result}")
                continue
            provider_scores[provider.value] = result

        if not provider_scores:
            raise RuntimeError("All evaluation models failed")

        # Compute ensemble scores
        return self._compute_ensemble(provider_scores)

    async def _evaluate_with_gpt4(
        self, paper_text: str, section: str
    ) -> EvaluationResult:
        """Evaluate paper using GPT-4."""
        import time
        start_time = time.time()

        prompt = self._build_evaluation_prompt(paper_text, section)

        try:
            response = await self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are an expert academic reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            content = response.choices[0].message.content
            scores = self._parse_evaluation_response(content)

            return EvaluationResult(
                overall_quality=scores["overall_quality"],
                novelty=scores["novelty"],
                methodology=scores["methodology"],
                clarity=scores["clarity"],
                significance=scores["significance"],
                feedback=scores["feedback"],
                provider=ModelProvider.GPT4,
                confidence=0.9,  # GPT-4 high confidence
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"GPT-4 evaluation failed: {e}")
            raise

    async def _evaluate_with_claude(
        self, paper_text: str, section: str
    ) -> EvaluationResult:
        """Evaluate paper using Claude."""
        import time
        start_time = time.time()

        prompt = self._build_evaluation_prompt(paper_text, section)

        try:
            response = await self.anthropic_client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4"),
                max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.3")),
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            content = response.content[0].text
            scores = self._parse_evaluation_response(content)

            return EvaluationResult(
                overall_quality=scores["overall_quality"],
                novelty=scores["novelty"],
                methodology=scores["methodology"],
                clarity=scores["clarity"],
                significance=scores["significance"],
                feedback=scores["feedback"],
                provider=ModelProvider.CLAUDE,
                confidence=0.9,  # Claude high confidence
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Claude evaluation failed: {e}")
            raise

    async def _evaluate_with_nemotron(
        self, paper_text: str, section: str
    ) -> EvaluationResult:
        """Evaluate paper using Nemotron."""
        prompt = self._build_evaluation_prompt(paper_text, section)

        try:
            result = await self.nemotron_llm.complete(prompt)

            # Parse response
            scores = self._parse_evaluation_response(result.text)

            return EvaluationResult(
                overall_quality=scores["overall_quality"],
                novelty=scores["novelty"],
                methodology=scores["methodology"],
                clarity=scores["clarity"],
                significance=scores["significance"],
                feedback=scores["feedback"],
                provider=ModelProvider.NEMOTRON,
                confidence=0.7,  # Nemotron moderate confidence
                latency_ms=result.latency_ms,
            )

        except Exception as e:
            logger.error(f"Nemotron evaluation failed: {e}")
            raise

    def _build_evaluation_prompt(self, paper_text: str, section: str) -> str:
        """Build evaluation prompt for paper assessment."""
        return f"""Evaluate the following {section} section of a scientific paper on a scale of 1-10:

Paper Section:
{paper_text}

Provide scores for:
1. Overall Quality (1-10)
2. Novelty (1-10)
3. Methodology (1-10)
4. Clarity (1-10)
5. Significance (1-10)

Format your response as:
OVERALL_QUALITY: X.X
NOVELTY: X.X
METHODOLOGY: X.X
CLARITY: X.X
SIGNIFICANCE: X.X
FEEDBACK: [Your detailed feedback here]
"""

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse evaluation response into structured scores."""
        scores = {
            "overall_quality": 7.0,
            "novelty": 7.0,
            "methodology": 7.0,
            "clarity": 7.0,
            "significance": 7.0,
            "feedback": "",
        }

        lines = response.split("\n")
        for line in lines:
            line = line.strip()

            if line.startswith("OVERALL_QUALITY:"):
                try:
                    scores["overall_quality"] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("NOVELTY:"):
                try:
                    scores["novelty"] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("METHODOLOGY:"):
                try:
                    scores["methodology"] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("CLARITY:"):
                try:
                    scores["clarity"] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("SIGNIFICANCE:"):
                try:
                    scores["significance"] = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("FEEDBACK:"):
                scores["feedback"] = line.split(":", 1)[1].strip()

        return scores

    def _compute_ensemble(
        self, provider_scores: Dict[str, EvaluationResult]
    ) -> EnsembleResult:
        """Compute ensemble scores from multiple providers."""
        # Normalize weights based on available providers
        available_weights = {}
        total_weight = 0.0

        for provider_key, result in provider_scores.items():
            if provider_key == ModelProvider.GPT4.value:
                available_weights[provider_key] = self.ensemble_weight_gpt4
                total_weight += self.ensemble_weight_gpt4
            elif provider_key == ModelProvider.CLAUDE.value:
                available_weights[provider_key] = self.ensemble_weight_claude
                total_weight += self.ensemble_weight_claude
            elif provider_key == ModelProvider.NEMOTRON.value:
                available_weights[provider_key] = self.ensemble_weight_nemotron
                total_weight += self.ensemble_weight_nemotron

        # Normalize weights
        if total_weight > 0:
            for key in available_weights:
                available_weights[key] /= total_weight

        # Compute weighted ensemble
        ensemble_overall = 0.0
        ensemble_novelty = 0.0
        ensemble_methodology = 0.0
        ensemble_clarity = 0.0
        ensemble_significance = 0.0
        ensemble_confidence = 0.0
        total_latency = 0.0

        feedbacks = []

        for provider_key, result in provider_scores.items():
            weight = available_weights.get(provider_key, 0.0)

            ensemble_overall += result.overall_quality * weight
            ensemble_novelty += result.novelty * weight
            ensemble_methodology += result.methodology * weight
            ensemble_clarity += result.clarity * weight
            ensemble_significance += result.significance * weight
            ensemble_confidence += result.confidence * weight
            total_latency += result.latency_ms

            feedbacks.append(f"[{result.provider.value}] {result.feedback}")

        combined_feedback = "\n\n".join(feedbacks)

        return EnsembleResult(
            overall_quality=ensemble_overall,
            novelty=ensemble_novelty,
            methodology=ensemble_methodology,
            clarity=ensemble_clarity,
            significance=ensemble_significance,
            feedback=combined_feedback,
            provider_scores=provider_scores,
            ensemble_confidence=ensemble_confidence,
            total_latency_ms=total_latency,
        )

    def _convert_to_ensemble(self, result: EvaluationResult) -> EnsembleResult:
        """Convert single evaluation result to ensemble format."""
        return EnsembleResult(
            overall_quality=result.overall_quality,
            novelty=result.novelty,
            methodology=result.methodology,
            clarity=result.clarity,
            significance=result.significance,
            feedback=result.feedback,
            provider_scores={result.provider.value: result},
            ensemble_confidence=result.confidence,
            total_latency_ms=result.latency_ms,
        )

    async def summarize_paper(
        self, paper_text: str, max_length: int = 200, style: str = "concise"
    ) -> str:
        """
        Summarize paper using Nemotron (fast, cost-effective).

        Args:
            paper_text: Paper content to summarize
            max_length: Maximum summary length in words
            style: Summary style ('concise', 'detailed', 'bullet_points')

        Returns:
            Summary text
        """
        if self.use_nemotron_for_summarization:
            return await self.nemotron_llm.summarize(
                text=paper_text,
                max_length=max_length,
                style=style,
            )
        else:
            # Fallback to GPT-4/Claude
            logger.warning("Nemotron summarization disabled, using GPT-4 fallback")
            prompt = f"Summarize this paper in {max_length} words ({style} style):\n\n{paper_text}"
            response = await self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length * 2,
            )
            return response.choices[0].message.content

    async def extract_information(
        self, paper_text: str, fields: List[str]
    ) -> Dict[str, str]:
        """
        Extract specific information from paper using Nemotron.

        Args:
            paper_text: Paper content
            fields: Fields to extract (e.g., ['methodology', 'results', 'limitations'])

        Returns:
            Dictionary mapping field names to extracted values
        """
        if self.use_nemotron_for_extraction:
            return await self.nemotron_llm.extract_information(
                text=paper_text,
                fields=fields,
            )
        else:
            # Fallback to GPT-4/Claude
            logger.warning("Nemotron extraction disabled, using GPT-4 fallback")
            prompt = f"Extract the following fields from this paper: {', '.join(fields)}\n\nPaper:\n{paper_text}\n\nProvide as JSON."
            response = await self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )

            import json
            try:
                return json.loads(response.choices[0].message.content)
            except:
                return {field: "" for field in fields}

    async def retrieve_similar_papers(
        self,
        query: str,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve and rerank similar papers using NeMo Retriever.

        Args:
            query: Search query
            top_k_retrieve: Number of candidates to retrieve
            top_k_rerank: Number of final results after reranking

        Returns:
            Tuple of (reranked_documents, relevance_scores)
        """
        return await self.retrieval_pipeline.retrieve_and_rerank(
            query=query,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
        )

    async def close(self):
        """Close all service connections."""
        await self.nemotron_llm.close()
        await self.nemo_embedder.close()
        await self.nemo_reranker.close()
        await self.openai_client.close()
        await self.anthropic_client.close()


# Convenience function for quick setup
async def create_hybrid_rag_service() -> HybridRAGService:
    """Create a Hybrid RAG service with environment configuration."""
    return HybridRAGService()
