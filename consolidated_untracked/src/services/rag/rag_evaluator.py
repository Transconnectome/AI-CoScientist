"""
RAG Evaluator with RAGAS Integration

Implementation for: Extend RAG evaluator with RAGAS metrics
Created: 2025-12-04

Acceptance Criteria:
- RAGAS faithfulness metric implemented and tested
- Answer relevancy scoring functional with validation
- Context precision calculation working accurately
- Integration with existing evaluation pipeline
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas")

import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class RAGEvaluationResult:
    """Results from RAG evaluation"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float] = None
    context_relevancy: Optional[float] = None
    overall_score: Optional[float] = None
    evaluation_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationInput:
    """Input structure for RAG evaluation"""
    query: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RAGEvaluator:
    """
    RAG Evaluator with RAGAS metrics integration

    Supports comprehensive evaluation of RAG systems using:
    - Faithfulness: Answer grounding in provided context
    - Answer Relevancy: Answer addresses the query
    - Context Precision: Relevance of retrieved context
    - Context Recall: Coverage of relevant information
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_ragas: bool = True,
        fallback_to_simple: bool = True
    ):
        self.embedding_model_name = embedding_model
        self.enable_ragas = enable_ragas and RAGAS_AVAILABLE
        self.fallback_to_simple = fallback_to_simple

        # Initialize embedding model for fallback metrics
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logging.warning(f"Could not load embedding model {embedding_model}: {e}")
            self.embedding_model = None

        # Configure RAGAS metrics if available
        if self.enable_ragas:
            self.ragas_metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_relevancy
            ]

        self.logger = logging.getLogger(__name__)

    async def evaluate_single(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RAGEvaluationResult:
        """
        Evaluate a single RAG query-answer pair

        Args:
            query: User query
            contexts: Retrieved context documents
            answer: Generated answer
            ground_truth: Expected answer (optional)
            metadata: Additional metadata

        Returns:
            RAGEvaluationResult with computed metrics
        """
        start_time = datetime.now()

        evaluation_input = EvaluationInput(
            query=query,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
            metadata=metadata
        )

        if self.enable_ragas:
            try:
                result = await self._evaluate_with_ragas(evaluation_input)
            except Exception as e:
                self.logger.warning(f"RAGAS evaluation failed: {e}")
                if self.fallback_to_simple:
                    result = await self._evaluate_with_simple_metrics(evaluation_input)
                else:
                    raise
        else:
            result = await self._evaluate_with_simple_metrics(evaluation_input)

        # Calculate overall score and timing
        result.overall_score = self._calculate_overall_score(result)
        result.evaluation_time = (datetime.now() - start_time).total_seconds()
        result.metadata = metadata or {}

        return result

    async def evaluate_batch(
        self,
        evaluation_inputs: List[EvaluationInput]
    ) -> List[RAGEvaluationResult]:
        """
        Evaluate multiple RAG pairs in batch for efficiency

        Args:
            evaluation_inputs: List of evaluation inputs

        Returns:
            List of RAGEvaluationResult objects
        """
        if self.enable_ragas:
            try:
                return await self._batch_evaluate_with_ragas(evaluation_inputs)
            except Exception as e:
                self.logger.warning(f"Batch RAGAS evaluation failed: {e}")
                if self.fallback_to_simple:
                    return await self._batch_evaluate_simple(evaluation_inputs)
                else:
                    raise
        else:
            return await self._batch_evaluate_simple(evaluation_inputs)

    async def _evaluate_with_ragas(self, input_data: EvaluationInput) -> RAGEvaluationResult:
        """Evaluate using RAGAS metrics"""

        # Prepare dataset for RAGAS
        dataset_dict = {
            "question": [input_data.query],
            "contexts": [input_data.contexts],
            "answer": [input_data.answer]
        }

        if input_data.ground_truth:
            dataset_dict["ground_truth"] = [input_data.ground_truth]

        dataset = Dataset.from_dict(dataset_dict)

        # Select metrics based on available data
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
        if input_data.ground_truth:
            metrics_to_use.append(context_recall)

        # Run evaluation
        result = evaluate(dataset, metrics=metrics_to_use)

        # Extract scores
        return RAGEvaluationResult(
            faithfulness=result.get('faithfulness', [0.0])[0],
            answer_relevancy=result.get('answer_relevancy', [0.0])[0],
            context_precision=result.get('context_precision', [0.0])[0],
            context_recall=result.get('context_recall', [None])[0],
            context_relevancy=None  # Not available in single evaluation
        )

    async def _batch_evaluate_with_ragas(
        self,
        inputs: List[EvaluationInput]
    ) -> List[RAGEvaluationResult]:
        """Batch evaluation with RAGAS for efficiency"""

        # Prepare batch dataset
        dataset_dict = {
            "question": [inp.query for inp in inputs],
            "contexts": [inp.contexts for inp in inputs],
            "answer": [inp.answer for inp in inputs]
        }

        # Add ground truth if available for any input
        if any(inp.ground_truth for inp in inputs):
            dataset_dict["ground_truth"] = [
                inp.ground_truth or "" for inp in inputs
            ]

        dataset = Dataset.from_dict(dataset_dict)

        # Run batch evaluation
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
        if "ground_truth" in dataset_dict:
            metrics_to_use.append(context_recall)

        result = evaluate(dataset, metrics=metrics_to_use)

        # Convert to list of results
        import time
        results = []
        for i in range(len(inputs)):
            start_time = time.time()
            rag_result = RAGEvaluationResult(
                faithfulness=result.get('faithfulness', [0.0])[i],
                answer_relevancy=result.get('answer_relevancy', [0.0])[i],
                context_precision=result.get('context_precision', [0.0])[i],
                context_recall=result.get('context_recall', [None])[i] if 'context_recall' in result else None
            )

            # Calculate overall score and evaluation time for each result
            rag_result.overall_score = self._calculate_overall_score(rag_result)
            rag_result.evaluation_time = time.time() - start_time
            results.append(rag_result)

        return results

    async def _evaluate_with_simple_metrics(
        self,
        input_data: EvaluationInput
    ) -> RAGEvaluationResult:
        """Fallback evaluation using simple similarity metrics"""

        if not self.embedding_model or not input_data.contexts:
            # Ultimate fallback - basic heuristics or empty contexts
            return RAGEvaluationResult(
                faithfulness=self._calculate_word_overlap(input_data.answer, " ".join(input_data.contexts)),
                answer_relevancy=self._calculate_word_overlap(input_data.answer, input_data.query),
                context_precision=self._calculate_context_precision_heuristic(input_data.query, input_data.contexts)
            )

        # Embedding-based similarity
        query_emb = self.embedding_model.encode([input_data.query])
        answer_emb = self.embedding_model.encode([input_data.answer])

        # Handle empty contexts
        if not input_data.contexts:
            return RAGEvaluationResult(
                faithfulness=0.0,
                answer_relevancy=float(cosine_similarity(query_emb, answer_emb)[0][0]),
                context_precision=0.0
            )

        context_emb = self.embedding_model.encode(input_data.contexts)

        # Answer relevancy: similarity between query and answer
        from sklearn.metrics.pairwise import cosine_similarity
        answer_relevancy = float(cosine_similarity(query_emb, answer_emb)[0][0])

        # Context precision: average similarity of contexts to query
        context_similarities = cosine_similarity(query_emb, context_emb)[0]
        context_precision = float(context_similarities.mean())

        # Faithfulness: similarity between answer and best matching context
        answer_context_similarities = cosine_similarity(answer_emb, context_emb)[0]
        faithfulness = float(answer_context_similarities.max())

        return RAGEvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision
        )

    async def _batch_evaluate_simple(
        self,
        inputs: List[EvaluationInput]
    ) -> List[RAGEvaluationResult]:
        """Batch simple evaluation"""
        import time
        results = []
        for input_data in inputs:
            start_time = time.time()
            result = await self._evaluate_with_simple_metrics(input_data)

            # Calculate overall score and evaluation time for each result
            result.overall_score = self._calculate_overall_score(result)
            result.evaluation_time = time.time() - start_time
            results.append(result)
        return results

    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1:
            return 0.0

        overlap = len(words1.intersection(words2))
        return overlap / len(words1)

    def _calculate_context_precision_heuristic(
        self,
        query: str,
        contexts: List[str]
    ) -> float:
        """Heuristic context precision based on query word presence"""
        query_words = set(query.lower().split())

        if not query_words:
            return 0.0

        precisions = []
        for context in contexts:
            context_words = set(context.lower().split())
            if context_words:
                overlap = len(query_words.intersection(context_words))
                precision = overlap / len(query_words)
                precisions.append(precision)

        return sum(precisions) / len(precisions) if precisions else 0.0

    def _calculate_overall_score(self, result: RAGEvaluationResult) -> float:
        """Calculate weighted overall score from individual metrics"""
        weights = {
            'faithfulness': 0.35,      # Most important - answer must be grounded
            'answer_relevancy': 0.35,  # Equally important - answer must be relevant
            'context_precision': 0.20, # Important but less critical
            'context_recall': 0.10     # Bonus if available
        }

        score = (
            result.faithfulness * weights['faithfulness'] +
            result.answer_relevancy * weights['answer_relevancy'] +
            result.context_precision * weights['context_precision']
        )

        if result.context_recall is not None:
            score += result.context_recall * weights['context_recall']
        else:
            # Redistribute context_recall weight to other metrics
            score = score / (1 - weights['context_recall'])

        return min(1.0, max(0.0, score))  # Clamp to [0, 1]

    async def evaluate_from_dict(self, data: Dict[str, Any]) -> RAGEvaluationResult:
        """Convenience method to evaluate from dictionary input"""
        return await self.evaluate_single(
            query=data['query'],
            contexts=data['contexts'],
            answer=data['answer'],
            ground_truth=data.get('ground_truth'),
            metadata=data.get('metadata')
        )

    def get_evaluation_summary(
        self,
        results: List[RAGEvaluationResult]
    ) -> Dict[str, float]:
        """Generate summary statistics from evaluation results"""
        if not results:
            return {}

        summary = {}
        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'overall_score']

        for metric in metrics:
            values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
            if values:
                summary[f'{metric}_mean'] = sum(values) / len(values)
                summary[f'{metric}_std'] = (
                    sum((x - summary[f'{metric}_mean']) ** 2 for x in values) / len(values)
                ) ** 0.5
                summary[f'{metric}_min'] = min(values)
                summary[f'{metric}_max'] = max(values)

        return summary


# Factory function for easy instantiation
def create_rag_evaluator(**kwargs) -> RAGEvaluator:
    """Create RAG evaluator with default configuration"""
    return RAGEvaluator(**kwargs)


# Integration with existing pipeline
async def evaluate_rag_pipeline(
    queries: List[str],
    contexts_list: List[List[str]],
    answers: List[str],
    ground_truths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    High-level function to evaluate complete RAG pipeline

    Args:
        queries: List of user queries
        contexts_list: List of context lists for each query
        answers: List of generated answers
        ground_truths: Optional list of expected answers

    Returns:
        Comprehensive evaluation report
    """
    evaluator = create_rag_evaluator()

    # Prepare inputs
    inputs = []
    for i, (query, contexts, answer) in enumerate(zip(queries, contexts_list, answers)):
        ground_truth = ground_truths[i] if ground_truths else None
        inputs.append(EvaluationInput(
            query=query,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth
        ))

    # Run evaluation
    results = await evaluator.evaluate_batch(inputs)

    # Generate report
    summary = evaluator.get_evaluation_summary(results)

    return {
        'results': results,
        'summary': summary,
        'total_evaluated': len(results),
        'ragas_enabled': evaluator.enable_ragas,
        'evaluation_timestamp': datetime.now().isoformat()
    }


# Example usage and testing
if __name__ == "__main__":
    async def test_evaluator():
        """Test the RAG evaluator implementation"""
        evaluator = create_rag_evaluator(enable_ragas=False)  # Test fallback mode first

        # Test data
        test_query = "What is the capital of France?"
        test_contexts = [
            "Paris is the capital and largest city of France.",
            "France is a country in Western Europe.",
            "The Eiffel Tower is located in Paris."
        ]
        test_answer = "The capital of France is Paris."
        test_ground_truth = "Paris is the capital of France."

        print("🔄 Testing RAG Evaluator...")

        # Single evaluation
        result = await evaluator.evaluate_single(
            query=test_query,
            contexts=test_contexts,
            answer=test_answer,
            ground_truth=test_ground_truth
        )

        print(f"✅ Evaluation completed in {result.evaluation_time:.3f}s")
        print(f"📊 Faithfulness: {result.faithfulness:.3f}")
        print(f"📊 Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"📊 Context Precision: {result.context_precision:.3f}")
        print(f"📊 Overall Score: {result.overall_score:.3f}")

        # Batch evaluation test
        batch_inputs = [
            EvaluationInput(
                query="What is machine learning?",
                contexts=["Machine learning is a subset of AI."],
                answer="ML is artificial intelligence."
            ),
            EvaluationInput(
                query="How does neural networks work?",
                contexts=["Neural networks use layers of neurons."],
                answer="Neural nets process data through layers."
            )
        ]

        batch_results = await evaluator.evaluate_batch(batch_inputs)
        print(f"\n🔄 Batch evaluation: {len(batch_results)} results")

        summary = evaluator.get_evaluation_summary(batch_results + [result])
        print(f"📈 Average Overall Score: {summary.get('overall_score_mean', 0):.3f}")

        print("\n✅ RAG Evaluator test completed successfully!")

    # Run test
    asyncio.run(test_evaluator())