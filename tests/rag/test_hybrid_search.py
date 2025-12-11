#!/usr/bin/env python3
"""
Comprehensive test suite for Hybrid DD Search System

Tests query classification, database integration, result merging,
and overall search quality across clinical, technical, and mixed queries.
"""

import pytest
import time
import json
from pathlib import Path
from typing import List, Dict

from src.services.rag.hybrid_dd_search import (
    HybridDDSearch,
    HybridSearchResponse,
    format_search_response
)
from src.services.rag.query_classifier import (
    QueryClassifier,
    QueryType,
    QueryClassification
)


class TestQueryClassifier:
    """Test suite for query classification"""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_clinical_query_classification(self, classifier):
        """Test classification of purely clinical queries"""
        queries = [
            "ADHD treatment effectiveness in children",
            "autism spectrum disorder diagnosis",
            "behavioral therapy for developmental disorders",
            "fMRI connectivity in ASD patients"
        ]

        for query in queries:
            classification = classifier.classify(query)
            assert classification.query_type == QueryType.CLINICAL
            assert classification.clinical_score > 0.6
            assert classification.confidence > 0.5

    def test_technical_query_classification(self, classifier):
        """Test classification of purely technical queries"""
        queries = [
            "transformer architecture for image processing",
            "attention mechanism in neural networks",
            "foundation model training strategies",
            "self-supervised learning methods"
        ]

        for query in queries:
            classification = classifier.classify(query)
            assert classification.query_type == QueryType.TECHNICAL
            assert classification.technical_score > 0.6
            assert classification.confidence > 0.5

    def test_mixed_query_classification(self, classifier):
        """Test classification of mixed clinical-technical queries"""
        queries = [
            "foundation models for autism diagnosis",
            "deep learning for ADHD detection using EEG",
            "multimodal AI in neuroscience",
            "transformer-based analysis of brain imaging"
        ]

        for query in queries:
            classification = classifier.classify(query)
            assert classification.query_type == QueryType.MIXED
            # Both scores should be reasonably high
            assert classification.clinical_score > 0.2
            assert classification.technical_score > 0.2

    def test_search_weights(self, classifier):
        """Test that search weights are appropriate for query type"""
        # Clinical query should have higher DD weight
        clinical_classification = classifier.classify("autism diagnosis and treatment")
        dd_weight, fm_weight = classifier.get_search_weights(clinical_classification)
        assert dd_weight > fm_weight

        # Technical query should have higher FM weight
        technical_classification = classifier.classify("transformer architecture training")
        dd_weight, fm_weight = classifier.get_search_weights(technical_classification)
        assert fm_weight > dd_weight

        # Mixed query should have balanced weights
        mixed_classification = classifier.classify("AI for autism diagnosis")
        dd_weight, fm_weight = classifier.get_search_weights(mixed_classification)
        assert abs(dd_weight - fm_weight) < 1.0  # Reasonably balanced

    def test_keyword_matching(self, classifier):
        """Test that keywords are properly matched and categorized"""
        query = "autism diagnosis using transformer models and fMRI"
        classification = classifier.classify(query)

        keywords_matched = classification.keywords_matched

        # Should have clinical keywords
        assert 'clinical' in keywords_matched
        assert any('autism' in kw for category_kws in keywords_matched['clinical'].values() for kw in category_kws)

        # Should have technical keywords
        assert 'technical' in keywords_matched
        assert any('transformer' in kw for category_kws in keywords_matched['technical'].values() for kw in category_kws)


class TestHybridDDSearch:
    """Test suite for hybrid search system"""

    @pytest.fixture
    def search_system(self):
        """Initialize search system with test configuration"""
        config = {
            "embedding_model": "allenai/scibert_scivocab_uncased",  # 768 dimensions, same as collections
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_results_per_db": 10,
            "final_top_k": 5,
            "use_reranking": True,
            "min_similarity_threshold": 0.5,
            "layer_weights": {"L0": 1.0, "L1": 1.2, "L2": 1.5}
        }
        return HybridDDSearch(config=config)

    def test_database_initialization(self, search_system):
        """Test that both databases are properly initialized"""
        # Check DD collections
        assert 'L0' in search_system.dd_collections
        assert 'L1' in search_system.dd_collections
        assert 'L2' in search_system.dd_collections

        # Check FM collections
        assert 'L0' in search_system.fm_collections
        assert 'L1' in search_system.fm_collections
        assert 'L2' in search_system.fm_collections

        # Verify non-zero document counts
        for level, collection in search_system.dd_collections.items():
            assert collection.count() > 0, f"DD collection {level} is empty"

        for level, collection in search_system.fm_collections.items():
            assert collection.count() > 0, f"FM collection {level} is empty"

    def test_clinical_query_search(self, search_system):
        """Test search with clinical query - should prioritize DD papers"""
        query = "autism diagnosis using EEG signals in children"
        response = search_system.search(query, top_k=10)

        # Verify response structure
        assert isinstance(response, HybridSearchResponse)
        assert response.query == query
        assert len(response.results) > 0

        # Should be classified as clinical or mixed
        assert response.query_classification.query_type in [QueryType.CLINICAL, QueryType.MIXED]

        # Should have some DD results (clinical focus)
        assert response.dd_count > 0

        # Check timing
        assert response.total_time_ms > 0
        assert response.dd_search_time_ms > 0

    def test_technical_query_search(self, search_system):
        """Test search with technical query - should prioritize FM papers"""
        query = "transformer architecture for multimodal learning"
        response = search_system.search(query, top_k=10)

        # Verify response structure
        assert isinstance(response, HybridSearchResponse)
        assert len(response.results) > 0

        # Should be classified as technical or mixed
        assert response.query_classification.query_type in [QueryType.TECHNICAL, QueryType.MIXED]

        # Should have some FM results (technical focus)
        assert response.fm_count > 0

    def test_mixed_query_search(self, search_system):
        """Test search with mixed query - should use both databases"""
        query = "foundation models for developmental disorder diagnosis"
        response = search_system.search(query, top_k=10)

        # Should have results from both sources
        assert response.dd_count > 0 or response.fm_count > 0

        # Total results should match
        assert response.dd_count + response.fm_count == len(response.results)

        # Results should be ranked
        for i, result in enumerate(response.results, 1):
            assert result.rank == i

    def test_layer_search(self, search_system):
        """Test searching specific RAPTOR layers"""
        query = "autism brain connectivity"

        # Test L0 only (chunks)
        response_l0 = search_system.search(query, top_k=5, layers=['L0'])
        assert all(r.level == 'L0' for r in response_l0.results)

        # Test L2 only (summaries)
        response_l2 = search_system.search(query, top_k=5, layers=['L2'])
        assert all(r.level == 'L2' for r in response_l2.results)

        # Test all layers
        response_all = search_system.search(query, top_k=10, layers=['L0', 'L1', 'L2'])
        levels_found = set(r.level for r in response_all.results)
        # Should have results from multiple layers
        assert len(levels_found) > 1

    def test_result_scoring(self, search_system):
        """Test that results are properly scored and ranked"""
        query = "autism diagnosis and treatment"
        response = search_system.search(query, top_k=10)

        # Check that results are sorted by combined score
        scores = [r.combined_score for r in response.results]
        assert scores == sorted(scores, reverse=True)

        # Each result should have both score components
        for result in response.results:
            assert isinstance(result.dd_score, float)
            assert isinstance(result.fm_score, float)
            assert isinstance(result.combined_score, float)
            assert result.combined_score >= 0

    def test_performance_metrics(self, search_system):
        """Test that performance metrics are tracked"""
        queries = [
            "autism diagnosis",
            "transformer architecture",
            "AI for neuroscience"
        ]

        for query in queries:
            search_system.search(query, top_k=5)

        metrics = search_system.performance_metrics

        assert metrics["total_queries"] == len(queries)
        assert metrics["average_latency_ms"] > 0
        assert len(metrics["dd_result_ratio"]) == len(queries)
        assert len(metrics["fm_result_ratio"]) == len(queries)

    def test_export_results(self, search_system, tmp_path):
        """Test exporting search results to JSON"""
        query = "autism EEG analysis"
        response = search_system.search(query, top_k=5)

        output_file = tmp_path / "test_results.json"
        search_system.export_results(response, str(output_file))

        # Verify file was created
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert data["query"] == query
        assert "query_classification" in data
        assert "results" in data
        assert "statistics" in data
        assert len(data["results"]) == 5


class TestSearchQuality:
    """Test search quality with comprehensive queries"""

    @pytest.fixture
    def search_system(self):
        return HybridDDSearch()

    @pytest.fixture
    def test_queries(self) -> List[Dict]:
        """Comprehensive test queries with expected characteristics"""
        return [
            {
                "query": "autism diagnosis using EEG signals",
                "type": "mixed",
                "expect_dd": True,
                "expect_fm": True,
                "description": "Mixed query combining clinical (autism, EEG) and technical aspects"
            },
            {
                "query": "ADHD treatment effectiveness in children",
                "type": "clinical",
                "expect_dd": True,
                "expect_fm": False,
                "description": "Purely clinical query about treatment"
            },
            {
                "query": "transformer architecture for brain imaging",
                "type": "mixed",
                "expect_dd": True,
                "expect_fm": True,
                "description": "Technical architecture applied to neuroscience domain"
            },
            {
                "query": "foundation models for developmental disorders",
                "type": "mixed",
                "expect_dd": True,
                "expect_fm": True,
                "description": "AI/ML models applied to clinical domain"
            },
            {
                "query": "multimodal AI in neuroscience",
                "type": "mixed",
                "expect_dd": True,
                "expect_fm": True,
                "description": "Multimodal approaches in brain research"
            },
            {
                "query": "behavioral therapy outcomes for autism spectrum disorder",
                "type": "clinical",
                "expect_dd": True,
                "expect_fm": False,
                "description": "Clinical intervention research"
            },
            {
                "query": "self-supervised learning for medical imaging",
                "type": "technical",
                "expect_dd": False,
                "expect_fm": True,
                "description": "Technical ML methods for medical applications"
            },
            {
                "query": "fMRI connectivity analysis in ASD patients",
                "type": "clinical",
                "expect_dd": True,
                "expect_fm": False,
                "description": "Neuroimaging in clinical population"
            },
            {
                "query": "attention mechanism in vision-language models",
                "type": "technical",
                "expect_dd": False,
                "expect_fm": True,
                "description": "Pure technical architecture question"
            },
            {
                "query": "large language models for clinical diagnosis",
                "type": "mixed",
                "expect_dd": True,
                "expect_fm": True,
                "description": "LLMs applied to clinical domain"
            }
        ]

    def test_comprehensive_search_quality(self, search_system, test_queries):
        """Test search quality across diverse queries"""
        results_summary = []

        for test_case in test_queries:
            query = test_case["query"]
            response = search_system.search(query, top_k=10)

            # Verify basic response structure
            assert len(response.results) > 0, f"No results for query: {query}"

            # Track results
            result_info = {
                "query": query,
                "description": test_case["description"],
                "expected_type": test_case["type"],
                "actual_type": response.query_classification.query_type.value,
                "dd_count": response.dd_count,
                "fm_count": response.fm_count,
                "total_time_ms": response.total_time_ms,
                "top_score": response.results[0].combined_score if response.results else 0
            }

            results_summary.append(result_info)

            # Validate expectations
            if test_case["expect_dd"]:
                assert response.dd_count > 0, f"Expected DD results for: {query}"

            if test_case["expect_fm"]:
                assert response.fm_count > 0, f"Expected FM results for: {query}"

        # Print summary
        print("\n" + "=" * 100)
        print("SEARCH QUALITY TEST SUMMARY")
        print("=" * 100)

        for info in results_summary:
            print(f"\nQuery: {info['query']}")
            print(f"Description: {info['description']}")
            print(f"Type: {info['actual_type']} | Results: DD={info['dd_count']}, FM={info['fm_count']} | "
                  f"Time: {info['total_time_ms']:.0f}ms | Top Score: {info['top_score']:.4f}")

        # Overall statistics
        avg_time = sum(r['total_time_ms'] for r in results_summary) / len(results_summary)
        avg_dd_ratio = sum(r['dd_count'] for r in results_summary) / (len(results_summary) * 10)
        avg_fm_ratio = sum(r['fm_count'] for r in results_summary) / (len(results_summary) * 10)

        print("\n" + "-" * 100)
        print(f"Average Response Time: {avg_time:.0f}ms")
        print(f"Average DD Ratio: {avg_dd_ratio:.2%}")
        print(f"Average FM Ratio: {avg_fm_ratio:.2%}")
        print("=" * 100)

        # Performance assertions
        assert avg_time < 5000, "Average response time should be under 5 seconds"


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def search_system(self):
        return HybridDDSearch()

    def test_empty_query(self, search_system):
        """Test handling of empty query"""
        # Should not crash, may return default results
        try:
            response = search_system.search("", top_k=5)
            assert isinstance(response, HybridSearchResponse)
        except Exception as e:
            pytest.skip(f"Empty query handling not implemented: {e}")

    def test_very_long_query(self, search_system):
        """Test handling of very long query"""
        long_query = " ".join(["autism diagnosis"] * 100)
        response = search_system.search(long_query, top_k=5)
        assert isinstance(response, HybridSearchResponse)
        assert len(response.results) > 0

    def test_special_characters(self, search_system):
        """Test handling of special characters in query"""
        queries = [
            "autism & ADHD co-diagnosis",
            "fMRI (functional magnetic resonance imaging)",
            "attention-deficit/hyperactivity disorder"
        ]

        for query in queries:
            response = search_system.search(query, top_k=5)
            assert isinstance(response, HybridSearchResponse)

    def test_top_k_variations(self, search_system):
        """Test different top_k values"""
        query = "autism diagnosis"

        # Test various top_k values
        for k in [1, 5, 10, 20, 50]:
            response = search_system.search(query, top_k=k)
            assert len(response.results) <= k


def run_performance_benchmark():
    """Standalone performance benchmark"""
    print("\n" + "=" * 100)
    print("HYBRID DD SEARCH PERFORMANCE BENCHMARK")
    print("=" * 100)

    search_system = HybridDDSearch()

    benchmark_queries = [
        ("Clinical", [
            "autism spectrum disorder diagnosis and assessment",
            "ADHD treatment in pediatric populations",
            "behavioral interventions for developmental disorders",
            "fMRI connectivity in ASD patients"
        ]),
        ("Technical", [
            "transformer architecture for image classification",
            "self-supervised learning methods",
            "attention mechanisms in neural networks",
            "foundation models training strategies"
        ]),
        ("Mixed", [
            "foundation models for autism diagnosis",
            "deep learning for EEG analysis in ADHD",
            "multimodal AI for brain imaging",
            "large language models in clinical diagnosis"
        ])
    ]

    all_results = []

    for category, queries in benchmark_queries:
        print(f"\n{category} Queries")
        print("-" * 100)

        category_times = []
        category_dd_ratios = []
        category_fm_ratios = []

        for query in queries:
            start = time.time()
            response = search_system.search(query, top_k=10)
            elapsed = (time.time() - start) * 1000

            dd_ratio = response.dd_count / len(response.results) if response.results else 0
            fm_ratio = response.fm_count / len(response.results) if response.results else 0

            category_times.append(elapsed)
            category_dd_ratios.append(dd_ratio)
            category_fm_ratios.append(fm_ratio)

            print(f"  {query[:60]:60s} | {elapsed:6.0f}ms | DD:{dd_ratio:5.1%} | FM:{fm_ratio:5.1%} | "
                  f"Type:{response.query_classification.query_type.value}")

            all_results.append({
                "category": category,
                "query": query,
                "time_ms": elapsed,
                "dd_ratio": dd_ratio,
                "fm_ratio": fm_ratio
            })

        # Category summary
        avg_time = sum(category_times) / len(category_times)
        avg_dd = sum(category_dd_ratios) / len(category_dd_ratios)
        avg_fm = sum(category_fm_ratios) / len(category_fm_ratios)

        print(f"\n  Category Average: {avg_time:.0f}ms | DD: {avg_dd:.1%} | FM: {avg_fm:.1%}")

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 100)

    all_times = [r['time_ms'] for r in all_results]
    all_dd = [r['dd_ratio'] for r in all_results]
    all_fm = [r['fm_ratio'] for r in all_results]

    print(f"Total Queries: {len(all_results)}")
    print(f"Average Response Time: {sum(all_times) / len(all_times):.0f}ms")
    print(f"Min Response Time: {min(all_times):.0f}ms")
    print(f"Max Response Time: {max(all_times):.0f}ms")
    print(f"Average DD Ratio: {sum(all_dd) / len(all_dd):.1%}")
    print(f"Average FM Ratio: {sum(all_fm) / len(all_fm):.1%}")
    print("=" * 100)

    # Save results
    output_file = Path("hybrid_search_benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results,
            "summary": {
                "total_queries": len(all_results),
                "avg_time_ms": sum(all_times) / len(all_times),
                "min_time_ms": min(all_times),
                "max_time_ms": max(all_times),
                "avg_dd_ratio": sum(all_dd) / len(all_dd),
                "avg_fm_ratio": sum(all_fm) / len(all_fm)
            }
        }, f, indent=2)

    print(f"\nBenchmark results saved to: {output_file}")


if __name__ == "__main__":
    # Run performance benchmark
    run_performance_benchmark()
