"""
Unified RAG Orchestrator

Implementation for: Implement unified RAG orchestrator
Created: 2025-12-05

Acceptance Criteria:
- Strategy registry for all 6 RAG systems functional
- Query routing logic with fallback mechanisms
- Performance logging and metrics integration
- Configuration-driven strategy selection
- Thread-safe concurrent execution support

This orchestrator provides a unified interface to all RAG strategies with intelligent
routing, performance optimization, and comprehensive monitoring.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Configuration and utilities
from src.core.config import settings
from src.monitoring.rag_metrics import get_metrics_manager, rag_metrics_decorator, RAGMetrics
from src.services.rag.enhanced_dd_raptor import create_enhanced_dd_raptor, EnhancedDDRaptorSystem

# Setup logging
logger = logging.getLogger(__name__)

class RAGStrategy(Enum):
    """Available RAG strategy types"""
    HYBRID = "hybrid"
    ENHANCED_DD_RAPTOR = "enhanced_dd_raptor"
    GRAPH_RAG = "graph_rag"
    GOLDEN_REFERENCE = "golden_reference"
    SIMPLE_RAG = "simple_rag"
    MULTIMODAL_RAG = "multimodal_rag"
    PSYCHOLOGY_RAG = "psychology_rag"

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class QueryDomain(Enum):
    """Research domain categories"""
    NEUROSCIENCE = "neuroscience"
    QUANTUM_ML = "quantum_ml"
    GENERAL = "general"
    DEVELOPMENTAL_DISORDERS = "developmental_disorders"
    PSYCHOLOGY = "psychology"

@dataclass
class QueryContext:
    """Context information for query processing"""
    query: str
    complexity: QueryComplexity
    domain: QueryDomain
    intent: str  # factual, comparative, synthesis
    confidence: float
    metadata: Dict[str, Any]
    user_preferences: Optional[Dict[str, Any]] = None

@dataclass
class RAGResponse:
    """Standardized response from RAG strategies"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    strategy_used: RAGStrategy
    performance_metrics: Optional[RAGMetrics] = None
    metadata: Dict[str, Any] = None

class RAGStrategyInterface(ABC):
    """Abstract interface for RAG strategies"""

    @abstractmethod
    async def search(self, query_context: QueryContext) -> RAGResponse:
        """Execute search with the strategy"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if strategy is available"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> RAGStrategy:
        """Get strategy identifier"""
        pass

    @abstractmethod
    def estimate_performance(self, query_context: QueryContext) -> float:
        """Estimate performance score for this query"""
        pass

class RAGStrategyConfig:
    """Configuration for RAG strategies"""

    def __init__(self):
        self.strategy_configs = {
            RAGStrategy.HYBRID: {
                "enabled": True,
                "priority": 1,
                "domains": [QueryDomain.GENERAL, QueryDomain.NEUROSCIENCE],
                "complexity_range": [QueryComplexity.SIMPLE, QueryComplexity.MEDIUM],
                "max_concurrent": 5
            },
            RAGStrategy.ENHANCED_DD_RAPTOR: {
                "enabled": True,
                "priority": 2,
                "domains": [QueryDomain.DEVELOPMENTAL_DISORDERS, QueryDomain.NEUROSCIENCE, QueryDomain.GENERAL],
                "complexity_range": [QueryComplexity.SIMPLE, QueryComplexity.MEDIUM, QueryComplexity.COMPLEX],
                "max_concurrent": 3
            },
            RAGStrategy.GRAPH_RAG: {
                "enabled": True,
                "priority": 3,
                "domains": [QueryDomain.QUANTUM_ML, QueryDomain.GENERAL],
                "complexity_range": [QueryComplexity.COMPLEX],
                "max_concurrent": 2
            },
            RAGStrategy.GOLDEN_REFERENCE: {
                "enabled": True,
                "priority": 4,
                "domains": [QueryDomain.GENERAL, QueryDomain.NEUROSCIENCE, QueryDomain.QUANTUM_ML],
                "complexity_range": [QueryComplexity.SIMPLE, QueryComplexity.MEDIUM, QueryComplexity.COMPLEX],
                "max_concurrent": 10
            },
            RAGStrategy.SIMPLE_RAG: {
                "enabled": True,
                "priority": 5,
                "domains": [QueryDomain.GENERAL],
                "complexity_range": [QueryComplexity.SIMPLE],
                "max_concurrent": 10
            },
            RAGStrategy.MULTIMODAL_RAG: {
                "enabled": False,  # Not implemented yet
                "priority": 6,
                "domains": [QueryDomain.GENERAL],
                "complexity_range": [QueryComplexity.COMPLEX],
                "max_concurrent": 2
            },
            RAGStrategy.PSYCHOLOGY_RAG: {
                "enabled": True,
                "priority": 1,  # High priority for psychology queries
                "domains": [QueryDomain.PSYCHOLOGY],
                "complexity_range": [QueryComplexity.SIMPLE, QueryComplexity.MEDIUM, QueryComplexity.COMPLEX],
                "max_concurrent": 5
            }
        }

    def get_config(self, strategy: RAGStrategy) -> Dict[str, Any]:
        """Get configuration for a strategy"""
        return self.strategy_configs.get(strategy, {})

    def is_strategy_suitable(self, strategy: RAGStrategy, query_context: QueryContext) -> bool:
        """Check if strategy is suitable for the query context"""
        config = self.get_config(strategy)

        if not config.get("enabled", False):
            return False

        # Check domain compatibility
        if query_context.domain not in config.get("domains", []):
            return False

        # Check complexity compatibility
        if query_context.complexity not in config.get("complexity_range", []):
            return False

        return True

class MockRAGStrategy(RAGStrategyInterface):
    """Mock implementation for testing and fallback"""

    def __init__(self, strategy: RAGStrategy):
        self.strategy = strategy
        self._available = True

    async def search(self, query_context: QueryContext) -> RAGResponse:
        """Mock search implementation"""
        await asyncio.sleep(0.1)  # Simulate processing time

        # Generate mock response
        response = RAGResponse(
            answer=f"Mock answer from {self.strategy.value} strategy for: {query_context.query[:50]}...",
            sources=[
                {
                    "title": f"Mock Source 1 - {self.strategy.value}",
                    "content": "Mock content snippet from source 1",
                    "relevance": 0.9
                },
                {
                    "title": f"Mock Source 2 - {self.strategy.value}",
                    "content": "Mock content snippet from source 2",
                    "relevance": 0.8
                }
            ],
            confidence=0.85,
            strategy_used=self.strategy,
            metadata={
                "mock": True,
                "processing_time": 0.1,
                "strategy": self.strategy.value
            }
        )

        return response

    def is_available(self) -> bool:
        """Check availability"""
        return self._available

    def get_strategy_name(self) -> RAGStrategy:
        """Get strategy name"""
        return self.strategy

    def estimate_performance(self, query_context: QueryContext) -> float:
        """Estimate performance score"""
        # Mock performance estimation based on strategy and context
        base_score = {
            RAGStrategy.HYBRID: 0.8,
            RAGStrategy.ENHANCED_DD_RAPTOR: 0.85,
            RAGStrategy.GRAPH_RAG: 0.9,
            RAGStrategy.GOLDEN_REFERENCE: 0.7,
            RAGStrategy.SIMPLE_RAG: 0.6,
            RAGStrategy.MULTIMODAL_RAG: 0.95
        }.get(self.strategy, 0.5)

        # Adjust based on complexity
        complexity_modifier = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MEDIUM: 0.9,
            QueryComplexity.COMPLEX: 0.8
        }.get(query_context.complexity, 0.7)

        return base_score * complexity_modifier


class EnhancedDDRaptorAdapter(RAGStrategyInterface):
    """Adapter for EnhancedDDRaptorSystem to match RAGStrategyInterface"""
    
    def __init__(self, system: EnhancedDDRaptorSystem):
        self.system = system
        self.strategy = RAGStrategy.ENHANCED_DD_RAPTOR

    def is_available(self) -> bool:
        return True # Assuming availability if initialized

    def get_strategy_name(self) -> RAGStrategy:
        return self.strategy

    def estimate_performance(self, query_context: QueryContext) -> float:
        # High confidence for Developmental Disorders domain
        if query_context.domain == QueryDomain.DEVELOPMENTAL_DISORDERS:
            return 0.95
        elif query_context.domain == QueryDomain.NEUROSCIENCE:
            return 0.85
        return 0.6

    async def search(self, query_context: QueryContext) -> RAGResponse:
        # Delegate to real system
        result = await self.system.search(query_context.query)
        
        # Convert SearchResult to RAGResponse
        sources = []
        for i, doc in enumerate(result.documents):
            sources.append({
                "title": result.metadatas[i].get("title", f"Source {i}"),
                "content": doc,
                "relevance": result.relevancy_score # Simplified
            })
            
        return RAGResponse(
            answer="[Content retrieved via Enhanced DD-RAPTOR]", # RAG only retrieves, answer generation is upstream
            sources=sources,
            confidence=result.confidence,
            strategy_used=self.strategy,
            metadata={"latency_ms": result.latency_ms}
        )

class UnifiedRAGOrchestrator:
    """
    Unified orchestrator for all RAG strategies with intelligent routing,
    performance optimization, and comprehensive monitoring
    """

    def __init__(self, config: Optional[RAGStrategyConfig] = None):
        """Initialize the orchestrator"""
        self.config = config or RAGStrategyConfig()
        self.strategies: Dict[RAGStrategy, RAGStrategyInterface] = {}
        self.metrics_manager = get_metrics_manager()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=20)

        # Performance tracking
        self._strategy_performance = {}
        self._request_counts = {strategy: 0 for strategy in RAGStrategy}

        # Initialize strategies
        self._initialize_strategies()

        logger.info("Unified RAG Orchestrator initialized")

    def _initialize_strategies(self):
        """Initialize all available RAG strategies"""
        for strategy in RAGStrategy:
            try:
                # For now, use mock implementations
                # In production, these would be actual strategy implementations
                strategy_impl = MockRAGStrategy(strategy)

                if strategy_impl.is_available() and self.config.get_config(strategy).get("enabled", False):
                    self.strategies[strategy] = strategy_impl
                    logger.info(f"Initialized strategy: {strategy.value}")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy.value}: {e}")

        logger.info(f"Initialized {len(self.strategies)} RAG strategies")

    async def initialize_real_strategies(self):
        """Initialize real RAG strategy implementations (Async)"""
        logger.info("Initializing REAL RAG strategy implementations...")

        # 1. Enhanced DD-RAPTOR
        if self.config.get_config(RAGStrategy.ENHANCED_DD_RAPTOR).get("enabled", False):
            try:
                # Initialize real DD-RAPTOR system
                dd_raptor_system = await create_enhanced_dd_raptor()
                # Wrap in adapter
                self.strategies[RAGStrategy.ENHANCED_DD_RAPTOR] = EnhancedDDRaptorAdapter(dd_raptor_system)
                logger.info("✅ ENABLED: Real Enhanced DD-RAPTOR Strategy initialized")
            except Exception as e:
                if settings.strict_mode:
                    logger.error("🛑 STRICT MODE: Failed to initialize Real RAG Strategy. Halting.")
                    raise RuntimeError(f"Strict Mode Failure: Could not initialize Enhanced DD-RAPTOR: {e}") from e
                
                logger.error(f"❌ FAILED: Could not initialize Enhanced DD-RAPTOR: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback to mock is already in place
        
        # 2. Add other real strategies from multi_strategy_search
        try:
            from src.services.rag.multi_strategy_search import create_real_strategies, ChromaDBConfig
            
            # Map paths to config
            cfg = ChromaDBConfig(
                golden_references_path="chromadb_data",
                dd_raptor_path="chromadb_data_dd",
                grants_path="chromadb_grants_fixed_20251210_200233",
                esm3_papers_path="chromadb_new_papers_20251210_204818"
            )
            
            real_strategies = await create_real_strategies(cfg)
            for strategy_type, strategy_impl in real_strategies.items():
                if strategy_type == RAGStrategy.ENHANCED_DD_RAPTOR:
                    continue # Already handled or can be overwritten
                
                if self.config.get_config(strategy_type).get("enabled", False):
                    self.strategies[strategy_type] = strategy_impl
                    logger.info(f"✅ ENABLED: Real {strategy_type.value} Strategy initialized")
                    
        except ImportError as e:
            logger.warning(f"Could not import multi_strategy_search: {e}")
        except Exception as e:
            logger.error(f"Error initializing real strategies from multi_strategy_search: {e}")
        
        logger.info("Real RAG strategy initialization complete")



    async def search(
        self,
        query_context: QueryContext,
        strategy_override: Optional[RAGStrategy] = None,
        enable_fallback: bool = True
    ) -> RAGResponse:
        """
        Execute unified search with intelligent strategy selection

        Args:
            query_context: Query context with classification info
            strategy_override: Force specific strategy (optional)
            enable_fallback: Enable fallback to other strategies on failure

        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()

        # Select strategy
        if strategy_override and strategy_override in self.strategies:
            selected_strategies = [strategy_override]
        else:
            selected_strategies = self._select_strategies(query_context)

        if not selected_strategies:
            raise ValueError("No suitable strategies available for this query")

        # Try strategies in order of preference
        last_error = None
        for strategy in selected_strategies:
            try:
                logger.debug(f"Attempting strategy: {strategy.value}")

                # Execute search with metrics
                response = await self._execute_with_metrics(
                    strategy, query_context, start_time
                )

                # Update performance tracking
                self._update_performance_tracking(strategy, response, time.time() - start_time)

                return response

            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")
                last_error = e

                if not enable_fallback:
                    break

                continue

        # All strategies failed
        error_msg = f"All strategies failed. Last error: {last_error}"
        logger.error(error_msg)
        self.metrics_manager.record_error("unified_orchestrator", "all_strategies_failed")
        raise RuntimeError(error_msg)

    def _select_strategies(self, query_context: QueryContext) -> List[RAGStrategy]:
        """Select and rank strategies based on query context"""
        suitable_strategies = []

        for strategy, impl in self.strategies.items():
            if self.config.is_strategy_suitable(strategy, query_context):
                # Calculate suitability score
                performance_score = impl.estimate_performance(query_context)
                config = self.config.get_config(strategy)
                priority_score = 1.0 / config.get("priority", 10)  # Higher priority = lower number

                # Historical performance adjustment
                historical_score = self._get_historical_performance(strategy)

                total_score = (performance_score * 0.5 +
                             priority_score * 0.3 +
                             historical_score * 0.2)

                suitable_strategies.append((strategy, total_score))

        # Sort by score (descending) and return strategies
        suitable_strategies.sort(key=lambda x: x[1], reverse=True)
        return [strategy for strategy, _ in suitable_strategies]

    def _get_historical_performance(self, strategy: RAGStrategy) -> float:
        """Get historical performance score for strategy"""
        return self._strategy_performance.get(strategy, 0.5)

    async def _execute_with_metrics(
        self,
        strategy: RAGStrategy,
        query_context: QueryContext,
        start_time: float
    ) -> RAGResponse:
        """Execute strategy with comprehensive metrics collection"""
        strategy_impl = self.strategies[strategy]

        # Execute search
        response = await strategy_impl.search(query_context)

        # Create performance metrics
        execution_time = time.time() - start_time

        metrics = RAGMetrics(
            latency=execution_time,
            quality_score=response.confidence,
            tokens_processed=len(query_context.query) + len(response.answer),
            retrieval_time=execution_time * 0.3,  # Estimate
            generation_time=execution_time * 0.7,  # Estimate
            context_relevance=response.confidence,
            faithfulness=response.confidence * 0.9,
            answer_relevancy=response.confidence * 0.95,
            strategy=strategy.value,
            timestamp=datetime.now()
        )

        # Record metrics
        self.metrics_manager.record_rag_request(metrics)

        # Add metrics to response
        response.performance_metrics = metrics

        return response

    def _update_performance_tracking(
        self,
        strategy: RAGStrategy,
        response: RAGResponse,
        execution_time: float
    ):
        """Update historical performance tracking"""
        with self._lock:
            self._request_counts[strategy] += 1

            # Update rolling average of performance
            current_score = self._strategy_performance.get(strategy, 0.5)
            new_score = (response.confidence + (1.0 / max(execution_time, 0.1))) / 2

            # Exponential moving average
            alpha = 0.1
            self._strategy_performance[strategy] = (alpha * new_score +
                                                  (1 - alpha) * current_score)

    async def search_parallel(
        self,
        query_contexts: List[QueryContext],
        max_concurrent: int = 5
    ) -> List[RAGResponse]:
        """Execute multiple searches in parallel"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_search(query_context: QueryContext) -> RAGResponse:
            async with semaphore:
                return await self.search(query_context)

        tasks = [bounded_search(ctx) for ctx in query_contexts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_strategy_health(self) -> Dict[str, Any]:
        """Get health status of all strategies"""
        health_status = {}

        for strategy, impl in self.strategies.items():
            health_status[strategy.value] = {
                "available": impl.is_available(),
                "request_count": self._request_counts[strategy],
                "performance_score": self._strategy_performance.get(strategy, 0.0),
                "config": self.config.get_config(strategy)
            }

        return health_status

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "strategies": self.get_strategy_health(),
            "total_requests": sum(self._request_counts.values()),
            "active_strategies": len([s for s in self.strategies.values() if s.is_available()]),
            "metrics_summary": self.metrics_manager.get_all_metrics()
        }

    async def warmup(self):
        """Warm up all strategies with test queries"""
        logger.info("Warming up RAG strategies...")

        test_contexts = [
            QueryContext(
                query="What is machine learning?",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.GENERAL,
                intent="factual",
                confidence=1.0,
                metadata={"warmup": True}
            ),
            QueryContext(
                query="How do neural networks process information in the brain?",
                complexity=QueryComplexity.MEDIUM,
                domain=QueryDomain.NEUROSCIENCE,
                intent="comparative",
                confidence=0.9,
                metadata={"warmup": True}
            )
        ]

        for ctx in test_contexts:
            try:
                await self.search(ctx, enable_fallback=False)
            except Exception as e:
                logger.debug(f"Warmup failed for {ctx.domain.value}: {e}")

        logger.info("RAG strategies warmup completed")

    def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("Shutting down Unified RAG Orchestrator...")
        self._executor.shutdown(wait=True)
        logger.info("Orchestrator shutdown complete")

# Factory function for easy instantiation
def create_unified_orchestrator(
    config: Optional[RAGStrategyConfig] = None
) -> UnifiedRAGOrchestrator:
    """Create unified RAG orchestrator with default configuration"""
    return UnifiedRAGOrchestrator(config)

# Global instance for application-wide use
_global_orchestrator: Optional[UnifiedRAGOrchestrator] = None

def get_orchestrator() -> UnifiedRAGOrchestrator:
    """Get global orchestrator instance"""
    global _global_orchestrator

    if _global_orchestrator is None:
        _global_orchestrator = create_unified_orchestrator()

    return _global_orchestrator

# Example usage and testing
if __name__ == "__main__":
    async def test_orchestrator():
        """Test the unified orchestrator"""
        print("🔄 Testing Unified RAG Orchestrator...")

        # Create orchestrator
        orchestrator = create_unified_orchestrator()

        # Warmup
        await orchestrator.warmup()

        # Test query
        query_context = QueryContext(
            query="What are the latest developments in quantum machine learning?",
            complexity=QueryComplexity.COMPLEX,
            domain=QueryDomain.QUANTUM_ML,
            intent="synthesis",
            confidence=0.9,
            metadata={"test": True}
        )

        # Execute search
        response = await orchestrator.search(query_context)

        print(f"✅ Search completed")
        print(f"📊 Strategy used: {response.strategy_used.value}")
        print(f"📊 Confidence: {response.confidence:.3f}")
        print(f"📊 Answer: {response.answer[:100]}...")

        # Test parallel search
        query_contexts = [
            QueryContext(
                query=f"Test query {i}",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.GENERAL,
                intent="factual",
                confidence=1.0,
                metadata={"test": True, "batch": True}
            ) for i in range(3)
        ]

        responses = await orchestrator.search_parallel(query_contexts, max_concurrent=2)
        print(f"🔄 Parallel search completed: {len(responses)} responses")

        # Get performance summary
        summary = orchestrator.get_performance_summary()
        print(f"📈 Performance summary: {summary['total_requests']} total requests")

        # Cleanup
        orchestrator.shutdown()
        print("✅ Unified RAG Orchestrator test completed successfully!")

    # Run test
    asyncio.run(test_orchestrator())