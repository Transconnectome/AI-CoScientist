import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator,
    QueryContext,
    QueryComplexity,
    QueryDomain,
    RAGStrategy
)

async def test_golden_reference_retrieval():
    print("🚀 Initializing Unified RAG Orchestrator...")
    orchestrator = UnifiedRAGOrchestrator()
    await orchestrator.initialize_real_strategies()
    
    queries = [
        {
            "text": "What are the core components of the SwinTransformer4D architecture for fMRI?",
            "domain": QueryDomain.NEUROSCIENCE,
            "complexity": QueryComplexity.COMPLEX
        },
        {
            "text": "Explain the concept of foundation models in medical imaging.",
            "domain": QueryDomain.GENERAL,
            "complexity": QueryComplexity.MEDIUM
        },
        {
            "text": "How does RAPTOR improve RAG performance in scientific papers?",
            "domain": QueryDomain.GENERAL,
            "complexity": QueryComplexity.COMPLEX
        }
    ]
    
    print("\n🔍 Starting Verification Searches...\n")
    
    for q in queries:
        print(f"--- Query: {q['text']} ---")
        context = QueryContext(
            query=q['text'],
            complexity=q['complexity'],
            domain=q['domain'],
            intent="synthesis",
            confidence=0.8,
            metadata={}
        )
        
        try:
            response = await orchestrator.search(context)
            
            print(f"✅ Strategy Used: {response.strategy_used.value}")
            print(f"✅ Confidence: {response.confidence:.3f}")
            print(f"✅ Answer Snippet: {response.answer[:200]}...")
            print(f"✅ Sources Found: {len(response.sources)}")
            
            for i, source in enumerate(response.sources[:3]):
                print(f"   [{i+1}] {source['title']} (Score: {source['relevance']:.3f})")
                print(f"       Database: {source.get('database', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50 + "\n")

from src.services.rag.hybrid_dd_search import HybridDDSearch

async def test_hybrid_dd_fm_search():
    print("\n🚀 Testing Hybrid DD + FM Search...")
    hybrid_search = HybridDDSearch()
    
    query = "How can we use foundation models to improve developmental disorder diagnosis?"
    print(f"🔍 Hybrid Search Query: {query}")
    
    try:
        response = hybrid_search.search(query, top_k=5)
        
        print(f"✅ Query Classification: {response.query_classification.query_type.value}")
        print(f"✅ Total Results: {len(response.results)}")
        print(f"✅ DD Results: {response.dd_count}")
        print(f"✅ FM Results: {response.fm_count}")
        
        for i, result in enumerate(response.results):
            source_tag = "🔬 [DD]" if result.source == "DD" else "🤖 [FM]"
            print(f"   {i+1}. {source_tag} Score: {result.combined_score:.3f} | Rank: {result.rank}")
            print(f"      Reasoning: {result.reasoning}")
            
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    await test_golden_reference_retrieval()
    await test_hybrid_dd_fm_search()

if __name__ == "__main__":
    asyncio.run(main())
