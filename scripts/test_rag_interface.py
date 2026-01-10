import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.hybrid_rag_service import HybridRAGService
from src.core.config import settings

async def test_retrieval():
    print("Initializing HybridRAGService...")
    # Initialize without LLMs to speed up if possible, but the init might force it.
    # We rely on defaults.
    rag = HybridRAGService()
    
    print("Testing retrieve_similar_papers...")
    query = "EEG classification transformer"
    try:
        results = await rag.retrieve_similar_papers(
            query=query,
            top_k_retrieve=3,
            top_k_rerank=2
        )
        print(f"Result type: {type(results)}")
        
        if isinstance(results, tuple):
            docs, scores = results
            print(f"Docs type: {type(docs)}, length: {len(docs)}")
            print(f"Scores type: {type(scores)}, length: {len(scores)}")
            
            if len(docs) > 0:
                print(f"First doc keys: {docs[0].keys()}")
                print(f"First doc content sample: {docs[0].get('content', 'No content')[:50]}...")
        else:
            print("Unexpected return type.")
            
    except Exception as e:
        print(f"Error during retrieval: {e}")
    finally:
        await rag.close()

if __name__ == "__main__":
    # Export API key if needed from env var inside the logical block
    # but we assume .env is loaded or env vars are set.
    asyncio.run(test_retrieval())
