import asyncio
import chromadb
import os
from dotenv import load_dotenv
from src.services.paper.style_extractor import StyleExtractor
# Mock LLM for verification
class MockLLM:
    async def generate(self, prompt, **kwargs):
        if "transition" in prompt.lower():
            return '["However", "Furthermore", "In contrast"]', "mock"
        if "tone" in prompt.lower():
            return '{"tone": "academic", "voice": "passive", "confidence": "high"}', "mock"
        return "{}", "mock"

async def main():
    load_dotenv()
    
    # Initialize ChromaDB
    persist_directory = "chromadb_data"
    if not os.path.exists(persist_directory):
        print(f"❌ ChromaDB directory not found at {persist_directory}")
        return

    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = "golden_references_advanced_L0"
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ Connected to collection: {collection_name}")
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return

    # Get a few documents
    results = collection.peek(limit=3)
    if not results['documents']:
        print("❌ No documents found in collection")
        return

    print(f"✅ Found {len(results['documents'])} documents")

    # Initialize StyleExtractor with MockLLM to avoid config issues
    print("ℹ️  Using MockLLM for verification to avoid environment issues")
    llm = MockLLM()
    extractor = StyleExtractor(llm)

    for i, doc in enumerate(results['documents']):
        print(f"\n--- Document {i+1} ---")
        print(f"Preview: {doc[:100]}...")
        
        # Analyze structure
        metrics = await extractor.analyze_style(doc)
        print(f"Metrics: {metrics}")
        
        # Analyze transitions
        transitions = await extractor.extract_transitions(doc)
        print(f"Transitions: {transitions}")
        
        # Analyze tone
        tone = await extractor.analyze_tone(doc)
        print(f"Tone: {tone}")

if __name__ == "__main__":
    asyncio.run(main())
