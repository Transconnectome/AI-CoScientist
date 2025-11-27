from mcp.server.fastmcp import FastMCP
import chromadb
import json
from typing import List, Dict, Any

# Initialize FastMCP
mcp = FastMCP("Developmental Disability RAG")

# Initialize ChromaDB
# We use absolute path or relative to where it's run. 
# Assuming run from project root.
DB_PATH = "chromadb_data_dd"

from sentence_transformers import SentenceTransformer, CrossEncoder

# Initialize embedding model globally to avoid reloading
print("Loading SciBERT model...")
embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
print("✓ SciBERT loaded")

# Initialize Cross-Encoder globally
print("Loading Cross-Encoder...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✓ Cross-Encoder loaded")

def get_collection():
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        # Get collection without specifying embedding function (uses default metadata)
        return client.get_collection(name="dd_papers_L0")
    except Exception as e:
        return None

@mcp.tool()
def search_dd_papers(query: str, n_results: int = 5) -> str:
    """
    Search for relevant information in the developmental disability research papers.
    Uses hybrid retrieval: Vector Search (Top-50) -> Cross-Encoder Re-ranking (Top-N).
    
    Args:
        query: The search query (e.g., "autism diagnosis deep learning")
        n_results: Number of results to return (default: 5)
        
    Returns:
        A formatted string containing the search results.
    """
    collection = get_collection()
    if not collection:
        return "Error: Could not connect to ChromaDB. Make sure 'chromadb_data_dd' exists."

    try:
        # Manually embed the query
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # 1. Retrieval (Fetch 50 candidates)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=50
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        if not documents:
            return "No results found."

        # 2. Re-ranking
        pairs = [[query, doc] for doc in documents]
        scores = cross_encoder.predict(pairs)
        
        # Sort by score
        scored_results = []
        for i, score in enumerate(scores):
            scored_results.append({
                'document': documents[i],
                'metadata': metadatas[i],
                'score': score
            })
        
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top N
        final_results = scored_results[:n_results]
        
        output = []
        output.append(f"Found {len(documents)} candidates, returning top {n_results} after re-ranking for query: '{query}'\n")
        
        for i, res in enumerate(final_results):
            title = res['metadata'].get('paper_title', 'Unknown Title')
            section = res['metadata'].get('section', 'Unknown Section')
            score = res['score']
            
            output.append(f"--- Result {i+1} (Score: {score:.4f}) ---")
            output.append(f"Paper: {title}")
            output.append(f"Section: {section}")
            output.append(f"Content: {res['document']}\n")
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error querying database: {str(e)}"

if __name__ == "__main__":
    mcp.run()
