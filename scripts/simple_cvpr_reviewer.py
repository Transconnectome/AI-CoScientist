import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import settings
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

async def generate_review(paper_path, output_path):
    print(f"Processing {paper_path}...")
    
    # 0. Find DB Path
    try:
        with open("latest_papers_db_path.txt", "r") as f:
            db_path = f.read().strip()
        print(f"Using KBL: {db_path}")
    except FileNotFoundError:
        print("Error: latest_papers_db_path.txt not found. Did ingestion run?")
        return

    # 1. Read Paper
    with open(paper_path, 'r') as f:
        content = f.read()
    
    # 2. RAG Search (Direct ChromaDB)
    print("Loading Search Models...")
    # Optimize: load only if not testing just generation, but we need context
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    client = chromadb.PersistentClient(path=db_path)
    # Ensure collection exists
    try:
        collection = client.get_collection("new_papers")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    print("Searching Knowledge Base...")
    query_text = content[:1000]
    query_embedding = embedder.encode([query_text]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    
    context_text = ""
    if results and 'documents' in results:
        docs = results['documents'][0] 
        metadatas = results['metadatas'][0]
        
        print(f"Found {len(docs)} relevant papers.")
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            title = meta.get('title', 'Unknown')
            context_text += f"\n--- Reference {i+1}: {title} ---\n{doc}\n"

    # 3. LLM Call (Gemini)
    print("Generating Review with Gemini...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = settings.google_api_key
        
    genai.configure(api_key=api_key)
    
    # Use a solid model for reasoning
    model_name = "gemini-2.5-pro"
    # If using preview models, names might differ. fallback to gemini-1.5-pro
    
    print(f"Using model: {model_name}")
    model = genai.GenerativeModel(model_name)
    
    system_prompt = """You are a distinguished CVPR Area Chair and Reviewer. 
    Your task is to write a rigorous, constructive, and critical review for a CVPR submission.
    Use the provided Reference Context to critique the paper's novelty and compare it with SOTA.
    
    Output Format: Markdown.
    Structure:
    1. Summary
    2. Strengths
    3. Weaknesses
    4. Detailed Comments (Methodology, Experiments, Writing)
    5. Comparison with SOTA (referencing specific papers from context)
    6. Overall Rating (1-5) & Confidence (1-5)
    """
    
    user_prompt = f"""
    ### Reference SOTA Context:
    {context_text}
    
    ### Target Paper Content:
    {content[:20000]} ... (truncated)
    
    ### Instructions:
    Review this paper based on the system instructions.
    """
    
    # Gemini combine prompts
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        response = model.generate_content(combined_prompt)
        review_content = response.text
        
        # 4. Save
        with open(output_path, 'w') as f:
            f.write(review_content)
        
        print(f"Review saved to {output_path}")
        
    except Exception as e:
        print(f"Gemini Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    asyncio.run(generate_review(args.input, args.output))
