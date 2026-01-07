
import asyncio
import os
import chromadb
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PAPER_TEXT_PATH = "/home/juke/git/AI-CoScientist/paper_text.txt"
REVIEWS_PATH = "/home/juke/git/AI-CoScientist/data/first_round_reviews.txt"
OUTPUT_PATH = "/home/juke/git/AI-CoScientist/data/final_npp_review_antigravity.md"
CHROMADB_PATH = "chromadb_data"

# Antigravity Persona Definition
ANTIGRAVITY_PERSONA = """
You are Antigravity, an advanced AI scientist and Editor-in-Chief of Neuropsychopharmacology (NPP).
You possess the combined knowledge of the world's top neuroscientists and peer reviewers.
You are critical, precise, fair, and deeply insightful. You do not tolerate mediocrity but champion true scientific innovation.
"""

class RAGReviewer:
    def __init__(self):
        # LLM Setup - Using the most up-to-date model available
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-3-pro-preview") 
        
        # RAG Setup
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('allenai/scibert_scivocab_uncased')
        self.client = chromadb.PersistentClient(path=CHROMADB_PATH)
        try:
            self.collection = self.client.get_collection("golden_references_advanced_L0")
            print("Connected to RAG collection.")
        except Exception as e:
            print(f"Warning: RAG collection not found ({e}). Proceeding without RAG context.")
            self.collection = None

    def retrieve_context(self, query, n_results=10):
        if not self.collection:
            return ""
            
        print(f"Retrieving context for: {query[:50]}...")
        embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results
        )
        
        context_parts = []
        if results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                source = meta.get('filename', 'Unknown')
                context_parts.append(f"[Source: {source}]\n{doc}")
        
        return "\n\n".join(context_parts)

    async def generate_review(self, paper_text, past_reviews):
        # 1. Retrieve relevant context from RAG
        query_text = paper_text[:2000] 
        context = self.retrieve_context(query_text, n_results=15)
        
        prompt = f"""
{ANTIGRAVITY_PERSONA}

Your task is to evaluate a revised manuscript titled "Sensorimotor circuit connectivity as a candidate biomarker for responsiveness to sertraline in obsessive-compulsive disorder" and generate the FINAL REVIEW REPORT for the journal.

**Context:**
The paper claims to predict sertraline response using sensorimotor circuit fcMRI in N=54 patients.
Reviewer 1 was critical about lack of cross-validation and small sample.
Reviewer 2 questioned "prediction".

**RAG Context (Relevant Literature):**
{context}

**Input Data:**

--- PAPER TEXT START ---
{paper_text[:60000]} 
--- PAPER TEXT END ---

--- FIRST ROUND REVIEWS START ---
{past_reviews}
--- FIRST ROUND REVIEWS END ---

**Instruction:**
1. Critically evaluate if the text provided (which is the REVISED version) has addressed the comments.
   - Did they add Cross-Validation (LOOCV)?
   - Did they compare to clinical predictors?
   - Did they reframe as "Exploratory"?
   - Did they add the Limitations?
   
2. Based on your evaluation, fill out the following form STRICTLY.

**Output Format:**
Please output ONLY the filled form below.

---
We strive to accept only the top 10%. Please be sure your written review reflects your percentage selections.

1) Originality of the paper:
[Select one: Top 5% / Top 10% / Top 25% / Top 50% / Bottom 50%]

2) Overall scientific quality of the paper (including methodology):
[Select one: Top 5% / Top 10% / Top 25% / Top 50% / Bottom 50%]

3) Priority Rating of this paper, based on impact on the field:
[Select one: High / Medium / Low]

3a) Would the priority rating increase if the paper was revised?
[Select one: Yes / No - (Explain briefly if needed)]

4) If this manuscript is revised and sent out for re-review, would you be willing to re-review it?
[Select one: Yes / No]

Neuropsychopharmacology (NPP) is a member of the Neuroscience Peer Review Consortium. If the manuscript is rejected, at the author’s request, NPP will send reviewer comments to the editorial office of the new journal. Do you consent to your identity being shared with the editorial office as well?
[Select one: Yes / No]

Recommendation:
[Select one: Accept / Minor Revisions / Major Revisions / Reject]

I have no conflict of interest - financial or otherwise - with the paper I have just reviewed.
[Confirmed]

Confidential remarks for the Editor/Publisher:
[Write a private note to the editor. Be direct, authoritative, and decisive.]

Remarks to be sent to the author:
[Write the formal review here. Acknowledge revisions, but be rigorous about the remaining flaws (sample size N=54). Decide if "Exploratory" is enough for NPP. If Rejecting, explain why with high scientific authority.]
---
"""
        response = self.model.generate_content(prompt)
        return response.text

async def main():
    print("Reading files...")
    with open(PAPER_TEXT_PATH, 'r') as f:
        paper_text = f.read()
    
    with open(REVIEWS_PATH, 'r') as f:
        reviews_text = f.read()
        
    reviewer = RAGReviewer()
    print("Generating review with Antigravity (Gemini 3 Pro Preview)...")
    review = await reviewer.generate_review(paper_text, reviews_text)
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(review)
    print(f"Review saved to {OUTPUT_PATH}")
    print(review)

if __name__ == "__main__":
    asyncio.run(main())
