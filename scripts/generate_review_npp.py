
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
OUTPUT_PATH = "/home/juke/git/AI-CoScientist/data/final_npp_review.md"
CHROMADB_PATH = "chromadb_data"

class RAGReviewer:
    def __init__(self):
        # LLM Setup
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp") 
        
        # RAG Setup
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('allenai/scibert_scivocab_uncased')
        self.client = chromadb.PersistentClient(path=CHROMADB_PATH)
        # We'll query L0 chunks for specific details and L2 for broad summaries if available
        # But L0 is usually best for "checking facts"
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
        # We'll use the abstract or introduction as the query
        query_text = paper_text[:2000] # First 2000 chars likely contain abstract/intro
        context = self.retrieve_context(query_text, n_results=15)
        
        prompt = f"""
You are the Editor-in-Chief of Neuropsychopharmacology (NPP) and a world-leading expert in OCD neuroimaging and clinical trials.
You are reviewing a revised manuscript titled "Sensorimotor circuit connectivity as a candidate biomarker for responsiveness to sertraline in obsessive-compulsive disorder".

Your task is to evaluate whether the authors have successfully addressed the First Round Reviews and to generate the FINAL REVIEW REPORT for the journal.

**Context:**
The paper claims to predict sertraline response using sensorimotor circuit fcMRI in N=54 patients.
Reviewer 1 was very critical about the lack of cross-validation and small sample size (N=54 < N=100 recommended by Marek 2025).
Reviewer 2 questioned the "prediction" terminology.
Reviewer 3 noted the redundancy between group stats and prediction.

**RAG Context (Relevant Literature):**
The following excerpts are from the references cited in the paper (and other relevant literature), retrieved from your internal knowledge base:
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
   - Did they add Cross-Validation (LOOCV)? (Search text for "LOOCV", "leave-one-out").
   - Did they compare to clinical predictors? (Search for "clinical-only model", "combined model", "DeLong").
   - Did they reframe as "Exploratory"? (Search for "exploratory", "candidate biomarker").
   - Did they add the Limitations? (Search for "Generalizability", "Single site").
   
2. Based on your evaluation, fill out the following form STRICTLY.

**Tone:**
Rigorous, scientific, fair but high-standard. NPP has a 10% acceptance rate.
If they addressed the major concerns (LOOCV, Clinical comparison, Tone shift), this might be acceptable as a "Candidate Biomarker" paper.
If they failed to do LOOCV or significant validation, it should be Rejected.

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
[Write a private note to the editor evaluating specific strengths/weaknesses and whether the revision was successful. Be blunt here.]

Remarks to be sent to the author:
[Write the formal review here. Acknowledge the extensive revisions. Point out any remaining minor issues or major flaws. If they fixed the LOOCV and Clinical comparison, praise that. If the sample size N=54 is still a hard limit, acknowledge it but accept it as "exploratory" if they framed it well.]
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
    print("Generating review with RAG...")
    review = await reviewer.generate_review(paper_text, reviews_text)
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(review)
    print(f"Review saved to {OUTPUT_PATH}")
    print(review)

if __name__ == "__main__":
    asyncio.run(main())
