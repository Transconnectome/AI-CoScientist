
import asyncio
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

PAPER_TEXT_PATH = "/home/juke/git/AI-CoScientist/paper_text.txt"
GENERATED_REVIEW_PATH = "/home/juke/git/AI-CoScientist/data/final_npp_review_antigravity.md"
FIRST_ROUND_REVIEWS_PATH = "/home/juke/git/AI-CoScientist/data/first_round_reviews.txt"
OUTPUT_PATH = "/home/juke/git/AI-CoScientist/data/meta_review_critique.md"

# Meta-Reviewer Persona
META_REVIEWER_PERSONA = """
You are Dr. AI-CoScientist, the Senior Executive Editor of Neuropsychopharmacology (NPP) and a meta-science expert. 
Your role is to Quality Control other reviewers. You are extremely rigorous, objective, and critical. 
You do not tolerate "hallucinated" improvements. You need to verify if the reviewer's decision is scientifically sound.
"""

async def evaluate_review():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    genai.configure(api_key=api_key)
    # Using the most capable model for this high-level reasoning task
    model = genai.GenerativeModel("gemini-3-pro-preview") 
    
    print("Reading documents...")
    with open(PAPER_TEXT_PATH, 'r') as f:
        paper_text = f.read()
    with open(GENERATED_REVIEW_PATH, 'r') as f:
        generated_review = f.read()
    with open(FIRST_ROUND_REVIEWS_PATH, 'r') as f:
        past_reviews = f.read()

    prompt = f"""
{META_REVIEWER_PERSONA}

**Task:**
Evaluate the quality and scientific validity of the "Generated Review" provided below for a manuscript submitted to Neuropsychopharmacology.

**Context:**
The manuscript is a REVISION. 
The "Generated Review" (by reviewer 'Antigravity') recommends **ACCEPT**.
You must determine if this decision is defensible or if the reviewer is being too lenient/hallucinating improvements.

**Input Data:**

1. **The Revised Manuscript (Excerpt):**
--- START MANUSCRIPT ---
{paper_text[:40000]}
... [truncated]
--- END MANUSCRIPT ---

2. **First Round Reviewer Comments (The 'To-Do' List):**
--- START PAST REVIEWS ---
{past_reviews}
--- END PAST REVIEWS ---

3. **The Generated Review (To Be Evaluated):**
--- START GENERATED REVIEW ---
{generated_review}
--- END GENERATED REVIEW ---

**Evaluation Instructions:**
Critize the Generated Review on the following points:
1.  **Factuality:** Did the reviewer claim the authors added LOOCV? Check the manuscript text. Did they *actually* add it? Quote the text confirming or refuting this.
2.  **Soundness:** The reviewer accepts N=54 as "Exploratory" because it's a "drug-naive" cohort. Is this a valid scientific tradeoff for a top-tier journal like NPP (Top 10%)? Or is the reviewer lowering standards?
3.  **Completeness:** Did the reviewer miss any major unaddressed points from the First Round Reviews?

**Final Verdict:**
- **Grade the Review:** (A / B / C / F)
- **Decision Endorsement:** Do you agree with the "ACCEPT" recommendation? (Yes / No / Uncertain)
- **Correction:** If the review is flawed, write a corrected paragraph for the Confidential Remarks to the Editor.

**Output Format:**
Markdown report. Be concise but brutal.
"""
    print("Generating Meta-Review Critique...")
    response = model.generate_content(prompt)
    return response.text

async def main():
    critique = await evaluate_review()
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(critique)
    
    print(f"\n--- Meta-Review Critique ---\n{critique}\n-----------------------------")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
