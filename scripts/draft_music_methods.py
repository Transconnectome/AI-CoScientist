
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/juke/git/AI-CoScientist/.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
print(f"Using model: {model_name}")
model = genai.GenerativeModel(model_name)

prompt = """
You are an expert MICCAI paper writer. Write a new subsection for the "Methods" section titled "Music Decoding and Genre Classification".

CONTEXT:
We have added a new benchmark using the "Music Genre fMRI Dataset" (OpenNeuro ds003720).
- Data: 5 Subjects, 4800 total volumes per subject (approx), listening to 540 music clips (15s each) from 10 genres (GTZAN dataset: Pop, Rock, Jazz, etc.).
- Task: We fine-tune the Titans-Neuro foundation model (pre-trained on HCP/Biobank) to decode the musical genre from the fMRI signal.
- Architecture: We use the Semantic Decoder head (contrastive learning with audio embeddings) to map BOLD signals to the shared localized latent space of the music audio.

REQUIREMENTS:
1. Formal mathematical notation for the Cross-Modal Retrieval task (fMRI $x$ -> Audio $y$).
2. Mention the specific dataset parameters (N=5, 10 genres).
3. Explain the evaluation metric: Top-1 Classification Accuracy and Retrieval Rank.
4. Use standard LaTeX.

OUTPUT: Just the LaTeX subsection.
"""

print("Generating Music Methods...")
response = model.generate_content(prompt)
print(response.text)

with open("research/music_methods_draft.tex", "w") as f:
    f.write(response.text)
