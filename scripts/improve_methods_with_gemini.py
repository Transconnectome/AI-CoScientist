
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/juke/git/AI-CoScientist/.env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Try finding it in process env
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=api_key)

# Model selection
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
print(f"Using model: {model_name}")

model = genai.GenerativeModel(model_name)

# Read input file
input_path = "/home/juke/.gemini/antigravity/brain/146745b7-3916-4a29-942a-cc808885831f/research/methods_refined.tex"
with open(input_path, "r") as f:
    methods_text = f.read()

# Prompt (adapted from AI-CoScientist)
prompt = f"""You are an expert scientific editor for MICCAI (Medical Image Computing and Computer Assisted Intervention). 
Improve this methods section to be more rigorous, mathematically precise, and persuasive.

REQUIREMENTS:
1. Add detailed implementation details for reproducibility (e.g., specific hyperparameters if implied).
2. Enhance mathematical formulation (ensure defined variables and dimensions).
3. strengthen the "Subject-Conditioned Learning" explanation.
4. Improve clarity with subsections.
5. Emphasize the novelty of "Atlas-Guided Masking" vs random masking.

CURRENT METHODS:
{methods_text}

OUTPUT: Provide ONLY the improved methods text in LaTeX format. Do not include markdown code blocks."""

print("Generating improvements...")
response = model.generate_content(prompt)
improved_text = response.text

# Save output
output_path = "/home/juke/.gemini/antigravity/brain/146745b7-3916-4a29-942a-cc808885831f/research/methods_ai_improved.tex"
with open(output_path, "w") as f:
    f.write(improved_text)

print(f"Saved improved text to {output_path}")
