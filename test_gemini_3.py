import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3-flash-preview")

print("Testing Gemini 3...")
try:
    response = model.generate_content("Explain the significance of SciBERT in one sentence.")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
