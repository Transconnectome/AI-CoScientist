import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai

def list_models():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY")
        return

    print(f"Listing models with key: {api_key[:10]}...")
    genai.configure(api_key=api_key)
    
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    list_models()
