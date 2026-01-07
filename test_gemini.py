import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai

async def test_gemini_direct():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY")
        return

    print(f"Testing Gemini with key: {api_key[:10]}...")
    genai.configure(api_key=api_key)
    
    models = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    
    for model_name in models:
        print(f"\nTesting model: {model_name}")
        try:
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async("Hello")
            print(f"SUCCESS: {response.text}")
            return
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini_direct())
