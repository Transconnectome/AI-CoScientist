
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")

if not api_key:
    exit(1)

print(f"Configuring Gemini with key: {api_key[:5]}...")
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    print("Generating content with gemini-2.5-pro...")
    response = model.generate_content("Hello, are you working?")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
