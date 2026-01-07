import os
from dotenv import load_dotenv

print("Loading dotenv...")
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"OPENAI_API_KEY found: {key[:5]}...")
else:
    print("OPENAI_API_KEY NOT found")






