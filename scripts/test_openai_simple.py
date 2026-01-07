import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

async def test_openai_direct():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key")
        return

    client = AsyncOpenAI(api_key=api_key)
    
    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    for model in models:
        print(f"\nTesting model: {model}")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print(f"SUCCESS: {response.choices[0].message.content}")
            return # Stop after first success
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_openai_direct())






