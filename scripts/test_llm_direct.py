import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("Starting script...")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
print("Added path...")

try:
    from src.services.llm.adapters.openai import OpenAIAdapter
    from src.services.llm.adapters.anthropic import AnthropicAdapter
    from src.services.llm.types import LLMRequest, TaskType
    print("Imports successful.")
except Exception as e:
    print(f"Imports failed: {e}")
    sys.exit(1)

async def test_openai(api_key):
    print("\nTesting OpenAI...")
    if not api_key:
        print("SKIP: OPENAI_API_KEY not found")
        return

    try:
        print("Initializing OpenAI adapter...")
        adapter = OpenAIAdapter(api_key=api_key)
        print("Creating request...")
        request = LLMRequest(
            prompt="Hello, are you working? Reply with 'Yes, OpenAI is working'.",
            task_type=TaskType.HYPOTHESIS_GENERATION
        )
        print("Sending request...")
        response = await adapter.complete(request)
        print(f"SUCCESS: {response.content}")
        print(f"Model used: {response.model}")
    except Exception as e:
        print(f"FAILED: {e}")

async def main():
    print("Loading dotenv...")
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    await test_openai(openai_key)

if __name__ == "__main__":
    asyncio.run(main())






