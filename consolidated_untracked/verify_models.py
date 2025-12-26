import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load .env.local explicitly
load_dotenv(project_root / ".env.local")

from src.core.config import settings
from src.services.llm.adapters.openai import OpenAIAdapter
from src.services.llm.adapters.anthropic import AnthropicAdapter
from src.services.llm.adapters.gemini import GeminiAdapter
from src.services.llm.types import LLMRequest, TaskType, LLMConfig, ModelProvider

async def test_adapter(name, adapter_cls, api_key_attr, model_name):
    print(f"\n--- Testing {name} Adapter ({model_name}) ---")
    
    api_key = getattr(settings, api_key_attr, None)
    if not api_key:
        print(f"❌ SKIPPED: {api_key_attr} not found in settings.")
        return

    try:
        adapter = adapter_cls(api_key=api_key)
        req = LLMRequest(
            prompt="Hello, are you functional?",
            task_type=TaskType.HYPOTHESIS_GENERATION,
            config=LLMConfig(
                provider=ModelProvider.OPENAI if "open" in name.lower() else (ModelProvider.ANTHROPIC if "anth" in name.lower() else ModelProvider.GOOGLE),
                model=model_name,
                temperature=0.7,
                max_tokens=50
            )
        )
        print(f"Sending request to {model_name}...")
        response = await adapter.complete(req)
        print(f"✅ SUCCESS: {response.content}")
        print(f"Cost: ${response.cost}")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

async def main():
    print("Starting Model Verification for 2025 Premium Models...")
    
    tasks = [
        ("OpenAI (GPT-5)", OpenAIAdapter, "openai_api_key", "gpt-5"),
        ("Gemini (Configured)", GeminiAdapter, "google_api_key", settings.gemini_model),
        ("Gemini (Requested)", GeminiAdapter, "google_api_key", "gemini-3-pro"),
        ("Anthropic", AnthropicAdapter, "anthropic_api_key", settings.anthropic_model)
    ]

    for name, adapter_cls, api_key_attr, model_name in tasks:
        try:
             await test_adapter(name, adapter_cls, api_key_attr, model_name)
        except Exception as e:
             print(f"❌ CRITICAL ERROR testing {name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
