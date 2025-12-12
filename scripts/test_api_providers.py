#!/usr/bin/env python3
"""Quick test script to check API provider status."""

import asyncio
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_golden_references_advanced import (
    AnthropicProvider,
    OpenAIProvider,
    DeepSeekProvider,
    GeminiProvider,
    LLMProvider
)

async def test_provider(provider_class, name, api_key_env):
    """Test a single provider."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    # Check if API key exists
    api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"✗ No API key found for {api_key_env}")
        return False

    try:
        provider = provider_class(api_key=api_key)
        print(f"✓ Provider initialized")

        # Try a simple generation
        response, used_provider = await provider.generate(
            prompt="Say 'API works' in exactly two words.",
            max_tokens=10,
            temperature=0.0
        )

        print(f"✓ API call successful")
        print(f"  Response: {response[:100]}")
        print(f"  Provider: {used_provider.value}")
        return True

    except Exception as e:
        print(f"✗ Failed: {str(e)[:200]}")
        return False

async def main():
    """Test all providers."""
    print("API Provider Status Check")
    print("="*60)

    results = {}

    # Test each provider
    providers = [
        (AnthropicProvider, "Anthropic (Claude)", "ANTHROPIC_API_KEY"),
        (OpenAIProvider, "OpenAI (GPT)", "OPENAI_API_KEY"),
        (DeepSeekProvider, "DeepSeek", "DEEPSEEK_API_KEY"),
        (GeminiProvider, "Google Gemini", "GEMINI_API_KEY"),
    ]

    for provider_class, name, api_key_env in providers:
        results[name] = await test_provider(provider_class, name, api_key_env)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    working = [name for name, status in results.items() if status]
    failed = [name for name, status in results.items() if not status]

    print(f"\n✓ Working providers ({len(working)}):")
    for name in working:
        print(f"  - {name}")

    if failed:
        print(f"\n✗ Failed providers ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
