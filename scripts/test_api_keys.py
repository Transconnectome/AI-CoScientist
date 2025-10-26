#!/usr/bin/env python3
"""
Test script to verify all API keys work correctly.
Tests: OpenAI (GPT-4), Anthropic (Claude), NVIDIA NGC
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_env_file(env_path: str):
    """Load environment variables from .env file"""
    if not os.path.exists(env_path):
        print(f"❌ Error: {env_path} not found")
        return False

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    return True


def test_openai_key():
    """Test OpenAI API key"""
    print("\n" + "="*60)
    print("🔑 Testing OpenAI API Key...")
    print("="*60)

    try:
        from openai import OpenAI

        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key or 'YOUR_' in api_key:
            print("❌ OpenAI API key not configured")
            return False

        print(f"API Key: {api_key[:20]}...{api_key[-10:]}")

        client = OpenAI(api_key=api_key)

        print("Sending test request to GPT-4...")
        response = client.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'user', 'content': 'Say "OpenAI API test successful" and nothing else.'}
            ],
            max_tokens=20
        )

        result = response.choices[0].message.content
        print(f"✅ OpenAI API works!")
        print(f"Response: {result}")
        print(f"Model: {response.model}")
        print(f"Tokens used: {response.usage.total_tokens}")
        return True

    except Exception as e:
        print(f"❌ OpenAI API test failed: {str(e)}")
        return False


def test_anthropic_key():
    """Test Anthropic API key"""
    print("\n" + "="*60)
    print("🔑 Testing Anthropic API Key...")
    print("="*60)

    try:
        import anthropic

        api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if not api_key or 'YOUR_' in api_key:
            print("❌ Anthropic API key not configured")
            return False

        print(f"API Key: {api_key[:20]}...{api_key[-10:]}")

        # Initialize client without proxies parameter (version compatibility)
        client = anthropic.Anthropic(api_key=api_key)

        print("Sending test request to Claude...")
        # Try claude-3-5-sonnet (most commonly available)
        response = client.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=30,
            messages=[
                {'role': 'user', 'content': 'Say "Anthropic API test successful" and nothing else.'}
            ]
        )

        result = response.content[0].text
        print(f"✅ Anthropic API works!")
        print(f"Response: {result}")
        print(f"Model: {response.model}")
        print(f"Tokens used: {response.usage.input_tokens + response.usage.output_tokens}")
        return True

    except Exception as e:
        print(f"❌ Anthropic API test failed: {str(e)}")
        # Provide more helpful error message
        if "api_key" in str(e).lower():
            print("   → API key may be invalid or revoked")
        elif "model" in str(e).lower():
            print("   → Model not available with this API key")
        else:
            print(f"   → Try testing at: https://console.anthropic.com/")
        return False


def test_ngc_key():
    """Test NVIDIA NGC API key (simulated - actual test requires NIM container)"""
    print("\n" + "="*60)
    print("🔑 Testing NVIDIA NGC API Key...")
    print("="*60)

    api_key = os.getenv('NGC_API_KEY', '')
    if not api_key or 'YOUR_' in api_key:
        print("❌ NGC API key not configured")
        return False

    print(f"API Key: {api_key[:20]}...{api_key[-10:]}")

    # NGC key validation
    if api_key.startswith('nvapi-') and len(api_key) > 40:
        print("✅ NGC API key format is valid")
        print("⚠️  Note: Full NGC validation requires running NIM containers")
        print("    Will be tested during Connectome deployment")
        return True
    else:
        print("❌ NGC API key format invalid (should start with 'nvapi-')")
        return False


def main():
    """Run all API key tests"""
    print("\n" + "╔"+"═"*58+"╗")
    print("║" + " "*15 + "API KEY VALIDATION SUITE" + " "*19 + "║")
    print("╚"+"═"*58+"╝")

    # Load .env.local
    env_path = Path(__file__).parent.parent / '.env.local'
    print(f"\nLoading environment from: {env_path}")

    if not load_env_file(str(env_path)):
        print("\n❌ Failed to load .env.local file")
        print("Make sure .env.local exists with your API keys")
        return False

    print("✅ Environment loaded successfully")

    # Run tests
    results = {
        'OpenAI': test_openai_key(),
        'Anthropic': test_anthropic_key(),
        'NGC': test_ngc_key(),
    }

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    for service, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{service:15} {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL API KEYS VALIDATED SUCCESSFULLY!")
        print("="*60)
        print("\n✅ Ready for Connectome deployment")
        print("\nNext step: Run deployment script")
        print("  ./scripts/deploy_to_connectome_hybrid.sh")
    else:
        print("⚠️  SOME API KEYS FAILED VALIDATION")
        print("="*60)
        print("\n❌ Please check failed API keys before deployment")
        print("\nTroubleshooting:")
        print("1. Verify API keys are correct (no typos)")
        print("2. Check if keys are active (not revoked)")
        print("3. Ensure sufficient API credits/quota")

    print("\n⚠️  SECURITY REMINDER:")
    print("   ROTATE ALL API KEYS AFTER THIS SESSION!")
    print("   - OpenAI: https://platform.openai.com/api-keys")
    print("   - Anthropic: https://console.anthropic.com/settings/keys")
    print("   - NGC: https://org.ngc.nvidia.com/setup/api-key")
    print("")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
