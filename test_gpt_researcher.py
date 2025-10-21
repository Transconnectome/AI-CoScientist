"""
Test GPT Researcher functionality

This script tests the GPT Researcher integration for literature review.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_basic_research():
    """Test basic research functionality."""
    print("=" * 60)
    print("Testing GPT Researcher Basic Functionality")
    print("=" * 60)

    # Check API keys
    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    print("\n1. Checking API Keys:")
    print(f"   TAVILY_API_KEY: {'✓ Set' if tavily_key else '✗ Missing'}")
    print(f"   OPENAI_API_KEY: {'✓ Set' if openai_key else '✗ Missing'}")

    if not tavily_key:
        print("\n❌ TAVILY_API_KEY is required for GPT Researcher")
        print("   Get your key from: https://tavily.com")
        print("   Add to .env file: TAVILY_API_KEY=your-key-here")
        return False

    if not openai_key:
        print("\n❌ OPENAI_API_KEY is required for GPT Researcher")
        return False

    # Try importing GPT Researcher
    print("\n2. Checking GPT Researcher Installation:")
    try:
        from gpt_researcher import GPTResearcher
        print("   ✓ GPT Researcher imported successfully")
    except ImportError as e:
        print(f"   ✗ GPT Researcher not installed: {e}")
        print("\n   Install with: pip install gpt-researcher")
        return False

    # Test our service wrapper
    print("\n3. Testing GPT Researcher Service:")
    try:
        from src.services.external.gpt_researcher_service import GPTResearcherService
        print("   ✓ GPTResearcherService imported successfully")

        # Initialize service
        service = GPTResearcherService()
        print("   ✓ Service initialized")

    except Exception as e:
        print(f"   ✗ Service initialization failed: {e}")
        return False

    # Run a simple research query
    print("\n4. Running Sample Research Query:")
    print("   Query: 'Recent advances in transformer models for NLP'")

    try:
        result = await service.systematic_literature_review(
            research_question="Recent advances in transformer models for NLP",
            domain="machine learning",
            depth="quick"
        )

        if result["success"]:
            print("   ✓ Research completed successfully")
            print(f"   - Found {result['num_sources']} sources")
            print(f"   - Report length: {len(result['report'])} characters")
            print("\n   First 500 characters of report:")
            print("   " + "-" * 56)
            print("   " + result['report'][:500].replace("\n", "\n   "))
            print("   " + "-" * 56)
            print("\n   Sources:")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"   {i}. {source}")
            if len(result['sources']) > 5:
                print(f"   ... and {len(result['sources']) - 5} more")
        else:
            print(f"   ✗ Research failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"   ✗ Research query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)
    return True


async def test_advanced_features():
    """Test advanced GPT Researcher features."""
    print("\n" + "=" * 60)
    print("Testing Advanced Features")
    print("=" * 60)

    try:
        from src.services.external.gpt_researcher_service import GPTResearcherService
        service = GPTResearcherService()

        # Test question decomposition
        print("\n1. Testing Question Decomposition:")
        question = "How can AI improve scientific research in neuroscience?"
        print(f"   Original question: {question}")

        subquestions = await service.decompose_research_question(
            question=question,
            num_subquestions=3
        )

        print(f"   Generated {len(subquestions)} sub-questions:")
        for i, sq in enumerate(subquestions, 1):
            print(f"   {i}. {sq}")

        print("\n✅ Advanced features working!")

    except Exception as e:
        print(f"\n✗ Advanced features test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    success = await test_basic_research()

    if success:
        await test_advanced_features()

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
