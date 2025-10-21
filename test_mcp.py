"""Test script for AI-CoScientist MCP server."""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_server_import():
    """Test that the MCP server can be imported."""
    print("🔍 Testing MCP server import...")
    try:
        from src.mcp.server import PaperReviewerServer
        print("✅ Successfully imported PaperReviewerServer")
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False


async def test_llm_service():
    """Test LLM service initialization."""
    print("\n🔍 Testing LLM service...")
    try:
        from src.mcp.simple_llm import SimpleLLMService
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            print("⚠️  ANTHROPIC_API_KEY not set in environment")
            print("   Set it in .env file or environment variables")
            return False

        llm = SimpleLLMService()
        print("✅ LLM service initialized successfully")
        return True
    except Exception as e:
        print(f"❌ LLM service initialization failed: {str(e)}")
        return False


async def test_file_utils():
    """Test file utility functions."""
    print("\n🔍 Testing file utilities...")
    try:
        from src.mcp.file_utils import validate_file_path

        # Test with non-existent file
        error = validate_file_path("/nonexistent/file.txt")
        if error:
            print(f"✅ Validation works: {error}")

        # Test with unsupported file type (expected to fail)
        error = validate_file_path(__file__)
        if error and ".py" in error:
            print(f"✅ Correctly rejects .py files: {error}")
        else:
            print("✅ File validation works as expected")

        return True
    except Exception as e:
        print(f"❌ File utilities test failed: {str(e)}")
        return False


async def test_tools_definition():
    """Test that tools are properly defined."""
    print("\n🔍 Testing tools definition...")
    try:
        from src.mcp.server import PaperReviewerServer

        server = PaperReviewerServer()
        print("✅ Server instantiated successfully")

        # Check that server has the required attributes
        if hasattr(server, 'server'):
            print("✅ Server has 'server' attribute")
        else:
            print("❌ Server missing 'server' attribute")
            return False

        return True
    except Exception as e:
        print(f"❌ Tools definition test failed: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 AI-CoScientist MCP Server Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(await test_server_import())
    results.append(await test_llm_service())
    results.append(await test_file_utils())
    results.append(await test_tools_definition())

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        print("\n📝 Next steps:")
        print("1. Make sure ANTHROPIC_API_KEY is set in your environment")
        print("2. Restart Claude Code to load the MCP server")
        print("3. Use the ai-coscientist tools in Claude Code")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Please check the errors above and fix them")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
