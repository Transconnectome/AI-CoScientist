#!/usr/bin/env python3
"""
QuantERA QML-RAPTOR Quick Demo
Demonstrates the system capabilities with minimal setup
"""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agent import QuantERAAgent


def demo_agent_capabilities():
    """Demonstrate the agent's research capabilities"""
    print("🚀 QuantERA QML-RAPTOR System Demo")
    print("=" * 50)

    # Initialize agent
    print("📚 Initializing QuantERA Research Agent...")
    try:
        agent = QuantERAAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("💡 Try running: python setup.py")
        return

    # Check system status
    print("\n🔍 Checking system status...")
    status = agent.get_system_status()
    print(f"Status: {status['status']}")

    if status['status'] != 'operational':
        print("⚠️  System not fully operational. Some features may be limited.")
        print("💡 For full functionality, add papers using the test system.")

    # Demo queries
    demo_queries = [
        "What is a variational quantum eigensolver?",
        "How does barren plateau affect quantum optimization?",
        "Compare quantum and classical machine learning"
    ]

    print(f"\n🤖 Testing {len(demo_queries)} sample queries...")
    print("=" * 50)

    for i, query in enumerate(demo_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 30)

        try:
            response = agent.query(query)

            print(f"💬 Answer: {response.answer[:300]}{'...' if len(response.answer) > 300 else ''}")
            print(f"🎯 Confidence: {response.confidence:.1%}")
            print(f"📚 Sources: {len(response.sources)}")

            if response.follow_up_suggestions:
                print(f"🔍 Suggested follow-ups:")
                for suggestion in response.follow_up_suggestions[:2]:
                    print(f"   • {suggestion}")

        except Exception as e:
            print(f"❌ Query failed: {e}")

        print()

    # Interactive demo
    print("🎮 Interactive Demo")
    print("=" * 30)
    print("Ask your own quantum ML research questions!")
    print("Type 'quit' to exit, or 'help' for assistance.")

    while True:
        try:
            user_query = input("\n❓ Your question: ").strip()

            if not user_query:
                continue

            if user_query.lower() in ['quit', 'exit', 'q']:
                break

            if user_query.lower() == 'help':
                print("\n💡 Sample questions you can ask:")
                print("   • What is quantum machine learning?")
                print("   • How do variational quantum algorithms work?")
                print("   • What are the challenges with NISQ devices?")
                print("   • Compare different quantum optimization methods")
                print("   • What is quantum advantage in machine learning?")
                continue

            # Process user query
            response = agent.query(user_query)

            print(f"\n💭 Research Assistant Response:")
            print(f"   {response.answer}")
            print(f"\n📊 Confidence: {response.confidence:.1%}")

            if response.follow_up_suggestions:
                print(f"\n🔗 You might also ask:")
                for suggestion in response.follow_up_suggestions[:3]:
                    print(f"   • {suggestion}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")

    print("\n👋 Thanks for trying QuantERA QML-RAPTOR!")
    print("🚀 For full system capabilities, run: python test_system.py")


if __name__ == "__main__":
    # Setup minimal logging
    logging.basicConfig(level=logging.WARNING)

    # Run demo
    demo_agent_capabilities()