#!/usr/bin/env python3
"""
Rebuttal Letter Improvement Script with RAG

Specialized script for improving rebuttal letters (response to reviewers)
using RAG-enhanced suggestions.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import anthropic

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.embeddings import EmbeddingService
from src.services.rag import RAGManager


def load_text(filepath: str) -> str:
    """Load text from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_text(text: str, filepath: str):
    """Save text to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


async def improve_rebuttal_with_rag(
    rebuttal_text: str,
    rag_manager: RAGManager,
    api_key: str
) -> str:
    """
    Improve rebuttal letter using RAG-enhanced prompts.

    Args:
        rebuttal_text: Original rebuttal text
        rag_manager: RAG manager for pattern search
        api_key: Anthropic API key

    Returns:
        Improved rebuttal text
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Search for similar rebuttal improvements
    print(f"  🔍 Searching RAG for similar rebuttal improvements...")

    # Rebuttal-specific search
    rag_results = await rag_manager.search_similar_improvements(
        section="Rebuttal",
        problem_description="improve clarity, conciseness, and professionalism",
        current_scores={"clarity": 7.0, "overall_quality": 7.0},
        n_results=3
    )

    # Format RAG context
    rag_context = rag_manager.format_rag_context(rag_results) if rag_results else ""

    if rag_context:
        print(f"  ✅ Found {len(rag_results)} similar improvement patterns")
    else:
        print(f"  ℹ️  No similar patterns found (will learn from this improvement)")

    prompt = f"""You are an expert scientific editor specializing in rebuttal letters and reviewer responses.

TASK: Improve this rebuttal letter to reviewers

REBUTTAL-SPECIFIC REQUIREMENTS:
1. **Clarity**: Make responses crystal clear and easy to follow
2. **Conciseness**: Remove redundancy while keeping completeness
3. **Professional Tone**: Maintain respectful, constructive tone
4. **Structure**: Improve organization and flow
5. **Specificity**: Ensure all claims are backed by specific evidence
6. **Gratitude**: Maintain appropriate thankfulness without being excessive

CURRENT REBUTTAL:
{rebuttal_text[:8000]}  # First 8000 chars for context

{rag_context}

IMPROVEMENT STRATEGY:
1. **Opening**: Strengthen gratitude statements (brief, sincere)
2. **Responses**: Make each response more direct and clear
3. **Evidence**: Strengthen connections between claims and evidence
4. **Transitions**: Improve logical flow between responses
5. **Formatting**: Ensure consistent bullet formatting
6. **Tone**: Balance professionalism with warmth

SPECIFIC IMPROVEMENTS TO MAKE:
- Remove overly verbose "thank you" statements
- Consolidate repetitive explanations
- Strengthen topic sentences
- Add clear signposting ("First...", "Second...", "Finally...")
- Ensure each response directly addresses the reviewer's concern
- Make revised manuscript sections more prominent

OUTPUT: Provide the improved rebuttal letter. Maintain the overall structure but improve clarity, conciseness, and professionalism."""

    response = client.messages.create(
        model='claude-sonnet-4-5-20250929',
        max_tokens=8000,
        temperature=0.7,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response.content[0].text.strip()


async def main():
    print("=" * 70)
    print("📝 Rebuttal Letter Improvement with RAG")
    print("=" * 70)

    # Check API keys
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return 1

    openai_key = os.getenv('OPENAI_API_KEY')

    # Load rebuttal
    original_file = 'response_original.txt'
    if not os.path.exists(original_file):
        print(f"❌ Rebuttal file not found: {original_file}")
        return 1

    print(f"\n📄 Loading rebuttal: {original_file}")
    original_text = load_text(original_file)
    print(f"   Length: {len(original_text):,} characters")

    # Initialize RAG system
    print("\n🔧 Initializing RAG system...")
    rag_enabled = False
    rag_manager = None

    if openai_key:
        try:
            embedding_service = EmbeddingService(provider="openai", api_key=openai_key)
            rag_manager = RAGManager(
                embedding_service=embedding_service,
                chromadb_mode="auto"
            )
            rag_enabled = rag_manager.is_enabled()

            if rag_enabled:
                stats = rag_manager.get_statistics()
                print(f"   ✅ RAG System ready ({stats['client_type']} mode)")
                print(f"   📚 Stored patterns: {stats['total_patterns']}")
            else:
                print("   ⚠️  RAG System not available, using standard mode")

        except Exception as e:
            print(f"   ⚠️  RAG initialization failed: {e}")
            print("   Continuing without RAG features")
    else:
        print("   ℹ️  RAG disabled (no OpenAI API key)")

    # Improve rebuttal
    print("\n🔧 Improving rebuttal letter...")

    if rag_enabled and rag_manager:
        improved_text = await improve_rebuttal_with_rag(
            original_text,
            rag_manager,
            api_key
        )
    else:
        # Fallback without RAG
        print("⚠️  Using standard improvement (RAG not available)")
        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Improve this rebuttal letter for clarity, conciseness, and professionalism:

{original_text[:8000]}

Focus on:
1. Clear, direct responses
2. Professional tone
3. Better organization
4. Remove redundancy"""

        response = client.messages.create(
            model='claude-sonnet-4-5-20250929',
            max_tokens=8000,
            temperature=0.7,
            messages=[{'role': 'user', 'content': prompt}]
        )
        improved_text = response.content[0].text.strip()

    # Save improved version
    output_file = 'output/rebuttals/response_improved_rag.txt'
    save_text(improved_text, output_file)
    print(f"\n💾 Saved improved rebuttal: {output_file}")

    # Store pattern if RAG enabled
    if rag_enabled and rag_manager:
        print("\n💾 Storing improvement pattern in RAG...")
        await rag_manager.store_improvement_pattern(
            paper_id="rebuttal_response",
            section="Rebuttal",
            before_text=original_text[:5000],  # Store excerpt
            after_text=improved_text[:5000],
            scores_before={"overall_quality": 7.0, "clarity": 7.0},
            scores_after={"overall_quality": 8.0, "clarity": 8.5},
            strategy="clarity + conciseness + professional tone",
            field="neuroscience",
            paper_type="rebuttal_letter"
        )
        print("   ✅ Pattern stored for future rebuttals")

        stats = rag_manager.get_statistics()
        print(f"\n📚 RAG STATISTICS:")
        print(f"   Total patterns stored: {stats['total_patterns']}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ REBUTTAL IMPROVEMENT COMPLETE")
    print("=" * 70)
    print(f"\nOriginal length: {len(original_text):,} characters")
    print(f"Improved length: {len(improved_text):,} characters")
    print(f"\nOutput file: {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
