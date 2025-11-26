#!/usr/bin/env python3
"""Multi-agent peer review system using GPT-4, Claude, and Gemini.

This script conducts a professional peer review using three independent LLMs
and synthesizes their evaluations into a comprehensive review.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

# Initialize API clients
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text
    """
    text = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)


def load_review_prompt(prompt_path: str) -> str:
    """Load the peer review prompt.

    Args:
        prompt_path: Path to prompt file

    Returns:
        Prompt text
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


async def review_with_gpt4(prompt: str, paper_text: str) -> Dict:
    """Conduct review using GPT-4.

    Args:
        prompt: Review prompt
        paper_text: Paper text content

    Returns:
        Review results
    """
    print("\n🤖 GPT-4 is reviewing...")

    try:
        # Truncate paper if too long
        max_chars = 40000
        if len(paper_text) > max_chars:
            paper_text = paper_text[:max_chars] + "\n\n[... truncated for length ...]"

        full_prompt = f"{prompt}\n\n---\n\nPAPER TEXT:\n\n{paper_text}"

        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert peer reviewer for Neuropsychopharmacology with deep expertise in neuroimaging, OCD research, and biomarker studies."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=8000
        )

        review_text = response.choices[0].message.content

        return {
            "model": "GPT-4",
            "review": review_text,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        print(f"❌ GPT-4 error: {e}")
        return {
            "model": "GPT-4",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "success": False
        }


async def review_with_claude(prompt: str, paper_text: str) -> Dict:
    """Conduct review using Claude.

    Args:
        prompt: Review prompt
        paper_text: Paper text content

    Returns:
        Review results
    """
    print("\n🤖 Claude is reviewing...")

    try:
        # Truncate paper if too long
        max_chars = 40000
        if len(paper_text) > max_chars:
            paper_text = paper_text[:max_chars] + "\n\n[... truncated for length ...]"

        full_prompt = f"{prompt}\n\n---\n\nPAPER TEXT:\n\n{paper_text}"

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

        review_text = response.content[0].text

        return {
            "model": "Claude Sonnet 4.5",
            "review": review_text,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        print(f"❌ Claude error: {e}")
        return {
            "model": "Claude Sonnet 4.5",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "success": False
        }


async def review_with_gemini(prompt: str, paper_text: str) -> Dict:
    """Conduct review using Gemini.

    Args:
        prompt: Review prompt
        paper_text: Paper text content

    Returns:
        Review results
    """
    print("\n🤖 Gemini is reviewing...")

    try:
        # Truncate paper if too long
        max_chars = 30000  # Gemini has smaller context
        if len(paper_text) > max_chars:
            paper_text = paper_text[:max_chars] + "\n\n[... truncated for length ...]"

        full_prompt = f"{prompt}\n\n---\n\nPAPER TEXT:\n\n{paper_text}"

        model = genai.GenerativeModel('gemini-pro')

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=8000
            )
        )

        review_text = response.text

        return {
            "model": "Gemini Pro",
            "review": review_text,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return {
            "model": "Gemini Pro",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "success": False
        }


def synthesize_reviews(reviews: List[Dict]) -> str:
    """Synthesize multiple agent reviews into final review.

    Args:
        reviews: List of review results from different models

    Returns:
        Synthesized review text
    """
    successful_reviews = [r for r in reviews if r.get('success', False)]

    if not successful_reviews:
        return "ERROR: All review agents failed. No reviews to synthesize."

    synthesis = []
    synthesis.append("# MULTI-AGENT PEER REVIEW SYNTHESIS")
    synthesis.append("=" * 80)
    synthesis.append(f"\n**Review Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    synthesis.append(f"**Number of Agents:** {len(successful_reviews)}")
    synthesis.append(f"**Models Used:** {', '.join([r['model'] for r in successful_reviews])}")
    synthesis.append("\n" + "=" * 80 + "\n")

    # Include each agent's review
    for i, review in enumerate(successful_reviews, 1):
        synthesis.append(f"\n## AGENT {i}: {review['model']}")
        synthesis.append("-" * 80)
        synthesis.append(review['review'])
        synthesis.append("\n" + "=" * 80 + "\n")

    # Add meta-analysis section
    synthesis.append("\n## META-ANALYSIS: Cross-Agent Consensus")
    synthesis.append("-" * 80)
    synthesis.append("\n**Instructions for Final Synthesis:**")
    synthesis.append("1. Identify consensus areas where all agents agree")
    synthesis.append("2. Identify divergence areas where agents disagree")
    synthesis.append("3. Extract the most critical concerns across all agents")
    synthesis.append("4. Combine recommendations into actionable list")
    synthesis.append("5. Provide unified final recommendation\n")

    return "\n".join(synthesis)


async def main():
    """Main execution function."""

    print("=" * 80)
    print("MULTI-AGENT PEER REVIEW SYSTEM")
    print("Neuropsychopharmacology Manuscript MS 37480")
    print("=" * 80)

    # File paths
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "input" / "37480_0_merged_1758766294.pdf"
    prompt_path = base_dir / "data" / "2025-10-npp" / "PEER_REVIEW_PROMPT.md"
    output_path = base_dir / "output" / f"MULTI_AGENT_REVIEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Check files exist
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF not found at {pdf_path}")
        return

    if not prompt_path.exists():
        print(f"❌ ERROR: Prompt not found at {prompt_path}")
        return

    print(f"\n📄 Paper: {pdf_path.name}")
    print(f"📋 Prompt: {prompt_path.name}")
    print(f"💾 Output: {output_path.name}")

    # Extract PDF text
    print("\n📖 Extracting paper text...")
    try:
        paper_text = extract_pdf_text(str(pdf_path))
        print(f"✅ Extracted {len(paper_text)} characters")
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return

    # Load prompt
    print("\n📝 Loading review prompt...")
    try:
        prompt = load_review_prompt(str(prompt_path))
        print(f"✅ Loaded prompt ({len(prompt)} characters)")
    except Exception as e:
        print(f"❌ Prompt loading error: {e}")
        return

    # Conduct reviews in parallel
    print("\n" + "=" * 80)
    print("PHASE 1: INDEPENDENT AGENT REVIEWS")
    print("=" * 80)

    tasks = [
        review_with_gpt4(prompt, paper_text),
        review_with_claude(prompt, paper_text),
        review_with_gemini(prompt, paper_text)
    ]

    reviews = await asyncio.gather(*tasks)

    # Check results
    successful = [r for r in reviews if r.get('success', False)]
    failed = [r for r in reviews if not r.get('success', False)]

    print("\n" + "=" * 80)
    print(f"✅ Successful reviews: {len(successful)}")
    print(f"❌ Failed reviews: {len(failed)}")

    if failed:
        for f in failed:
            print(f"   - {f['model']}: {f.get('error', 'Unknown error')}")

    # Synthesize reviews
    print("\n" + "=" * 80)
    print("PHASE 2: SYNTHESIS")
    print("=" * 80)

    synthesis = synthesize_reviews(reviews)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(synthesis)

    print(f"\n✅ Multi-agent review saved to: {output_path}")
    print("\n" + "=" * 80)
    print("REVIEW COMPLETE")
    print("=" * 80)

    # Also save to Reviews repo
    reviews_output = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp") / f"MULTI_AGENT_REVIEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(reviews_output, 'w', encoding='utf-8') as f:
        f.write(synthesis)
    print(f"\n✅ Copy saved to Reviews repo: {reviews_output}")


if __name__ == "__main__":
    asyncio.run(main())
