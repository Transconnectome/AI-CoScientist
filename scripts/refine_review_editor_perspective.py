#!/usr/bin/env python3
"""Refine peer review from editor perspective to ensure professional standards."""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API client
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def refine_review(review_path: str, prompt_path: str, output_path: str):
    """Refine review from editor perspective.

    Args:
        review_path: Path to original review
        prompt_path: Path to editor refinement prompt
        output_path: Path for refined review
    """
    print("=" * 80)
    print("EDITOR PERSPECTIVE REVIEW REFINEMENT")
    print("=" * 80)

    # Read original review
    print(f"\n📄 Reading original review: {Path(review_path).name}")
    with open(review_path, 'r', encoding='utf-8') as f:
        original_review = f.read()
    print(f"✅ Loaded {len(original_review)} characters")

    # Read refinement prompt
    print(f"\n📋 Reading refinement prompt: {Path(prompt_path).name}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        refinement_prompt = f.read()
    print(f"✅ Loaded {len(refinement_prompt)} characters")

    # Combine prompt and review
    full_prompt = f"{refinement_prompt}\n\n## ORIGINAL REVIEW TO REFINE\n\n{original_review}"

    print("\n🤖 Sending to Claude for editor refinement...")
    print("⏳ This may take 2-3 minutes due to length...")

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,  # Increased for full review
            temperature=0.2,   # Lower for precise refinement
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        refined_review = response.content[0].text

        print(f"✅ Received refined review ({len(refined_review)} characters)")

        # Save refined review
        print(f"\n💾 Saving refined review: {Path(output_path).name}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(refined_review)

        print(f"✅ Saved to: {output_path}")

        # Also save to Reviews repo
        reviews_output = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp") / Path(output_path).name
        with open(reviews_output, 'w', encoding='utf-8') as f:
            f.write(refined_review)

        print(f"✅ Copy saved to: {reviews_output}")

        print("\n" + "=" * 80)
        print("REFINEMENT COMPLETE")
        print("=" * 80)

        # Print summary of changes
        print("\n📊 Summary:")
        print(f"   Original length: {len(original_review):,} chars")
        print(f"   Refined length:  {len(refined_review):,} chars")
        print(f"   Difference:      {len(refined_review) - len(original_review):+,} chars")

        return refined_review

    except Exception as e:
        print(f"\n❌ Error during refinement: {e}")
        raise


def main():
    """Main execution."""

    # File paths
    base_dir = Path(__file__).parent.parent
    review_path = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp/MULTI_AGENT_REVIEW_20251026_190025.md")
    prompt_path = base_dir / "data" / "2025-10-npp" / "EDITOR_REFINEMENT_PROMPT.md"
    output_path = base_dir / "output" / f"REFINED_REVIEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Check files exist
    if not review_path.exists():
        print(f"❌ ERROR: Review not found at {review_path}")
        return

    if not prompt_path.exists():
        print(f"❌ ERROR: Prompt not found at {prompt_path}")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Refine review
    refine_review(str(review_path), str(prompt_path), str(output_path))


if __name__ == "__main__":
    main()
