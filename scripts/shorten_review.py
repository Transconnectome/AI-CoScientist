#!/usr/bin/env python3
"""Shorten review to realistic peer review length (1/5 of current)."""

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


def shorten_review(review_path: str, prompt_path: str, output_path: str):
    """Shorten review to realistic length by removing verbose details.

    Args:
        review_path: Path to original review
        prompt_path: Path to shortening prompt
        output_path: Path for shortened review
    """
    print("=" * 80)
    print("SHORTENING REVIEW TO REALISTIC LENGTH")
    print("=" * 80)

    # Read original review
    print(f"\n📄 Reading original review: {Path(review_path).name}")
    with open(review_path, 'r', encoding='utf-8') as f:
        original_review = f.read()
    print(f"✅ Loaded {len(original_review):,} characters")

    # Read shortening prompt
    print(f"\n📋 Reading shortening prompt: {Path(prompt_path).name}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        shortening_prompt = f.read()
    print(f"✅ Loaded {len(shortening_prompt):,} characters")

    # Combine prompt and review
    full_prompt = f"{shortening_prompt}\n\n## ORIGINAL REVIEW TO SHORTEN\n\n{original_review}"

    print("\n🤖 Sending to Claude for shortening...")
    print("⏳ This may take 2-3 minutes...")
    print(f"🎯 Target: Reduce from ~{len(original_review):,} to ~{len(original_review)//5:,} chars")

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,  # Shorter output expected
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        shortened_review = response.content[0].text

        print(f"✅ Received shortened review ({len(shortened_review):,} characters)")

        # Save shortened review
        print(f"\n💾 Saving shortened review: {Path(output_path).name}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(shortened_review)

        print(f"✅ Saved to: {output_path}")

        # Also save to Reviews repo
        reviews_output = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp") / Path(output_path).name
        with open(reviews_output, 'w', encoding='utf-8') as f:
            f.write(shortened_review)

        print(f"✅ Copy saved to: {reviews_output}")

        print("\n" + "=" * 80)
        print("SHORTENING COMPLETE")
        print("=" * 80)

        # Print summary of changes
        reduction_pct = (1 - len(shortened_review) / len(original_review)) * 100
        print("\n📊 Summary:")
        print(f"   Original length: {len(original_review):,} chars")
        print(f"   Shortened length: {len(shortened_review):,} chars")
        print(f"   Reduction: {reduction_pct:.1f}%")
        print(f"   Target was: {100/5:.1f}% reduction (1/5 length)")

        if reduction_pct >= 75:
            print(f"   ✅ Target achieved!")
        else:
            print(f"   ⚠️  Target not fully achieved (need {80-reduction_pct:.1f}% more reduction)")

        return shortened_review

    except Exception as e:
        print(f"\n❌ Error during shortening: {e}")
        raise


def main():
    """Main execution."""

    # File paths
    base_dir = Path(__file__).parent.parent
    review_path = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp/PROPER_REVIEW_20251026_230819.md")
    prompt_path = base_dir / "data" / "2025-10-npp" / "SHORTEN_REVIEW_PROMPT.md"
    output_path = base_dir / "output" / f"FINAL_REVIEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Check files exist
    if not review_path.exists():
        print(f"❌ ERROR: Review not found at {review_path}")
        return

    if not prompt_path.exists():
        print(f"❌ ERROR: Prompt not found at {prompt_path}")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Shorten review
    shorten_review(str(review_path), str(prompt_path), str(output_path))


if __name__ == "__main__":
    main()
