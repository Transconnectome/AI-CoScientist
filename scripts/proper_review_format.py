#!/usr/bin/env python3
"""Convert remarks to proper peer review format (third-person evaluation)."""

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


def convert_to_proper_review(remarks_path: str, prompt_path: str, output_path: str):
    """Convert letter-to-authors format to proper third-person peer review.

    Args:
        remarks_path: Path to original remarks
        prompt_path: Path to conversion prompt
        output_path: Path for properly formatted review
    """
    print("=" * 80)
    print("CONVERTING TO PROPER PEER REVIEW FORMAT")
    print("=" * 80)

    # Read original remarks
    print(f"\n📄 Reading original remarks: {Path(remarks_path).name}")
    with open(remarks_path, 'r', encoding='utf-8') as f:
        original_remarks = f.read()
    print(f"✅ Loaded {len(original_remarks)} characters")

    # Read conversion prompt
    print(f"\n📋 Reading conversion prompt: {Path(prompt_path).name}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        conversion_prompt = f.read()
    print(f"✅ Loaded {len(conversion_prompt)} characters")

    # Combine prompt and remarks
    full_prompt = f"{conversion_prompt}\n\n## ORIGINAL DOCUMENT TO TRANSFORM\n\n{original_remarks}"

    print("\n🤖 Sending to Claude for conversion...")
    print("⏳ This may take 2-3 minutes due to length...")

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,  # Large output needed
            temperature=0.2,   # Low temperature for precise format conversion
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        proper_review = response.content[0].text

        print(f"✅ Received proper review ({len(proper_review)} characters)")

        # Save proper review
        print(f"\n💾 Saving proper review: {Path(output_path).name}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(proper_review)

        print(f"✅ Saved to: {output_path}")

        # Also save to Reviews repo
        reviews_output = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp") / Path(output_path).name
        with open(reviews_output, 'w', encoding='utf-8') as f:
            f.write(proper_review)

        print(f"✅ Copy saved to: {reviews_output}")

        print("\n" + "=" * 80)
        print("CONVERSION COMPLETE")
        print("=" * 80)

        # Print summary of changes
        print("\n📊 Summary:")
        print(f"   Original length: {len(original_remarks):,} chars")
        print(f"   Converted length: {len(proper_review):,} chars")
        print(f"   Difference: {len(proper_review) - len(original_remarks):+,} chars")

        # Check for letter format markers
        has_dear_authors = "Dear Authors" in original_remarks
        removed_dear_authors = "Dear Authors" not in proper_review
        print(f"\n   'Dear Authors' removed: {has_dear_authors} → {removed_dear_authors}")

        original_you_count = original_remarks.lower().count(" you ")
        converted_you_count = proper_review.lower().count(" you ")
        print(f"   Second-person 'you' reduced: {original_you_count} → {converted_you_count}")

        return proper_review

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        raise


def main():
    """Main execution."""

    # File paths - use the HUMANIZED version as input
    base_dir = Path(__file__).parent.parent
    remarks_path = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp/HUMANIZED_REMARKS_20251026_230118.md")
    prompt_path = base_dir / "data" / "2025-10-npp" / "PROPER_REVIEW_FORMAT_PROMPT.md"
    output_path = base_dir / "output" / f"PROPER_REVIEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Check files exist
    if not remarks_path.exists():
        print(f"❌ ERROR: Remarks not found at {remarks_path}")
        return

    if not prompt_path.exists():
        print(f"❌ ERROR: Prompt not found at {prompt_path}")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to proper review format
    convert_to_proper_review(str(remarks_path), str(prompt_path), str(output_path))


if __name__ == "__main__":
    main()
