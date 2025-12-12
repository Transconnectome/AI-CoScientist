#!/usr/bin/env python3
"""Humanize reviewer remarks to remove AI-generated formatting."""

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


def humanize_remarks(remarks_path: str, prompt_path: str, output_path: str):
    """Transform remarks from AI-formatted to natural academic prose.

    Args:
        remarks_path: Path to original remarks
        prompt_path: Path to humanization prompt
        output_path: Path for humanized remarks
    """
    print("=" * 80)
    print("HUMANIZING REVIEWER REMARKS")
    print("=" * 80)

    # Read original remarks
    print(f"\n📄 Reading original remarks: {Path(remarks_path).name}")
    with open(remarks_path, 'r', encoding='utf-8') as f:
        original_remarks = f.read()
    print(f"✅ Loaded {len(original_remarks)} characters")

    # Read humanization prompt
    print(f"\n📋 Reading humanization prompt: {Path(prompt_path).name}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        humanization_prompt = f.read()
    print(f"✅ Loaded {len(humanization_prompt)} characters")

    # Combine prompt and remarks
    full_prompt = f"{humanization_prompt}\n\n## ORIGINAL REMARKS TO TRANSFORM\n\n{original_remarks}"

    print("\n🤖 Sending to Claude for humanization...")
    print("⏳ This may take 2-3 minutes due to length...")

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16000,  # Large output needed
            temperature=0.3,   # Some creativity for natural prose
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        humanized_remarks = response.content[0].text

        print(f"✅ Received humanized remarks ({len(humanized_remarks)} characters)")

        # Save humanized remarks
        print(f"\n💾 Saving humanized remarks: {Path(output_path).name}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(humanized_remarks)

        print(f"✅ Saved to: {output_path}")

        # Also save to Reviews repo
        reviews_output = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp") / Path(output_path).name
        with open(reviews_output, 'w', encoding='utf-8') as f:
            f.write(humanized_remarks)

        print(f"✅ Copy saved to: {reviews_output}")

        print("\n" + "=" * 80)
        print("HUMANIZATION COMPLETE")
        print("=" * 80)

        # Print summary of changes
        print("\n📊 Summary:")
        print(f"   Original length: {len(original_remarks):,} chars")
        print(f"   Humanized length: {len(humanized_remarks):,} chars")
        print(f"   Difference: {len(humanized_remarks) - len(original_remarks):+,} chars")

        # Count bullet points in original vs humanized
        original_bullets = original_remarks.count('\n- ') + original_remarks.count('\n1. ') + original_remarks.count('\n2. ')
        humanized_bullets = humanized_remarks.count('\n- ') + humanized_remarks.count('\n1. ') + humanized_remarks.count('\n2. ')
        print(f"   Bullet/list items reduced: {original_bullets} → {humanized_bullets}")

        return humanized_remarks

    except Exception as e:
        print(f"\n❌ Error during humanization: {e}")
        raise


def main():
    """Main execution."""

    # File paths
    base_dir = Path(__file__).parent.parent
    remarks_path = Path("/Users/jiookcha/Documents/git/Reviews/data/2025-10-npp/REMARKS_TO_AUTHORS.md")
    prompt_path = base_dir / "data" / "2025-10-npp" / "HUMANIZE_REMARKS_PROMPT.md"
    output_path = base_dir / "output" / f"HUMANIZED_REMARKS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Check files exist
    if not remarks_path.exists():
        print(f"❌ ERROR: Remarks not found at {remarks_path}")
        return

    if not prompt_path.exists():
        print(f"❌ ERROR: Prompt not found at {prompt_path}")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Humanize remarks
    humanize_remarks(str(remarks_path), str(prompt_path), str(output_path))


if __name__ == "__main__":
    main()
