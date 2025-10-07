#!/usr/bin/env python3
"""Script to generate automated reviews for papers.

Usage:
    python scripts/generate_review.py <paper_id>
    python scripts/generate_review.py <paper_id> --type journal
    python scripts/generate_review.py <paper_id> --output review.md
"""

import asyncio
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))


async def generate_paper_review(
    paper_id: str,
    review_type: str = "conference",
    output_path: str = None
):
    """Generate automated review for a paper.

    Args:
        paper_id: Paper UUID (string)
        review_type: Type of review (conference, journal, workshop)
        output_path: Optional path to save markdown output
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from src.core.config import settings
    from src.services.llm.service import LLMService
    from src.services.paper.review_generator import AutomatedReviewGenerator

    print("=" * 80)
    print("AUTOMATED REVIEW GENERATION")
    print("=" * 80)
    print(f"Paper ID: {paper_id}")
    print(f"Review Type: {review_type}")
    print()

    # Initialize services
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    llm_service = LLMService()

    async with async_session() as db:
        # Create review generator
        review_generator = AutomatedReviewGenerator(
            llm_service=llm_service,
            db=db,
            use_multitask=True  # Use multi-task model if trained
        )

        # Generate review
        print("🔍 Analyzing paper...")
        print("   - Multi-dimensional quality scoring")
        print("   - Qualitative analysis")
        print("   - Generating structured review")
        print()

        try:
            review = await review_generator.generate_review(
                paper_id=UUID(paper_id),
                review_type=review_type,
                include_recommendations=True
            )

            # Format as markdown
            markdown_review = review_generator.format_review_markdown(review)

            # Display review
            print("=" * 80)
            print("GENERATED REVIEW")
            print("=" * 80)
            print()
            print(markdown_review)
            print()

            # Save to file if requested
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, "w") as f:
                    f.write(markdown_review)

                print(f"✅ Review saved to: {output_file}")
            else:
                print("💡 Use --output <path> to save review to file")

            print()
            print("=" * 80)
            print("REVIEW STATISTICS")
            print("=" * 80)

            scores = review["scores"]
            print(f"Overall Quality:  {scores['overall']:.1f} / 10")
            print(f"Novelty:          {scores['novelty']:.1f} / 10")
            print(f"Methodology:      {scores['methodology']:.1f} / 10")
            print(f"Clarity:          {scores['clarity']:.1f} / 10")
            print(f"Significance:     {scores['significance']:.1f} / 10")
            print()

            review_content = review["review"]
            print(f"Strengths identified: {len(review_content.get('strengths', []))}")
            print(f"Weaknesses identified: {len(review_content.get('weaknesses', []))}")
            print(f"Questions for authors: {len(review_content.get('questions_for_authors', []))}")

            if "improvement_recommendations" in review_content:
                recommendations = review_content["improvement_recommendations"]
                print(f"Improvement recommendations: {len(recommendations)}")

                # Count by priority
                high_priority = sum(1 for r in recommendations if r.get("priority") == "high")
                print(f"  - High priority: {high_priority}")
                print(f"  - Medium/Low priority: {len(recommendations) - high_priority}")

            print()
            print(f"Recommendation: {review_content.get('recommendation', 'N/A').upper()}")
            print(f"Confidence: {review_content.get('confidence', 'N/A').capitalize()}")

            print()
            print("=" * 80)

        except ValueError as e:
            print(f"❌ Error: {e}")
            return

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return

    await engine.dispose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate automated review for a paper"
    )
    parser.add_argument(
        "paper_id",
        help="Paper UUID"
    )
    parser.add_argument(
        "--type",
        choices=["conference", "journal", "workshop"],
        default="conference",
        help="Type of review (default: conference)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for markdown review"
    )

    args = parser.parse_args()

    asyncio.run(generate_paper_review(
        paper_id=args.paper_id,
        review_type=args.type,
        output_path=args.output
    ))
