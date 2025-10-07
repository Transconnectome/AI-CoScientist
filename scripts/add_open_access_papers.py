#!/usr/bin/env python3
"""Add open access papers to validation dataset.

Processes papers from open_access_papers folder and merges with existing dataset.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text or None if failed
    """
    try:
        import PyPDF2

        text = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages[:50]:  # Limit to first 50 pages
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        return "\n\n".join(text)

    except Exception as e:
        print(f"   ⚠️  PDF extraction failed: {e}")
        return None


def extract_paper_info(text: str, filename: str) -> Dict[str, str]:
    """Extract title and abstract from paper text.

    Args:
        text: Full paper text
        filename: Original filename

    Returns:
        Dictionary with title, abstract, content
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Try to find title (usually in first few lines, all caps or title case)
    title = filename.replace('.pdf', '').replace('.docx', '').replace('_', ' ')
    for i, line in enumerate(lines[:10]):
        if len(line) > 20 and len(line) < 200:
            # Looks like a title
            if line[0].isupper() and not line.endswith('.'):
                title = line
                break

    # Try to find abstract
    abstract = ""
    abstract_keywords = ['abstract', 'summary', 'overview']

    for i, line in enumerate(lines[:100]):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in abstract_keywords):
            # Found abstract section, get next few paragraphs
            abstract_lines = []
            for j in range(i+1, min(i+20, len(lines))):
                if len(lines[j]) > 50:  # Substantial paragraph
                    abstract_lines.append(lines[j])
                    if len('\n'.join(abstract_lines)) > 500:  # Enough abstract
                        break
            abstract = '\n'.join(abstract_lines)
            break

    # Full content (limit to 20000 chars for processing)
    content = text[:20000]

    return {
        "title": title[:200],  # Limit title length
        "abstract": abstract[:1000] if abstract else "Abstract not found",
        "content": content
    }


async def score_paper_with_gpt4(
    paper_info: Dict[str, str],
    openai_client
) -> Dict[str, int]:
    """Score paper using GPT-4 across 5 dimensions.

    Args:
        paper_info: Dict with title, abstract, content
        openai_client: OpenAI client instance

    Returns:
        Dict with 5 quality scores (1-10)
    """
    prompt = f"""
    You are an expert peer reviewer for top-tier scientific journals.
    Evaluate this scientific paper across 5 dimensions on a 1-10 scale.

    Title: {paper_info['title']}

    Abstract: {paper_info['abstract']}

    Content Preview:
    {paper_info['content'][:3000]}

    Rate the paper on a scale of 1-10 for each dimension:

    1. **Overall Quality** (1-10):
       - 10: Top-tier journal publication (Nature, Science level)
       - 8: Excellent journal publication
       - 6: Good journal publication
       - 4: Needs major revision
       - 2: Reject

    2. **Novelty** (1-10):
       - 10: Groundbreaking new idea/method
       - 8: Significant advancement
       - 6: Incremental improvement
       - 4: Minor novelty
       - 2: Little to no novelty

    3. **Methodology** (1-10):
       - 10: Rigorous, flawless methodology
       - 8: Strong methodology
       - 6: Adequate methodology
       - 4: Methodological issues
       - 2: Serious flaws

    4. **Clarity** (1-10):
       - 10: Exceptionally clear and well-written
       - 8: Clear and well-organized
       - 6: Acceptable clarity
       - 4: Difficult to follow
       - 2: Very unclear

    5. **Significance** (1-10):
       - 10: Field-changing research
       - 8: Important contribution
       - 6: Meaningful contribution
       - 4: Limited impact
       - 2: Minimal impact

    Return ONLY a JSON object with integer scores:
    {{
        "overall": <1-10>,
        "novelty": <1-10>,
        "methodology": <1-10>,
        "clarity": <1-10>,
        "significance": <1-10>
    }}

    Be critical and realistic. Most papers should score 5-8.
    Return only valid JSON.
    """

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert peer reviewer for scientific journals."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        content = content.strip()

        scores = json.loads(content)

        # Validate scores
        required_keys = ["overall", "novelty", "methodology", "clarity", "significance"]
        for key in required_keys:
            if key not in scores:
                scores[key] = 5  # Default to middle
            else:
                # Ensure within 1-10 range
                scores[key] = max(1, min(10, int(scores[key])))

        return scores

    except Exception as e:
        print(f"   ⚠️  Scoring failed: {e}")
        # Return default scores
        return {
            "overall": 5,
            "novelty": 5,
            "methodology": 5,
            "clarity": 5,
            "significance": 5
        }


async def process_open_access_papers():
    """Process open access papers and add to dataset."""
    import os
    from openai import AsyncOpenAI

    print("=" * 80)
    print("PROCESSING OPEN ACCESS PAPERS")
    print("=" * 80)
    print()

    # Open access papers directory
    papers_dir = Path("/Users/jiookcha/Documents/git/AI-CoScientist/open_access_papers")

    if not papers_dir.exists():
        print(f"❌ Papers directory not found: {papers_dir}")
        return

    # Find all PDF files
    pdf_files = list(papers_dir.glob("*.pdf"))
    pdf_files = [f for f in pdf_files if not f.name.startswith('.')]  # Exclude hidden files

    print(f"📂 Found {len(pdf_files)} open access papers")
    print()

    # Initialize OpenAI client
    print("🔧 Initializing GPT-4 for paper evaluation...")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print()

    # Process each paper
    new_papers_data = []

    for i, paper_file in enumerate(pdf_files, 1):
        print(f"📄 [{i}/{len(pdf_files)}] Processing: {paper_file.name}")

        # Extract text
        print(f"   📖 Extracting text...")
        text = extract_text_from_pdf(paper_file)

        if not text or len(text) < 500:
            print(f"   ⚠️  Insufficient text extracted, skipping")
            continue

        # Extract paper info
        paper_info = extract_paper_info(text, paper_file.name)
        print(f"   📋 Title: {paper_info['title'][:60]}...")

        # Score with GPT-4
        print(f"   🤖 Evaluating with GPT-4...")
        scores = await score_paper_with_gpt4(paper_info, openai_client)

        print(f"   ✅ Scores: Overall={scores['overall']}, "
              f"Novelty={scores['novelty']}, "
              f"Method={scores['methodology']}, "
              f"Clarity={scores['clarity']}, "
              f"Significance={scores['significance']}")

        # Add to new papers list
        new_papers_data.append({
            "id": f"open_access_{i:03d}",
            "title": paper_info["title"],
            "abstract": paper_info["abstract"],
            "content": paper_info["content"],
            "human_scores": scores,
            "source_file": paper_file.name,
            "source": "open_access_2nd_3rd_tier"
        })

        print()

    if not new_papers_data:
        print("❌ No papers successfully processed")
        return

    # Load existing dataset
    existing_dataset_path = Path("data/validation/validation_dataset_v1.json")

    if existing_dataset_path.exists():
        with open(existing_dataset_path, 'r', encoding='utf-8') as f:
            existing_dataset = json.load(f)

        existing_papers = existing_dataset.get("papers", [])
        print(f"📊 Existing dataset: {len(existing_papers)} papers")
    else:
        existing_papers = []
        print("📊 No existing dataset found, creating new")

    # Merge datasets
    all_papers = existing_papers + new_papers_data

    # Re-assign IDs to maintain consistency
    for idx, paper in enumerate(all_papers, 1):
        paper["id"] = f"paper_{idx:03d}"

    # Create merged dataset
    merged_dataset = {
        "papers": all_papers,
        "metadata": {
            "total_papers": len(all_papers),
            "creation_date": "2025-10-06",
            "version": "2.0",
            "scorer": "GPT-4 (gpt-4-turbo)",
            "sources": {
                "papers_collection": len(existing_papers),
                "open_access_2nd_3rd_tier": len(new_papers_data)
            },
            "note": "Merged dataset with open access papers from 2nd/3rd tier journals"
        }
    }

    # Save merged dataset
    output_path = Path("data/validation/validation_dataset_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("DATASET MERGING COMPLETE")
    print("=" * 80)
    print(f"✅ Added {len(new_papers_data)} open access papers")
    print(f"✅ Total papers in merged dataset: {len(all_papers)}")
    print(f"✅ Saved to: {output_path}")
    print()

    # Show score distribution for new papers
    print("📊 New Papers Score Distribution:")
    overall_scores = [p["human_scores"]["overall"] for p in new_papers_data]

    for score in range(1, 11):
        count = overall_scores.count(score)
        if count > 0:
            bar = "█" * count
            print(f"   {score:2d}: {bar} ({count})")

    if overall_scores:
        avg_score = sum(overall_scores) / len(overall_scores)
        print(f"\n   Average Overall Score (new papers): {avg_score:.2f}")

    # Show combined distribution
    print("\n📊 Combined Dataset Score Distribution:")
    all_overall_scores = [p["human_scores"]["overall"] for p in all_papers]

    for score in range(1, 11):
        count = all_overall_scores.count(score)
        if count > 0:
            bar = "█" * count
            print(f"   {score:2d}: {bar} ({count})")

    avg_all = sum(all_overall_scores) / len(all_overall_scores)
    print(f"\n   Average Overall Score (all papers): {avg_all:.2f}")

    print()
    print("🚀 Ready for model training!")
    print(f"   Next: python scripts/train_hybrid_model.py")


if __name__ == "__main__":
    asyncio.run(process_open_access_papers())
