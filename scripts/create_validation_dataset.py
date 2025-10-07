#!/usr/bin/env python3
"""Automatically create validation dataset from paper collection.

Extracts text from PDF and DOCX files, analyzes each paper using GPT-4,
and generates validation dataset with 5-dimensional quality scores.
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


def extract_text_from_docx(docx_path: Path) -> Optional[str]:
    """Extract text from DOCX file.

    Args:
        docx_path: Path to DOCX file

    Returns:
        Extracted text or None if failed
    """
    try:
        from docx import Document

        doc = Document(docx_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

        return "\n\n".join(paragraphs)

    except Exception as e:
        print(f"   ⚠️  DOCX extraction failed: {e}")
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


async def create_validation_dataset():
    """Main function to create validation dataset."""
    import os
    from openai import AsyncOpenAI

    print("=" * 80)
    print("AUTOMATED VALIDATION DATASET CREATION")
    print("=" * 80)
    print()

    # Paper directory - scan entire papers_collection folder
    papers_dir = Path("/Users/jiookcha/Documents/git/AI-CoScientist/papers_collection")

    if not papers_dir.exists():
        print(f"❌ Papers directory not found: {papers_dir}")
        return

    # Find all PDF and DOCX files recursively
    pdf_files = list(papers_dir.glob("**/*.pdf"))
    docx_files = list(papers_dir.glob("**/*.docx"))

    all_files = pdf_files + docx_files

    # Filter out non-paper files
    excluded_keywords = [
        'cover letter', 'revision letter', 'response', 'supplementary',
        'supp', '보고서', '계획서', '프로포절', '미팅', '노트',
        'proposal', 'report', 'meeting', 'poster', 'workshop',
        'tutorial', 'manual', 'readme', 'template', 'policy',
        'table', 'figure', 'statistics', 'readme', '초록',
        'irb', '동의서', 'consent', '연습장'
    ]

    def is_likely_paper(filename: str) -> bool:
        """Check if file is likely a scientific paper."""
        name_lower = filename.lower()

        # Exclude hidden files
        if filename.startswith('.'):
            return False

        # Exclude files with excluded keywords
        for keyword in excluded_keywords:
            if keyword in name_lower:
                return False

        # Include if it looks like a manuscript or paper
        include_keywords = ['manuscript', 'draft', 'paper', 'article', '.pdf', '.docx']

        # If no specific inclusion keywords, check file size might help
        return True  # Default: include unless explicitly excluded

    all_files = [f for f in all_files if is_likely_paper(f.name)]

    # Limit to 80 papers to balance quality, cost, and time
    all_files = all_files[:80]

    # Count filtered files
    filtered_pdf = [f for f in all_files if f.suffix == '.pdf']
    filtered_docx = [f for f in all_files if f.suffix == '.docx']

    print(f"📂 Found {len(all_files)} papers (after filtering):")
    print(f"   - PDF files: {len(filtered_pdf)}")
    print(f"   - DOCX files: {len(filtered_docx)}")
    print()

    # Initialize OpenAI client
    print("🔧 Initializing GPT-4 for paper evaluation...")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print()

    # Process each paper
    papers_data = []

    for i, paper_file in enumerate(all_files, 1):
        print(f"📄 [{i}/{len(all_files)}] Processing: {paper_file.name}")

        # Extract text
        print(f"   📖 Extracting text...")
        if paper_file.suffix == '.pdf':
            text = extract_text_from_pdf(paper_file)
        else:  # .docx
            text = extract_text_from_docx(paper_file)

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

        # Add to dataset
        papers_data.append({
            "id": f"paper_{i:03d}",
            "title": paper_info["title"],
            "abstract": paper_info["abstract"],
            "content": paper_info["content"],
            "human_scores": scores,
            "source_file": paper_file.name
        })

        print()

    # Create validation dataset
    if not papers_data:
        print("❌ No papers successfully processed")
        return

    dataset = {
        "papers": papers_data,
        "metadata": {
            "total_papers": len(papers_data),
            "creation_date": "2025-10-06",
            "version": "1.0",
            "scorer": "GPT-4 (gpt-4-turbo)",
            "field": "Mixed (AI, Neuroscience, Quantum Computing, etc.)",
            "note": "Automatically scored using GPT-4 analysis"
        }
    }

    # Save to file
    output_path = Path("data/validation/validation_dataset_v1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("DATASET CREATION COMPLETE")
    print("=" * 80)
    print(f"✅ Successfully processed {len(papers_data)} papers")
    print(f"✅ Saved to: {output_path}")
    print()

    # Show score distribution
    print("📊 Score Distribution:")
    overall_scores = [p["human_scores"]["overall"] for p in papers_data]

    for score in range(1, 11):
        count = overall_scores.count(score)
        if count > 0:
            bar = "█" * count
            print(f"   {score:2d}: {bar} ({count})")

    avg_score = sum(overall_scores) / len(overall_scores)
    print(f"\n   Average Overall Score: {avg_score:.2f}")

    print()
    print("🚀 Ready for model training!")
    print(f"   Next: python scripts/train_hybrid_model.py")


if __name__ == "__main__":
    asyncio.run(create_validation_dataset())
