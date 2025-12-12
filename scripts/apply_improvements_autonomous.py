#!/usr/bin/env python3
"""
Autonomous Paper Improvement Script

Applies improvements to paper-mbbn.pdf based on generated strategy
using Claude API for content enhancement.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import anthropic

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.paper.ensemble_scorer import EnsemblePaperScorer

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


def load_paper_text(filepath: str) -> str:
    """Load paper text from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_improved_text(text: str, filepath: str):
    """Save improved text to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


async def init_scorer():
    """Initialize scorer once for reuse"""
    return EnsemblePaperScorer()


def extract_section(text: str, section_name: str) -> Optional[str]:
    """Extract a specific section from paper text"""
    # Common section headers
    section_patterns = {
        'Abstract': ['ABSTRACT', 'Abstract', '초록'],
        'Introduction': ['INTRODUCTION', 'Introduction', '서론', '1. Introduction'],
        'Methods': ['METHODS', 'Methods', 'METHODOLOGY', 'Methodology', '방법론', '2. Methods'],
        'Results': ['RESULTS', 'Results', '결과', '3. Results'],
        'Discussion': ['DISCUSSION', 'Discussion', '논의', '4. Discussion'],
        'Conclusion': ['CONCLUSION', 'Conclusion', '결론', '5. Conclusion']
    }

    patterns = section_patterns.get(section_name, [section_name])

    # Find section start
    start_idx = -1
    for pattern in patterns:
        idx = text.find(pattern)
        if idx != -1:
            start_idx = idx
            break

    if start_idx == -1:
        return None

    # Find next section (rough heuristic)
    next_section_idx = len(text)
    for other_section in section_patterns.keys():
        if other_section == section_name:
            continue
        for pattern in section_patterns[other_section]:
            idx = text.find(pattern, start_idx + 10)
            if idx != -1 and idx < next_section_idx:
                next_section_idx = idx

    return text[start_idx:next_section_idx].strip()


def improve_abstract_with_claude(abstract_text: str, api_key: str) -> str:
    """Improve abstract using Claude API following strategy"""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an expert scientific editor. Improve this abstract following these requirements:

REQUIREMENTS:
1. Use 4-sentence structure: Problem → Gap → Solution → Result
2. Strengthen the research gap identification
3. Add specific quantitative results
4. Improve clarity and conciseness
5. Maintain technical accuracy

CURRENT ABSTRACT:
{abstract_text}

STRATEGY GUIDANCE:
- Start with clear problem statement
- Explicitly state what existing work lacks
- Describe your solution concisely
- End with concrete results (numbers, percentages)

OUTPUT: Provide ONLY the improved abstract text, no explanations."""

    response = client.messages.create(
        model='claude-sonnet-4-5-20250929',
        max_tokens=800,
        temperature=0.7,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response.content[0].text.strip()


def improve_introduction_with_claude(intro_text: str, api_key: str) -> str:
    """Improve introduction using Claude API"""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an expert scientific editor. Improve this introduction section:

REQUIREMENTS:
1. Add clear "Our contribution" subsection highlighting 3 key differentiators
2. Strengthen literature review with explicit comparisons
3. Add road map paragraph at the end
4. Improve logical flow and transitions
5. Maintain technical depth

CURRENT INTRODUCTION:
{intro_text}

STRATEGY:
- Make contributions explicit and numbered
- Compare with 3-5 key related works
- End with paper structure overview

OUTPUT: Provide ONLY the improved introduction text."""

    response = client.messages.create(
        model='claude-sonnet-4-5-20250929',
        max_tokens=2000,
        temperature=0.7,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response.content[0].text.strip()


def improve_methods_with_claude(methods_text: str, api_key: str) -> str:
    """Improve methods section using Claude API"""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an expert scientific editor. Improve this methods section:

REQUIREMENTS:
1. Add detailed implementation details for reproducibility
2. Include hyperparameters and configuration details
3. Add statistical power analysis if missing
4. Improve clarity with subsections
5. Add algorithmic descriptions where appropriate

CURRENT METHODS:
{methods_text}

STRATEGY:
- Make all experimental details explicit
- Add subsections for better organization
- Include code/data availability statements

OUTPUT: Provide ONLY the improved methods text."""

    response = client.messages.create(
        model='claude-sonnet-4-5-20250929',
        max_tokens=2000,
        temperature=0.7,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response.content[0].text.strip()


def apply_improvements(
    original_text: str,
    target_sections: List[str],
    api_key: str
) -> Dict[str, str]:
    """Apply improvements to specified sections"""

    improvements = {}

    for section in target_sections:
        print(f"\n🔧 Improving {section}...")

        section_text = extract_section(original_text, section)
        if not section_text:
            print(f"  ⚠️  Could not extract {section}, skipping")
            continue

        print(f"  📏 Original length: {len(section_text)} chars")

        # Apply improvement based on section
        try:
            if section == 'Abstract':
                improved = improve_abstract_with_claude(section_text, api_key)
            elif section == 'Introduction':
                improved = improve_introduction_with_claude(section_text, api_key)
            elif section == 'Methods':
                improved = improve_methods_with_claude(section_text, api_key)
            else:
                print(f"  ℹ️  No specific improvement strategy for {section}")
                improved = section_text

            print(f"  ✅ Improved length: {len(improved)} chars")
            improvements[section] = improved

        except Exception as e:
            print(f"  ❌ Error improving {section}: {e}")
            improvements[section] = section_text  # Keep original

    return improvements


def replace_sections_in_text(
    original_text: str,
    improvements: Dict[str, str]
) -> str:
    """Replace improved sections in original text"""

    result = original_text

    for section_name, improved_content in improvements.items():
        original_section = extract_section(original_text, section_name)
        if original_section:
            result = result.replace(original_section, improved_content)

    return result


async def evaluate_paper_text(text: str, scorer: EnsemblePaperScorer) -> Dict:
    """Evaluate paper text using Ensemble Scorer"""
    result = await scorer.score_paper(text, return_individual=True)

    dimensions = result.get('dimensions', {})
    return {
        'overall_quality': result['overall'],
        'confidence': result['confidence'],
        'novelty': dimensions.get('novelty', 0.0),
        'methodology': dimensions.get('methodology', 0.0),
        'clarity': dimensions.get('clarity', 0.0),
        'significance': dimensions.get('significance', 0.0)
    }


async def main():
    print("=" * 60)
    print("🤖 AI-CoScientist Autonomous Improvement")
    print("=" * 60)

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return 1

    # Load original paper
    original_file = 'paper_mbbn_original.txt'
    if not os.path.exists(original_file):
        print(f"❌ Original paper not found: {original_file}")
        return 1

    print(f"\n📄 Loading paper: {original_file}")
    original_text = load_paper_text(original_file)
    print(f"   Length: {len(original_text):,} characters")

    # Initialize scorer (reuse across evaluations)
    print("\n🔧 Initializing ensemble scorer...")
    scorer = await init_scorer()

    # Initial evaluation
    print("\n📊 Baseline evaluation...")
    baseline_scores = await evaluate_paper_text(original_text, scorer)
    print(f"   Overall: {baseline_scores['overall_quality']:.2f}/10")
    print(f"   Novelty: {baseline_scores['novelty']:.2f}")
    print(f"   Methodology: {baseline_scores['methodology']:.2f}")
    print(f"   Clarity: {baseline_scores['clarity']:.2f}")
    print(f"   Significance: {baseline_scores['significance']:.2f}")

    # Priority sections to improve (from strategy)
    priority_sections = ['Abstract', 'Introduction', 'Methods']

    print(f"\n🎯 Target sections: {', '.join(priority_sections)}")
    print("   Strategy: Novelty → Clarity → Significance")

    # Apply improvements iteratively
    current_text = original_text
    iteration = 1
    max_iterations = 3
    target_score = 9.0

    while iteration <= max_iterations:
        print(f"\n{'=' * 60}")
        print(f"🔄 ITERATION {iteration}/{max_iterations}")
        print(f"{'=' * 60}")

        # Apply improvements
        improvements = apply_improvements(
            current_text,
            priority_sections,
            api_key
        )

        if not improvements:
            print("⚠️  No improvements applied")
            break

        # Replace sections
        improved_text = replace_sections_in_text(current_text, improvements)

        # Save iteration result
        iteration_file = f'paper_mbbn_iteration_{iteration}.txt'
        save_improved_text(improved_text, iteration_file)
        print(f"\n💾 Saved: {iteration_file}")

        # Evaluate improved version
        print(f"\n📊 Evaluating iteration {iteration}...")
        improved_scores = await evaluate_paper_text(improved_text, scorer)

        print(f"\n📈 SCORES (Iteration {iteration}):")
        print(f"   Overall: {improved_scores['overall_quality']:.2f}/10 "
              f"(Δ {improved_scores['overall_quality'] - baseline_scores['overall_quality']:+.2f})")
        print(f"   Novelty: {improved_scores['novelty']:.2f} "
              f"(Δ {improved_scores['novelty'] - baseline_scores['novelty']:+.2f})")
        print(f"   Methodology: {improved_scores['methodology']:.2f} "
              f"(Δ {improved_scores['methodology'] - baseline_scores['methodology']:+.2f})")
        print(f"   Clarity: {improved_scores['clarity']:.2f} "
              f"(Δ {improved_scores['clarity'] - baseline_scores['clarity']:+.2f})")
        print(f"   Significance: {improved_scores['significance']:.2f} "
              f"(Δ {improved_scores['significance'] - baseline_scores['significance']:+.2f})")

        # Check if target reached
        if improved_scores['overall_quality'] >= target_score:
            print(f"\n✅ TARGET REACHED! Score: {improved_scores['overall_quality']:.2f} >= {target_score}")
            break

        # Check diminishing returns
        improvement = improved_scores['overall_quality'] - baseline_scores['overall_quality']
        if iteration > 1 and improvement < 0.1:
            print(f"\n⚠️  Diminishing returns detected (Δ {improvement:+.2f})")
            print("   Stopping iteration")
            break

        # Update for next iteration
        current_text = improved_text
        baseline_scores = improved_scores
        iteration += 1

    # Final summary
    print("\n" + "=" * 60)
    print("✅ IMPROVEMENT COMPLETE")
    print("=" * 60)

    final_file = 'paper_mbbn_improved_final.txt'
    save_improved_text(current_text, final_file)
    print(f"\n💾 Final version: {final_file}")

    # Calculate final improvement
    final_scores = await evaluate_paper_text(current_text, scorer)
    original_scores = await evaluate_paper_text(original_text, scorer)

    print(f"\n📊 FINAL RESULTS:")
    print(f"   Iterations: {iteration - 1}")
    print(f"   Starting score: {original_scores['overall_quality']:.2f}/10")
    print(f"   Final score: {final_scores['overall_quality']:.2f}/10")
    print(f"   Total improvement: {final_scores['overall_quality'] - original_scores['overall_quality']:+.2f}")

    # Convert to DOCX
    if DOCX_AVAILABLE:
        print("\n📝 Converting to DOCX format...")
        docx_file = final_file.replace('.txt', '.docx')
        try:
            import subprocess
            result = subprocess.run(
                ['python', 'scripts/convert_txt_to_docx.py', final_file, docx_file],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ DOCX created: {docx_file}")
            else:
                print(f"⚠️  DOCX conversion failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  DOCX conversion error: {e}")
    else:
        print("\n⚠️  python-docx not installed, skipping DOCX conversion")
        print("   Install with: pip install python-docx")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
