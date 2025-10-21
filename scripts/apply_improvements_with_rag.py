#!/usr/bin/env python3
"""
Autonomous Paper Improvement Script with RAG

Enhanced version that learns from past successful improvements
using ChromaDB and RAG (Retrieval-Augmented Generation).
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

from src.services.paper.ensemble_scorer import EnsemblePaperScorer
from src.services.embeddings import EmbeddingService
from src.services.rag import RAGManager

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
    section_patterns = {
        'Abstract': ['ABSTRACT', 'Abstract', '초록'],
        'Introduction': ['INTRODUCTION', 'Introduction', '서론', '1. Introduction'],
        'Methods': ['METHODS', 'Methods', 'METHODOLOGY', 'Methodology', '방법론', '2. Methods'],
        'Results': ['RESULTS', 'Results', '결과', '3. Results'],
        'Discussion': ['DISCUSSION', 'Discussion', '논의', '4. Discussion'],
        'Conclusion': ['CONCLUSION', 'Conclusion', '결론', '5. Conclusion']
    }

    patterns = section_patterns.get(section_name, [section_name])

    start_idx = -1
    for pattern in patterns:
        idx = text.find(pattern)
        if idx != -1:
            start_idx = idx
            break

    if start_idx == -1:
        return None

    next_section_idx = len(text)
    for other_section in section_patterns.keys():
        if other_section == section_name:
            continue
        for pattern in section_patterns[other_section]:
            idx = text.find(pattern, start_idx + 10)
            if idx != -1 and idx < next_section_idx:
                next_section_idx = idx

    return text[start_idx:next_section_idx].strip()


async def improve_section_with_rag(
    section_text: str,
    section_name: str,
    current_scores: Dict[str, float],
    rag_manager: RAGManager,
    api_key: str
) -> str:
    """
    Improve section using RAG-enhanced prompts.

    Args:
        section_text: Original section text
        section_name: Section name (Abstract, Introduction, etc.)
        current_scores: Current evaluation scores
        rag_manager: RAG manager for pattern search
        api_key: Anthropic API key

    Returns:
        Improved section text
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Identify main problem
    problems = []
    if current_scores.get('clarity', 0) < 7.5:
        problems.append("low clarity score")
    if current_scores.get('novelty', 0) < 7.5:
        problems.append("low novelty score")
    if current_scores.get('significance', 0) < 7.5:
        problems.append("low significance score")

    problem_desc = ", ".join(problems) if problems else "general improvement"

    # Search for similar successful improvements
    print(f"  🔍 Searching RAG for similar {section_name} improvements...")
    rag_results = await rag_manager.search_similar_improvements(
        section=section_name,
        problem_description=problem_desc,
        current_scores=current_scores,
        n_results=3
    )

    # Build prompt with RAG context
    base_requirements = {
        'Abstract': """1. Use 4-sentence structure: Problem → Gap → Solution → Result
2. Strengthen the research gap identification
3. Add specific quantitative results
4. Improve clarity and conciseness
5. Maintain technical accuracy""",
        'Introduction': """1. Add clear "Our contribution" subsection highlighting 3 key differentiators
2. Strengthen literature review with explicit comparisons
3. Add road map paragraph at the end
4. Improve logical flow and transitions
5. Maintain technical depth""",
        'Methods': """1. Add detailed implementation details for reproducibility
2. Include hyperparameters and configuration details
3. Add statistical power analysis if missing
4. Improve clarity with subsections
5. Add algorithmic descriptions where appropriate"""
    }

    requirements = base_requirements.get(section_name, "Improve quality and clarity")

    # Format RAG context
    rag_context = rag_manager.format_rag_context(rag_results) if rag_results else ""

    if rag_context:
        print(f"  ✅ Found {len(rag_results)} similar improvement patterns")
    else:
        print(f"  ℹ️  No similar patterns found (cold start)")

    prompt = f"""You are an expert scientific editor with access to a database of successful paper improvements.

TASK: Improve this {section_name} section

REQUIREMENTS:
{requirements}

CURRENT {section_name.upper()}:
{section_text}

CURRENT SCORES:
- Overall: {current_scores.get('overall_quality', 0):.2f}/10
- Novelty: {current_scores.get('novelty', 0):.2f}/10
- Clarity: {current_scores.get('clarity', 0):.2f}/10
- Significance: {current_scores.get('significance', 0):.2f}/10

{rag_context}

STRATEGY:
- Learn from the successful patterns above
- Apply similar techniques that worked well
- Adapt strategies to this specific content
- Focus on the lowest-scoring dimensions

OUTPUT: Provide ONLY the improved {section_name} text, no explanations."""

    response = client.messages.create(
        model='claude-sonnet-4-5-20250929',
        max_tokens=2000,
        temperature=0.7,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response.content[0].text.strip()


async def apply_improvements_with_rag(
    original_text: str,
    target_sections: List[str],
    current_scores: Dict[str, float],
    rag_manager: RAGManager,
    api_key: str
) -> Dict[str, str]:
    """Apply RAG-enhanced improvements to specified sections"""

    improvements = {}

    for section in target_sections:
        print(f"\n🔧 Improving {section} with RAG...")

        section_text = extract_section(original_text, section)
        if not section_text:
            print(f"  ⚠️  Could not extract {section}, skipping")
            continue

        print(f"  📏 Original length: {len(section_text)} chars")

        try:
            improved = await improve_section_with_rag(
                section_text=section_text,
                section_name=section,
                current_scores=current_scores,
                rag_manager=rag_manager,
                api_key=api_key
            )

            print(f"  ✅ Improved length: {len(improved)} chars")
            improvements[section] = improved

        except Exception as e:
            print(f"  ❌ Error improving {section}: {e}")
            improvements[section] = section_text

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
    print("=" * 70)
    print("🤖 AI-CoScientist Autonomous Improvement with RAG")
    print("=" * 70)

    # Check API keys
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return 1

    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️  OPENAI_API_KEY not found, RAG features will be limited")

    # Load original paper
    original_file = sys.argv[1] if len(sys.argv) > 1 else 'paper_mbbn_original.txt'
    if not os.path.exists(original_file):
        print(f"❌ Original paper not found: {original_file}")
        return 1

    print(f"\n📄 Loading paper: {original_file}")
    original_text = load_paper_text(original_file)
    print(f"   Length: {len(original_text):,} characters")

    # Extract paper name for output files
    paper_name = Path(original_file).stem.replace('_original', '')

    # Initialize services
    print("\n🔧 Initializing services...")

    # Scorer
    scorer = await init_scorer()
    print("   ✅ Ensemble Scorer ready")

    # RAG system
    rag_enabled = False
    rag_manager = None

    if openai_key:
        try:
            embedding_service = EmbeddingService(provider="openai", api_key=openai_key)
            rag_manager = RAGManager(
                embedding_service=embedding_service,
                chromadb_mode="auto"  # Will try Docker, then local
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

    # Initial evaluation
    print("\n📊 Baseline evaluation...")
    baseline_scores = await evaluate_paper_text(original_text, scorer)
    print(f"   Overall: {baseline_scores['overall_quality']:.2f}/10")
    print(f"   Novelty: {baseline_scores['novelty']:.2f}")
    print(f"   Methodology: {baseline_scores['methodology']:.2f}")
    print(f"   Clarity: {baseline_scores['clarity']:.2f}")
    print(f"   Significance: {baseline_scores['significance']:.2f}")

    # Priority sections to improve
    priority_sections = ['Abstract', 'Introduction', 'Methods']

    print(f"\n🎯 Target sections: {', '.join(priority_sections)}")
    if rag_enabled:
        print("   Strategy: RAG-enhanced improvement (learning from past successes)")
    else:
        print("   Strategy: Standard improvement (no RAG)")

    # Apply improvements iteratively
    current_text = original_text
    iteration = 1
    max_iterations = 3
    target_score = 9.0

    while iteration <= max_iterations:
        print(f"\n{'=' * 70}")
        print(f"🔄 ITERATION {iteration}/{max_iterations}")
        print(f"{'=' * 70}")

        # Apply improvements (with or without RAG)
        if rag_enabled and rag_manager:
            improvements = await apply_improvements_with_rag(
                current_text,
                priority_sections,
                baseline_scores,
                rag_manager,
                api_key
            )
        else:
            # Fallback to standard improvement
            print("⚠️  Using standard improvement (RAG not available)")
            from apply_improvements_autonomous import apply_improvements
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
        iteration_file = f'output/papers/{paper_name}_rag_iteration_{iteration}.txt'
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

        # Store successful patterns in RAG
        if rag_enabled and improved_scores['overall_quality'] > baseline_scores['overall_quality']:
            print("\n💾 Storing successful patterns in RAG...")
            for section_name, improved_text_section in improvements.items():
                original_section = extract_section(current_text, section_name)
                if original_section and original_section != improved_text_section:
                    await rag_manager.store_improvement_pattern(
                        paper_id=paper_name,
                        section=section_name,
                        before_text=original_section,
                        after_text=improved_text_section,
                        scores_before=baseline_scores,
                        scores_after=improved_scores,
                        strategy=f"RAG-enhanced iteration {iteration}",
                        field="neuroscience",
                        paper_type="methodology"
                    )
            print("   ✅ Patterns stored for future use")

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
    print("\n" + "=" * 70)
    print("✅ IMPROVEMENT COMPLETE")
    print("=" * 70)

    final_file = f'output/papers/{paper_name}_rag_improved_final.txt'
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

    if rag_enabled:
        stats = rag_manager.get_statistics()
        print(f"\n📚 RAG STATISTICS:")
        print(f"   Total patterns stored: {stats['total_patterns']}")
        print(f"   These patterns will help improve future papers!")

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
                print(f"⚠️  DOCX conversion failed")
        except Exception as e:
            print(f"⚠️  DOCX conversion error: {e}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
