#!/usr/bin/env python3
"""Use AI-CoScientist to improve grant proposal based on evaluation results."""

import asyncio
import sys
from pathlib import Path
from docx import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.llm.service import LLMService
from src.services.paper.ensemble_scorer import EnsemblePaperScorer


async def improve_grant_proposal():
    """Improve grant proposal using AI-CoScientist."""

    print("=" * 80)
    print("AI-COSCIENTIST GRANT PROPOSAL IMPROVEMENT")
    print("=" * 80)
    print()

    # Read the grant proposal
    docx_path = 'data/grant_proposal_for_analysis.docx'
    print(f"📄 Reading: {docx_path}")
    doc = Document(docx_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    print(f"   Length: {len(full_text)} characters")
    print()

    # Initialize LLM service
    print("🔧 Initializing LLM service...")
    llm = LLMService(primary_provider="anthropic")  # Using Claude
    print("✅ LLM service ready")
    print()

    # Get improvement suggestions based on weak dimensions
    print("💡 Generating improvement suggestions for:")
    print("   • Clarity (6.90/10)")
    print("   • Significance (6.94/10)")
    print()

    # Clarity improvements
    print("📝 Clarity Improvement Suggestions:")
    print("-" * 80)

    clarity_prompt = f"""As an expert grant proposal reviewer, analyze this Korean government research proposal and provide specific suggestions to improve CLARITY and PRESENTATION.

Current Clarity Score: 6.90/10
Target: 8.0+/10

The proposal is for K-NeuroMind (Korean Brain Foundation Model) requesting government funding.

Please provide:
1. 3 specific structural improvements to enhance readability
2. 3 ways to make technical content more accessible to government officials (not researchers)
3. 2 visual elements or formatting suggestions

Focus on making the proposal compelling for 과학기술부 and 기재부 officials who need to justify budget allocation.

Here's the proposal text (first 8000 chars):
{full_text[:8000]}
"""

    clarity_suggestions = await llm.generate(
        prompt=clarity_prompt,
        max_tokens=2000,
        temperature=0.7
    )

    print(clarity_suggestions.content)
    print()
    print()

    # Significance improvements
    print("🎯 Significance Improvement Suggestions:")
    print("-" * 80)

    significance_prompt = f"""As an expert grant proposal reviewer, analyze this Korean government research proposal and provide specific suggestions to improve SIGNIFICANCE and IMPACT.

Current Significance Score: 6.94/10
Target: 8.0+/10

The proposal is for K-NeuroMind (Korean Brain Foundation Model) requesting government funding.

Please provide:
1. 3 ways to strengthen the societal impact message for government officials
2. 3 quantitative metrics to add that demonstrate national benefit
3. 2 comparison points with international projects (US BRAIN Initiative, EU HBP, China Brain Project)

Focus on convincing 기재부 (Ministry of Economy and Finance) that this investment will provide measurable returns.

Here's the proposal text (first 8000 chars):
{full_text[:8000]}
"""

    significance_suggestions = await llm.generate(
        prompt=significance_prompt,
        max_tokens=2000,
        temperature=0.7
    )

    print(significance_suggestions.content)
    print()
    print()

    # Generate specific text improvements
    print("✍️  Generating improved text sections:")
    print("-" * 80)

    rewrite_prompt = f"""Based on the analysis, rewrite the "정부 지원 필요성" (Government Support Necessity) section to be more compelling for government officials.

Current version focuses too much on technical details.
New version should:
1. Lead with concrete societal problems and costs
2. Show clear ROI (return on investment)
3. Use comparison with other countries
4. Be structured with clear bullet points
5. Use strong, persuasive language for budget justification

Original section:
그동안 축적된 뇌 데이터를 이용하여 한국형 브레인 파운데이션 모델을 개발하므로써 한국인 특이적 개별 뇌의 건강 및 예상 행동과 나아가, 뇌질환 위험도등을 '예측'하므로 뇌기능 이해의 고도화와 뇌질환 치료등 다양한 영역에 활용하기 될 수 있음

Please rewrite this to be much more impactful and persuasive for budget officials.
"""

    improved_section = await llm.generate(
        prompt=rewrite_prompt,
        max_tokens=1500,
        temperature=0.7
    )

    print(improved_section.content)
    print()
    print()

    # Save all suggestions to a file
    output_file = Path('data/AI_COSCIENTIST_IMPROVEMENT_SUGGESTIONS.md')

    suggestions_content = f"""# AI-CoScientist Improvement Suggestions
**Generated**: {Path().cwd()}
**Original Score**: 7.50/10
**Target Score**: 8.5+/10

## Weak Dimensions Identified
- **Clarity**: 6.90/10 → Target 8.0+/10
- **Significance**: 6.94/10 → Target 8.0+/10

---

## 1. CLARITY IMPROVEMENTS

{clarity_suggestions.content}

---

## 2. SIGNIFICANCE IMPROVEMENTS

{significance_suggestions.content}

---

## 3. IMPROVED TEXT SECTION: 정부 지원 필요성

{improved_section.content}

---

## Next Steps

1. Apply these suggestions to the grant proposal
2. Re-evaluate with AI-CoScientist
3. Iterate until target score (8.5+) is reached
4. Convert to final HWPX format

**Note**: These suggestions were generated using AI-CoScientist's LLM service with Claude (Anthropic).
"""

    output_file.write_text(suggestions_content, encoding='utf-8')
    print(f"💾 Suggestions saved to: {output_file}")
    print()

    print("=" * 80)
    print("✅ AI-CoScientist Improvement Process Complete!")
    print("=" * 80)
    print()
    print("📋 Summary:")
    print(f"  • Current overall score: 7.50/10")
    print(f"  • Weak dimensions: Clarity (6.90), Significance (6.94)")
    print(f"  • Improvement suggestions generated and saved")
    print(f"  • Next: Apply suggestions and re-evaluate")
    print()


if __name__ == '__main__':
    asyncio.run(improve_grant_proposal())
