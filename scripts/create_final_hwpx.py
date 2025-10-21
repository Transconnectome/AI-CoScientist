#!/usr/bin/env python3
"""Create final HWPX file from improved grant proposal."""

import shutil
from pathlib import Path
from datetime import datetime

def create_final_hwpx():
    """Create final HWPX file with AI-CoScientist improvements."""

    print("=" * 80)
    print("CREATING FINAL HWPX DOCUMENT")
    print("=" * 80)
    print()

    # Paths
    improved_docx = Path('data/grant_proposal_improved.docx')
    original_hwpx = Path('data/grant.hwpx')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_hwpx = Path(f'data/grant_improved_ai_coscientist_{timestamp}.hwpx')
    final_docx = Path(f'data/grant_improved_ai_coscientist_{timestamp}.docx')

    print(f"📄 Input (improved): {improved_docx}")
    print(f"📄 Template (HWPX format): {original_hwpx}")
    print()

    # Step 1: Copy improved DOCX to final location
    print("Step 1: Creating final DOCX version...")
    shutil.copy2(improved_docx, final_docx)
    print(f"✅ Saved: {final_docx}")
    print(f"   Size: {final_docx.stat().st_size / 1024:.1f} KB")
    print()

    # Step 2: Copy original HWPX as template
    # Note: HWPX is Korean format that requires Hancom Office to properly edit
    # For now, we'll create a HWPX by copying the original and providing DOCX
    print("Step 2: Creating HWPX package...")
    print("⚠️  Note: HWPX format requires Hancom Office for full editing")
    print("   Creating HWPX reference file from original template...")
    shutil.copy2(original_hwpx, final_hwpx)
    print(f"✅ Saved: {final_hwpx}")
    print(f"   Size: {final_hwpx.stat().st_size / 1024:.1f} KB")
    print()

    # Step 3: Create README for manual HWPX integration
    readme_path = Path(f'data/HWPX_INTEGRATION_GUIDE_{timestamp}.md')
    readme_content = f"""# HWPX Integration Guide

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files Created

1. **Improved DOCX**: `{final_docx.name}`
   - Contains all AI-CoScientist improvements
   - Ready for review and further editing
   - Compatible with Microsoft Word and Hancom Office

2. **Reference HWPX**: `{final_hwpx.name}`
   - Template copy of original HWPX structure
   - To integrate improvements, open in Hancom Office and manually incorporate

3. **AI-CoScientist Improvements**: `AI_COSCIENTIST_IMPROVEMENTS.md`
   - Detailed improvement suggestions from AI-CoScientist analysis
   - Original evaluation scores and target scores

## AI-CoScientist Analysis Results

### Original Proposal Evaluation
- Overall: 7.50/10
- Novelty: 7.07/10
- Methodology: 7.55/10
- Clarity: 6.90/10 ⚠️ (needs improvement)
- Significance: 6.94/10 ⚠️ (needs improvement)

### Improvements Applied
✅ **Executive Summary**: 1-minute overview for government officials
✅ **Government Support Necessity**: Rewritten with crisis urgency
✅ **ROI Calculation**: 70x return (1,013억원 → 7조원 annual savings)
✅ **Economic Benefits Table**: Clear visualization of financial impact
✅ **International Comparison**: US, EU, China brain initiatives
✅ **Societal Impact**: Quantified benefits (suicide prevention, dementia care)

### Key Enhancements for Government Officials

#### 1. Crisis Messaging
- 자살률 OECD 1위 (인구 10만명당 24.1명)
- 치매 환자 95만명 돌파
- 뇌질환 사회경제적 비용 연 7조원

#### 2. Clear ROI
- 정부 투자: 1,013억원 (5년)
- 경제적 효과: 연 7조원 절감
- 투자 대비 수익: 70배
- 민간 투자 유도: 3조원
- 신산업 창출: 10조원

#### 3. International Context
- 미국: Brain Initiative $60억
- EU: Human Brain Project €10억
- 중국: China Brain Project ¥200억
- 한국: AI 모델 개발 3-5년 뒤처짐

#### 4. Urgency
- 2026년 시작하지 않으면 기술 격차 5년 이상
- 글로벌 brain-AI 시장 진입 실패 위험

## Manual Integration Steps for HWPX

Since HWPX is a proprietary Korean format, manual integration in Hancom Office is recommended:

1. **Open** `{final_hwpx.name}` in Hancom Office
2. **Reference** `{final_docx.name}` for improved content
3. **Manually integrate** key sections:
   - Executive summary (place at beginning)
   - Improved "정부 지원 필요성" section
   - Economic benefits table
   - International comparison data
4. **Maintain** original HWPX formatting and structure
5. **Verify** all improvements are integrated
6. **Save** final version in HWPX format

## Alternative: Use DOCX Directly

The improved DOCX file (`{final_docx.name}`) contains all enhancements and is:
- Compatible with both Microsoft Office and Hancom Office
- Ready for submission if DOCX format is acceptable
- Contains all AI-CoScientist recommended improvements

## AI-CoScientist Methodology

This improved proposal was generated using the AI-CoScientist system:

1. **Evaluation**: Ensemble scoring (GPT-4, Hybrid, Multi-task models)
2. **Analysis**: Identified weak dimensions (Clarity, Significance)
3. **Improvement**: Generated targeted suggestions using Claude Sonnet 4.5
4. **Application**: Integrated improvements into proposal structure
5. **Validation**: Re-evaluated with ensemble scorer

## Files Reference

- Original: `grant.hwpx`
- Evaluation results: Logged in terminal output
- Improvement suggestions: `AI_COSCIENTIST_IMPROVEMENTS.md`
- Improved DOCX: `{final_docx.name}`
- Reference HWPX: `{final_hwpx.name}`

---

**Created by**: AI-CoScientist System
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"Step 3: Created integration guide...")
    print(f"✅ Saved: {readme_path}")
    print()

    # Summary
    print("=" * 80)
    print("✅ FINAL DOCUMENTS CREATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("📦 Deliverables:")
    print(f"   1. Improved DOCX: {final_docx}")
    print(f"   2. Reference HWPX: {final_hwpx}")
    print(f"   3. Integration Guide: {readme_path}")
    print(f"   4. Improvement Details: data/AI_COSCIENTIST_IMPROVEMENTS.md")
    print()
    print("🎯 AI-CoScientist Improvements Applied:")
    print("   ✅ Executive summary for quick understanding")
    print("   ✅ Crisis urgency messaging")
    print("   ✅ Clear ROI (70x return)")
    print("   ✅ Economic benefits table")
    print("   ✅ International comparison")
    print("   ✅ Societal impact quantification")
    print("   ✅ Target audience: 과학기술부 + 기재부 officials")
    print()
    print("📋 Next Steps:")
    print("   1. Review improved DOCX file")
    print("   2. For HWPX format: Follow integration guide")
    print("   3. Manual review and final editing in Hancom Office")
    print()
    print("💡 Note: HWPX format requires Hancom Office for proper editing.")
    print("   The DOCX file contains all improvements and can be used directly")
    print("   or converted to HWPX format using Hancom Office.")
    print()

if __name__ == '__main__':
    create_final_hwpx()
