#!/usr/bin/env python3
"""
Red Team / Blue Team 제안서 리뷰 시스템
=====================================
세계 최고 수준의 전문가 3명 관점에서 제안서를 공격(Red)하고 방어(Blue)

전문가 패널:
1. Dr. Catherine Lord (발달장애 전문가) - UCLA, ADOS-2 공동개발자
2. Dr. Karl Friston (신경과학 전문가) - UCL, Free Energy Principle 창시자  
3. Dr. Yoshua Bengio (AI 전문가) - Mila/Montreal, Turing Award 수상자

Usage:
    python scripts/red_blue_team_review.py --proposal data/발달장애/_grant_UPE_ENHANCED.md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExpertPersona:
    """전문가 페르소나 정의"""
    name: str
    title: str
    affiliation: str
    expertise: List[str]
    h_index: int
    notable_works: List[str]
    review_style: str
    red_team_focus: List[str]
    blue_team_focus: List[str]


# 세계 최고 수준 전문가 3인 정의
EXPERT_PANEL = [
    ExpertPersona(
        name="Dr. Catherine Lord",
        title="Distinguished Professor of Psychiatry & Education",
        affiliation="UCLA Semel Institute, USA",
        expertise=["Autism Spectrum Disorder", "ADOS-2 Development", "Developmental Psychopathology", "Longitudinal Studies"],
        h_index=180,
        notable_works=["ADOS-2 (Gold Standard ASD Diagnosis)", "Lancet Commission on ASD 2022", "Nature Reviews 2020"],
        review_style="Clinical rigor with developmental perspective. Demands replicable diagnostic criteria and longitudinal validation.",
        red_team_focus=[
            "진단 도구의 임상적 타당성",
            "조기 진단의 위양성/위음성 윤리적 문제",
            "발달 궤적의 개인차 반영 여부",
            "다문화/다인종 적용 가능성",
            "부모 심리적 영향 고려"
        ],
        blue_team_focus=[
            "종단 데이터의 희소성 극복",
            "멀티모달 통합의 임상적 가치",
            "조기 개입의 과학적 근거",
            "한국 특화 데이터의 글로벌 기여"
        ]
    ),
    ExpertPersona(
        name="Dr. Karl Friston",
        title="Scientific Director, Wellcome Centre for Human Neuroimaging",
        affiliation="University College London, UK",
        expertise=["Computational Neuroscience", "Free Energy Principle", "Dynamic Causal Modelling", "Statistical Parametric Mapping"],
        h_index=220,
        notable_works=["SPM (25,000+ citations)", "Free Energy Principle", "Active Inference Framework"],
        review_style="Mathematically rigorous, demands formal theoretical frameworks. Questions biological plausibility.",
        red_team_focus=[
            "신경과학적 이론적 기반 부재",
            "뇌 발달 역동성 모델링의 한계",
            "Physics-informed loss의 수학적 정의 부재",
            "인과관계 vs 상관관계 혼동",
            "뇌 가소성과 발달 비선형성"
        ],
        blue_team_focus=[
            "멀티스케일 뇌 모델링의 필요성",
            "Predictive Coding 관점의 발달장애",
            "DTI 기반 구조적 연결성 분석",
            "Neural ODE의 이론적 정당성"
        ]
    ),
    ExpertPersona(
        name="Dr. Yoshua Bengio",
        title="Full Professor & Scientific Director",
        affiliation="Mila - Quebec AI Institute, University of Montreal, Canada",
        expertise=["Deep Learning", "Foundation Models", "Causal Representation Learning", "AI Safety"],
        h_index=250,
        notable_works=["Attention Is All You Need (co-cited)", "GAN Theory", "Turing Award 2018"],
        review_style="Technically rigorous, questions scalability and generalization. Emphasizes AI safety and ethics.",
        red_team_focus=[
            "130B 모델 학습의 실현 가능성",
            "멀티모달 퓨전 아키텍처 검증 부재",
            "LoRA 비용 절감 주장의 과장",
            "오버피팅 및 일반화 위험",
            "AI 안전성 및 편향 문제"
        ],
        blue_team_focus=[
            "Foundation Model 접근법의 장점",
            "Transfer Learning의 효율성",
            "ESM3 활용의 혁신성",
            "강화학습 기반 치료 최적화"
        ]
    )
]


RED_TEAM_PROMPT_TEMPLATE = """
# 🔴 RED TEAM DEVASTATING CRITIQUE
## {expert_name} ({expert_affiliation})
### {expert_title}

---

## EXPERT PROFILE
- **H-index**: {h_index}
- **Major Works**: {notable_works}
- **Review Style**: {review_style}
- **Expertise**: {expertise}

---

## MISSION: FIND FATAL FLAWS

As {expert_name}, one of the world's leading experts in {primary_expertise}, you are tasked with a RUTHLESS critical analysis of this proposal. Your goal is to identify ALL weaknesses that could lead to:

1. **Project Failure** (technical infeasibility)
2. **Wasted Resources** (unrealistic claims)
3. **Ethical Harm** (patient safety risks)
4. **Scientific Misconduct** (overclaiming, cherry-picking)

---

## ATTACK VECTORS (Your Focus Areas)

{red_team_focus}

---

## PROPOSAL UNDER REVIEW

{proposal_content}

---

## REQUIRED OUTPUT FORMAT

### SECTION 1: FATAL FLAWS (Score: X/100)
Identify 3-5 critical issues that could KILL this project.

| # | Fatal Flaw | Impact | Probability | Risk Level |
|---|-----------|--------|-------------|------------|
| 1 | ... | Catastrophic/Major/Moderate | X% | CRITICAL/HIGH/MEDIUM |

### SECTION 2: TECHNICAL ATTACKS
- **Claim**: [Exact quote from proposal]
- **Attack**: [Why this is wrong/exaggerated]
- **Evidence**: [Counter-evidence or missing verification]
- **Damage**: [Consequence if unaddressed]

### SECTION 3: METHODOLOGICAL WEAKNESSES
Specific issues with:
- Sample size justification
- Statistical power
- Validation strategy
- Reproducibility

### SECTION 4: CREDIBILITY ATTACKS
- Overclaimed capabilities ("세계 최초", "혁신적")
- Unverified partnerships
- Unrealistic timelines
- Budget inconsistencies

### SECTION 5: RISK SCORE
**Overall Credibility Score**: X/100
**Recommendation**: REJECT / MAJOR REVISION / MINOR REVISION / ACCEPT
**Top 3 Reasons for Score**:
1. ...
2. ...
3. ...

---

## EXPERT VERDICT

As {expert_name}, my professional assessment is:

[2-3 paragraphs of expert-level critique in the voice of the expert]
"""


BLUE_TEAM_PROMPT_TEMPLATE = """
# 🔵 BLUE TEAM DEFENSE & IMPROVEMENT
## {expert_name} ({expert_affiliation})
### {expert_title}

---

## EXPERT PROFILE
- **H-index**: {h_index}
- **Major Works**: {notable_works}
- **Expertise**: {expertise}

---

## MISSION: DEFEND AND IMPROVE

As {expert_name}, you have reviewed the Red Team critique. Now provide:

1. **Valid Defense** - Where the Red Team is wrong or overstated
2. **Constructive Improvements** - How to fix legitimate weaknesses
3. **Reframing Strategy** - Better positioning of strengths

---

## DEFENSE FOCUS AREAS

{blue_team_focus}

---

## RED TEAM CRITIQUE TO ADDRESS

{red_team_critique}

---

## ORIGINAL PROPOSAL

{proposal_content}

---

## REQUIRED OUTPUT FORMAT

### SECTION 1: RED TEAM REBUTTAL

| Red Team Attack | Validity (0-100%) | Defense | Evidence |
|-----------------|-------------------|---------|----------|
| ... | X% valid | ... | ... |

### SECTION 2: ACKNOWLEDGED WEAKNESSES
Accept these critiques and propose fixes:

| Weakness | Severity | Proposed Fix | Timeline | Impact |
|----------|----------|--------------|----------|--------|
| ... | CRITICAL/HIGH/MEDIUM | ... | X weeks | +Y points |

### SECTION 3: HIDDEN STRENGTHS (Red Team Missed)
Strengths the Red Team overlooked:
1. ...
2. ...
3. ...

### SECTION 4: IMPROVEMENT ROADMAP
Prioritized action items:

**Week 1 (Critical)**:
- [ ] ...

**Week 2-3 (High)**:
- [ ] ...

**Week 4+ (Medium)**:
- [ ] ...

### SECTION 5: REVISED SCORE PROJECTION

| State | Score | Funding Probability |
|-------|-------|---------------------|
| Current (Red Team) | X/100 | Y% |
| After Week 1 Fixes | X/100 | Y% |
| After All Fixes | X/100 | Y% |

---

## EXPERT RECOMMENDATION

As {expert_name}, my professional recommendation for improvement is:

[2-3 paragraphs of constructive expert advice]
"""


SYNTHESIS_PROMPT_TEMPLATE = """
# ⚖️ RED vs BLUE SYNTHESIS
## Panel Moderator: Integration of Expert Reviews

---

## EXPERT PANEL SUMMARY

### 🩺 발달장애 전문가 (Dr. Catherine Lord)
**Red Team Score**: {dd_expert_red_score}/100
**Key Attacks**: {dd_expert_red_summary}
**Blue Team Defense**: {dd_expert_blue_summary}

### 🧠 신경과학 전문가 (Dr. Karl Friston)
**Red Team Score**: {neuro_expert_red_score}/100
**Key Attacks**: {neuro_expert_red_summary}
**Blue Team Defense**: {neuro_expert_blue_summary}

### 🤖 AI 전문가 (Dr. Yoshua Bengio)
**Red Team Score**: {ai_expert_red_score}/100
**Key Attacks**: {ai_expert_red_summary}
**Blue Team Defense**: {ai_expert_blue_summary}

---

## CONSENSUS ANALYSIS

### Points of Agreement (All 3 Experts)
1. ...
2. ...
3. ...

### Points of Disagreement
| Issue | Lord | Friston | Bengio |
|-------|------|---------|--------|
| ... | ... | ... | ... |

---

## FINAL WEIGHTED SCORE

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Clinical Validity (Lord) | 35% | X | X |
| Neuroscience Rigor (Friston) | 30% | X | X |
| AI Feasibility (Bengio) | 35% | X | X |
| **TOTAL** | 100% | - | **X/100** |

---

## ACTIONABLE RECOMMENDATIONS

### MUST FIX (Before Submission)
1. ...
2. ...
3. ...

### SHOULD FIX (If Time Permits)
1. ...
2. ...

### NICE TO HAVE
1. ...

---

## FINAL VERDICT

**Current Score**: X/100
**Achievable Score**: X/100 (with fixes)
**Funding Probability**: 
- Current: X%
- After Fixes: X%

**Recommendation**: [REJECT/MAJOR REVISION/MINOR REVISION/CONDITIONAL ACCEPT]
"""


def load_proposal(proposal_path: str) -> str:
    """제안서 로드"""
    with open(proposal_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_expert_focus(focus_list: List[str]) -> str:
    """전문가 집중 영역 포맷팅"""
    return "\n".join([f"- **{i+1}. {item}**" for i, item in enumerate(focus_list)])


def generate_red_team_prompt(expert: ExpertPersona, proposal: str) -> str:
    """Red Team 프롬프트 생성"""
    return RED_TEAM_PROMPT_TEMPLATE.format(
        expert_name=expert.name,
        expert_title=expert.title,
        expert_affiliation=expert.affiliation,
        h_index=expert.h_index,
        notable_works=", ".join(expert.notable_works),
        review_style=expert.review_style,
        expertise=", ".join(expert.expertise),
        primary_expertise=expert.expertise[0],
        red_team_focus=format_expert_focus(expert.red_team_focus),
        proposal_content=proposal
    )


def generate_blue_team_prompt(expert: ExpertPersona, proposal: str, red_critique: str) -> str:
    """Blue Team 프롬프트 생성"""
    return BLUE_TEAM_PROMPT_TEMPLATE.format(
        expert_name=expert.name,
        expert_title=expert.title,
        expert_affiliation=expert.affiliation,
        h_index=expert.h_index,
        notable_works=", ".join(expert.notable_works),
        expertise=", ".join(expert.expertise),
        blue_team_focus=format_expert_focus(expert.blue_team_focus),
        red_team_critique=red_critique,
        proposal_content=proposal
    )


def save_prompts(output_dir: Path, proposal_name: str, prompts: Dict[str, str]):
    """프롬프트 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for prompt_name, prompt_content in prompts.items():
        output_path = output_dir / f"{proposal_name}_{prompt_name}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        print(f"✅ Saved: {output_path}")


def generate_all_prompts(proposal_path: str, output_dir: str = None) -> Dict[str, str]:
    """모든 Red/Blue Team 프롬프트 생성"""
    
    proposal = load_proposal(proposal_path)
    proposal_name = Path(proposal_path).stem
    
    if output_dir is None:
        output_dir = Path(proposal_path).parent / "red_blue_review"
    else:
        output_dir = Path(output_dir)
    
    all_prompts = {}
    
    print("\n" + "="*80)
    print("🔴🔵 RED TEAM / BLUE TEAM REVIEW SYSTEM")
    print("="*80)
    print(f"\n📄 Proposal: {proposal_path}")
    print(f"📁 Output: {output_dir}")
    print(f"👥 Expert Panel: {len(EXPERT_PANEL)} world-class experts")
    print("\n" + "-"*80)
    
    # 각 전문가별 프롬프트 생성
    for i, expert in enumerate(EXPERT_PANEL, 1):
        print(f"\n[{i}/3] Processing: {expert.name}")
        print(f"    🎓 {expert.affiliation}")
        print(f"    📊 H-index: {expert.h_index}")
        
        # Red Team 프롬프트
        red_prompt = generate_red_team_prompt(expert, proposal)
        red_key = f"RED_TEAM_{expert.name.split()[-1].upper()}"
        all_prompts[red_key] = red_prompt
        print(f"    🔴 Red Team prompt generated ({len(red_prompt):,} chars)")
        
        # Blue Team 프롬프트 (Red Team 결과 placeholder)
        blue_prompt = generate_blue_team_prompt(
            expert, 
            proposal, 
            "[RED TEAM CRITIQUE WILL BE INSERTED HERE AFTER RED TEAM ANALYSIS]"
        )
        blue_key = f"BLUE_TEAM_{expert.name.split()[-1].upper()}"
        all_prompts[blue_key] = blue_prompt
        print(f"    🔵 Blue Team prompt generated ({len(blue_prompt):,} chars)")
    
    # 프롬프트 저장
    save_prompts(output_dir, proposal_name, all_prompts)
    
    # 종합 리포트 템플릿 저장
    synthesis_template = SYNTHESIS_PROMPT_TEMPLATE
    with open(output_dir / f"{proposal_name}_SYNTHESIS_TEMPLATE.md", 'w', encoding='utf-8') as f:
        f.write(synthesis_template)
    print(f"\n✅ Saved synthesis template")
    
    # 실행 가이드 생성
    guide = generate_execution_guide(proposal_name, output_dir)
    with open(output_dir / "EXECUTION_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(guide)
    print(f"✅ Saved execution guide")
    
    print("\n" + "="*80)
    print("✅ ALL PROMPTS GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\n📂 Output directory: {output_dir}")
    print(f"📝 Total files: {len(all_prompts) + 2}")
    print("\n🚀 Next Steps:")
    print("   1. Run Red Team prompts with Claude/GPT-4")
    print("   2. Insert Red Team results into Blue Team prompts")
    print("   3. Run Blue Team prompts")
    print("   4. Complete Synthesis template")
    print("   5. Apply improvements to proposal")
    
    return all_prompts


def generate_execution_guide(proposal_name: str, output_dir: Path) -> str:
    """실행 가이드 생성"""
    return f"""# 🎯 Red Team / Blue Team Review Execution Guide

## Proposal: {proposal_name}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Expert Panel

| # | Expert | Domain | H-index | Institution |
|---|--------|--------|---------|-------------|
| 1 | Dr. Catherine Lord | 발달장애/ASD | 180 | UCLA |
| 2 | Dr. Karl Friston | 신경과학/뇌영상 | 220 | UCL |
| 3 | Dr. Yoshua Bengio | AI/딥러닝 | 250 | Mila/Montreal |

---

## 🔴 STEP 1: Red Team Analysis

Execute each Red Team prompt in order:

### 1.1 발달장애 전문가 (Lord)
```bash
# Copy content from:
{output_dir}/{proposal_name}_RED_TEAM_LORD.md

# Paste to Claude Opus 4/GPT-4o and run
# Save output as: {proposal_name}_RED_RESULT_LORD.md
```

### 1.2 신경과학 전문가 (Friston)
```bash
# Copy content from:
{output_dir}/{proposal_name}_RED_TEAM_FRISTON.md

# Save output as: {proposal_name}_RED_RESULT_FRISTON.md
```

### 1.3 AI 전문가 (Bengio)
```bash
# Copy content from:
{output_dir}/{proposal_name}_RED_TEAM_BENGIO.md

# Save output as: {proposal_name}_RED_RESULT_BENGIO.md
```

---

## 🔵 STEP 2: Blue Team Defense

After Red Team analysis, update Blue Team prompts with actual critiques:

### 2.1 Edit Blue Team prompts
Replace `[RED TEAM CRITIQUE WILL BE INSERTED HERE...]` with actual Red Team results.

### 2.2 Run Blue Team prompts
Execute and save as:
- `{proposal_name}_BLUE_RESULT_LORD.md`
- `{proposal_name}_BLUE_RESULT_FRISTON.md`
- `{proposal_name}_BLUE_RESULT_BENGIO.md`

---

## ⚖️ STEP 3: Synthesis

### 3.1 Complete synthesis template
Edit `{proposal_name}_SYNTHESIS_TEMPLATE.md`:
- Insert scores from each expert
- Summarize key attacks and defenses
- Calculate weighted final score

### 3.2 Generate final report
Combine all results into:
- `{proposal_name}_FINAL_REVIEW_REPORT.md`

---

## 📊 STEP 4: Apply Improvements

### Priority Matrix
| Priority | Timeline | Expected Impact |
|----------|----------|-----------------|
| CRITICAL | Week 1 | +15-20 points |
| HIGH | Week 2-3 | +8-12 points |
| MEDIUM | Week 4+ | +3-5 points |

### Improvement Tracking
- [ ] Fix #1: [Critical issue from Red Team]
- [ ] Fix #2: [Second priority]
- [ ] Fix #3: [Third priority]

---

## 🎯 Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Lord Score | ?/100 | 75+ | ⬜ |
| Friston Score | ?/100 | 75+ | ⬜ |
| Bengio Score | ?/100 | 75+ | ⬜ |
| **Weighted Avg** | ?/100 | **75+** | ⬜ |

---

## 📞 Support

For questions about this review system:
- See: CLAUDE.md (AI-CoScientist documentation)
- System: Unified Proposal Engine (UPE)
- Method: 7-Strategy RAG + Red/Blue Team Analysis
"""


def main():
    parser = argparse.ArgumentParser(
        description="Red Team / Blue Team Proposal Review System"
    )
    parser.add_argument(
        "--proposal", "-p",
        type=str,
        required=True,
        help="Path to proposal markdown file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for prompts (default: proposal_dir/red_blue_review)"
    )
    parser.add_argument(
        "--expert",
        type=str,
        choices=["lord", "friston", "bengio", "all"],
        default="all",
        help="Generate prompts for specific expert or all"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.proposal):
        print(f"❌ Error: Proposal file not found: {args.proposal}")
        sys.exit(1)
    
    generate_all_prompts(args.proposal, args.output)


if __name__ == "__main__":
    main()






