#!/usr/bin/env python3
"""Section-specific paper evaluation module.

This module provides specialized evaluation for different paper sections
(Abstract, Introduction, Methods, Results, Discussion) with context-aware scoring.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List
from enum import Enum
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
import os
from dotenv import load_dotenv
from section_parser import SectionType, PaperSectionParser

load_dotenv()


class EvaluationContext(Enum):
    """Evaluation context modes."""
    STANDALONE = "standalone"  # Evaluate section in isolation
    INTEGRATED = "integrated"  # Evaluate with full paper context
    PROGRESSIVE = "progressive"  # Evaluate with previous sections


class SectionEvaluator:
    """Evaluate paper sections with section-specific criteria."""

    # Section-specific evaluation criteria and prompts
    SECTION_CRITERIA = {
        SectionType.ABSTRACT: {
            "dimensions": {
                "clarity": "Clear communication of research question, methods, and findings",
                "completeness": "Includes background, methods, results, and implications",
                "conciseness": "Efficiently conveys key information within word limits",
                "impact": "Compelling presentation of significance and novelty"
            },
            "prompt_template": """Evaluate this paper's ABSTRACT section:

{section_content}

Evaluation Context: {context_mode}
{context_info}

Rate the abstract on these dimensions (0-10):

1. **Clarity** (0-10): How clearly does it communicate the research?
   - Is the research question obvious?
   - Are methods briefly but clearly described?
   - Are key findings easy to understand?

2. **Completeness** (0-10): Does it cover all essential elements?
   - Background/motivation present?
   - Methods summarized?
   - Key results stated?
   - Implications/significance mentioned?

3. **Conciseness** (0-10): How efficiently does it use words?
   - Is every sentence necessary?
   - Are there redundancies?
   - Does it fit typical abstract length (150-250 words)?

4. **Impact** (0-10): How compelling is the presentation?
   - Does it grab attention?
   - Is the significance clear?
   - Does it make readers want to read more?

Respond in JSON format:
{{
    "clarity": {{"score": <float>, "justification": "<explanation>"}},
    "completeness": {{"score": <float>, "justification": "<explanation>"}},
    "conciseness": {{"score": <float>, "justification": "<explanation>"}},
    "impact": {{"score": <float>, "justification": "<explanation>"}},
    "overall": {{"score": <float>, "justification": "<overall assessment>"}},
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "suggestions": ["<improvement 1>", "<improvement 2>", ...]
}}"""
        },

        SectionType.INTRODUCTION: {
            "dimensions": {
                "motivation": "Clear establishment of research need and significance",
                "literature_review": "Comprehensive coverage of relevant prior work",
                "research_gap": "Identification of knowledge gaps this work addresses",
                "objectives": "Clear statement of research questions and goals"
            },
            "prompt_template": """Evaluate this paper's INTRODUCTION section:

{section_content}

Evaluation Context: {context_mode}
{context_info}

Rate the introduction on these dimensions (0-10):

1. **Motivation** (0-10): How well does it establish research need?
   - Is the problem clearly stated?
   - Is the significance convincing?
   - Does it explain why this matters?

2. **Literature Review** (0-10): How comprehensive is the background?
   - Are key prior works cited?
   - Is the field context clear?
   - Are relationships between works explained?

3. **Research Gap** (0-10): How clearly is the gap identified?
   - What's missing in current knowledge?
   - Why existing approaches are insufficient?
   - How this work addresses the gap?

4. **Objectives** (0-10): How clear are the research goals?
   - Are research questions explicit?
   - Are objectives measurable/testable?
   - Is scope appropriately defined?

Respond in JSON format:
{{
    "motivation": {{"score": <float>, "justification": "<explanation>"}},
    "literature_review": {{"score": <float>, "justification": "<explanation>"}},
    "research_gap": {{"score": <float>, "justification": "<explanation>"}},
    "objectives": {{"score": <float>, "justification": "<explanation>"}},
    "overall": {{"score": <float>, "justification": "<overall assessment>"}},
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "suggestions": ["<improvement 1>", "<improvement 2>", ...]
}}"""
        },

        SectionType.METHODS: {
            "dimensions": {
                "reproducibility": "Sufficient detail for replication",
                "rigor": "Scientific soundness and validity",
                "justification": "Clear rationale for methodological choices",
                "completeness": "All procedures and materials described"
            },
            "prompt_template": """Evaluate this paper's METHODS section:

{section_content}

Evaluation Context: {context_mode}
{context_info}

Rate the methods on these dimensions (0-10):

1. **Reproducibility** (0-10): Can others replicate this work?
   - Are procedures described in sufficient detail?
   - Are parameters and settings specified?
   - Are materials/tools clearly identified?
   - Could an expert reproduce the experiments?

2. **Rigor** (0-10): How scientifically sound are the methods?
   - Are controls appropriate?
   - Are sample sizes justified?
   - Are validation approaches adequate?
   - Are potential biases addressed?

3. **Justification** (0-10): Are methodological choices explained?
   - Why these methods over alternatives?
   - Are limitations acknowledged?
   - Are assumptions stated?

4. **Completeness** (0-10): Is everything described?
   - All experimental procedures?
   - Data collection methods?
   - Analysis approaches?
   - Statistical methods?

Respond in JSON format:
{{
    "reproducibility": {{"score": <float>, "justification": "<explanation>"}},
    "rigor": {{"score": <float>, "justification": "<explanation>"}},
    "justification": {{"score": <float>, "justification": "<explanation>"}},
    "completeness": {{"score": <float>, "justification": "<explanation>"}},
    "overall": {{"score": <float>, "justification": "<overall assessment>"}},
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "suggestions": ["<improvement 1>", "<improvement 2>", ...]
}}"""
        },

        SectionType.RESULTS: {
            "dimensions": {
                "clarity": "Clear presentation of findings",
                "evidence": "Sufficient data to support claims",
                "organization": "Logical flow and structure",
                "visualization": "Effective use of figures and tables"
            },
            "prompt_template": """Evaluate this paper's RESULTS section:

{section_content}

Evaluation Context: {context_mode}
{context_info}

Rate the results on these dimensions (0-10):

1. **Clarity** (0-10): How clearly are findings presented?
   - Are results easy to understand?
   - Is technical information accessible?
   - Are key findings highlighted?

2. **Evidence** (0-10): How well supported are claims?
   - Is data sufficient?
   - Are statistics appropriate?
   - Are uncertainties/errors reported?
   - Do results answer research questions?

3. **Organization** (0-10): How well structured is the section?
   - Is there a logical flow?
   - Are results grouped meaningfully?
   - Is progression clear?

4. **Visualization** (0-10): How effective are figures/tables?
   - Are visualizations clear and informative?
   - Are they properly referenced in text?
   - Do they enhance understanding?

Respond in JSON format:
{{
    "clarity": {{"score": <float>, "justification": "<explanation>"}},
    "evidence": {{"score": <float>, "justification": "<explanation>"}},
    "organization": {{"score": <float>, "justification": "<explanation>"}},
    "visualization": {{"score": <float>, "justification": "<explanation>"}},
    "overall": {{"score": <float>, "justification": "<overall assessment>"}},
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "suggestions": ["<improvement 1>", "<improvement 2>", ...]
}}"""
        },

        SectionType.DISCUSSION: {
            "dimensions": {
                "interpretation": "Insightful analysis of results",
                "implications": "Clear significance and broader impact",
                "limitations": "Honest acknowledgment of constraints",
                "future_work": "Thoughtful directions for advancement"
            },
            "prompt_template": """Evaluate this paper's DISCUSSION section:

{section_content}

Evaluation Context: {context_mode}
{context_info}

Rate the discussion on these dimensions (0-10):

1. **Interpretation** (0-10): How insightful is the analysis?
   - Are results meaningfully interpreted?
   - Are patterns and trends explained?
   - Are unexpected findings addressed?
   - Is interpretation balanced (not over-claiming)?

2. **Implications** (0-10): How well is significance conveyed?
   - Are broader impacts clear?
   - Is practical significance explained?
   - Are theoretical contributions stated?
   - How does this advance the field?

3. **Limitations** (0-10): Are constraints acknowledged?
   - Are study limitations honestly stated?
   - Are alternative explanations considered?
   - Is scope appropriately bounded?

4. **Future Work** (0-10): Are next steps thoughtful?
   - Are future directions specific?
   - Do they build on current findings?
   - Are they feasible and impactful?

Respond in JSON format:
{{
    "interpretation": {{"score": <float>, "justification": "<explanation>"}},
    "implications": {{"score": <float>, "justification": "<explanation>"}},
    "limitations": {{"score": <float>, "justification": "<explanation>"}},
    "future_work": {{"score": <float>, "justification": "<explanation>"}},
    "overall": {{"score": <float>, "justification": "<overall assessment>"}},
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "suggestions": ["<improvement 1>", "<improvement 2>", ...]
}}"""
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize evaluator.

        Args:
            api_key: Anthropic API key
        """
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.parser = PaperSectionParser()

    def evaluate_section(
        self,
        section_type: SectionType,
        section_content: str,
        context_mode: EvaluationContext = EvaluationContext.STANDALONE,
        full_paper_text: Optional[str] = None,
        previous_sections: Optional[Dict[SectionType, str]] = None
    ) -> Dict:
        """Evaluate a specific paper section.

        Args:
            section_type: Type of section to evaluate
            section_content: Content of the section
            context_mode: Evaluation context (standalone/integrated/progressive)
            full_paper_text: Full paper text (for integrated mode)
            previous_sections: Previous sections (for progressive mode)

        Returns:
            Evaluation results dictionary
        """
        if section_type not in self.SECTION_CRITERIA:
            raise ValueError(f"Unsupported section type: {section_type}")

        criteria = self.SECTION_CRITERIA[section_type]

        # Build context information
        context_info = self._build_context_info(
            context_mode, full_paper_text, previous_sections
        )

        # Format prompt
        prompt = criteria["prompt_template"].format(
            section_content=section_content,
            context_mode=context_mode.value,
            context_info=context_info
        )

        try:
            # Call Claude API
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                evaluation = json.loads(json_str)

                # Add metadata
                evaluation['section_type'] = section_type.value
                evaluation['context_mode'] = context_mode.value
                evaluation['word_count'] = len(section_content.split())

                return evaluation
            else:
                return self._fallback_evaluation(section_type, section_content)

        except Exception as e:
            print(f"Section evaluation error: {e}")
            return self._fallback_evaluation(section_type, section_content)

    def _build_context_info(
        self,
        context_mode: EvaluationContext,
        full_paper_text: Optional[str],
        previous_sections: Optional[Dict[SectionType, str]]
    ) -> str:
        """Build context information string for evaluation prompt.

        Args:
            context_mode: Evaluation context mode
            full_paper_text: Full paper text
            previous_sections: Dictionary of previous sections

        Returns:
            Context information string
        """
        if context_mode == EvaluationContext.STANDALONE:
            return "This section will be evaluated in isolation."

        elif context_mode == EvaluationContext.INTEGRATED:
            if full_paper_text:
                return f"""Full paper context available ({len(full_paper_text.split())} words).
Consider how this section fits within the complete paper."""
            else:
                return "Integrated mode requested but full paper not provided."

        elif context_mode == EvaluationContext.PROGRESSIVE:
            if previous_sections:
                section_list = ", ".join([s.value for s in previous_sections.keys()])
                return f"""Previous sections available: {section_list}
Consider consistency with and references to previous sections."""
            else:
                return "Progressive mode requested but previous sections not provided."

        return ""

    def _fallback_evaluation(self, section_type: SectionType, content: str) -> Dict:
        """Provide basic heuristic evaluation as fallback.

        Args:
            section_type: Section type
            content: Section content

        Returns:
            Basic evaluation dictionary
        """
        word_count = len(content.split())

        # Simple heuristic based on word count and structure
        base_score = 7.0
        if word_count < 50:
            base_score = 5.0
        elif word_count > 200:
            base_score = 7.5

        dimensions = list(self.SECTION_CRITERIA[section_type]["dimensions"].keys())

        result = {
            "section_type": section_type.value,
            "overall": {"score": base_score, "justification": "Heuristic evaluation"},
            "word_count": word_count,
            "strengths": ["Section present"],
            "weaknesses": ["LLM evaluation unavailable - using heuristics"],
            "suggestions": ["Enable LLM evaluation for detailed feedback"]
        }

        # Add dimension scores
        for dim in dimensions:
            result[dim] = {
                "score": base_score,
                "justification": "Heuristic scoring"
            }

        return result

    def evaluate_paper_sections(
        self,
        paper_text: str,
        sections_to_evaluate: Optional[List[SectionType]] = None,
        context_mode: EvaluationContext = EvaluationContext.INTEGRATED
    ) -> Dict[SectionType, Dict]:
        """Evaluate multiple sections of a paper.

        Args:
            paper_text: Full paper text
            sections_to_evaluate: List of sections to evaluate (None = all)
            context_mode: Evaluation context mode

        Returns:
            Dictionary mapping section types to evaluations
        """
        # Parse paper into sections
        sections = self.parser.parse(paper_text)

        if sections_to_evaluate is None:
            sections_to_evaluate = list(sections.keys())

        results = {}
        previous_sections = {}

        for section_type in sections_to_evaluate:
            if section_type not in sections:
                continue

            section = sections[section_type]

            # Determine context based on mode
            if context_mode == EvaluationContext.PROGRESSIVE:
                evaluation = self.evaluate_section(
                    section_type,
                    section.content,
                    context_mode,
                    previous_sections=previous_sections
                )
                previous_sections[section_type] = section.content
            elif context_mode == EvaluationContext.INTEGRATED:
                evaluation = self.evaluate_section(
                    section_type,
                    section.content,
                    context_mode,
                    full_paper_text=paper_text
                )
            else:  # STANDALONE
                evaluation = self.evaluate_section(
                    section_type,
                    section.content,
                    context_mode
                )

            results[section_type] = evaluation

        return results


def format_section_evaluation(evaluation: Dict) -> str:
    """Format section evaluation for display.

    Args:
        evaluation: Evaluation dictionary

    Returns:
        Formatted string
    """
    output = []

    section_type = evaluation.get('section_type', 'Unknown')
    overall_score = evaluation.get('overall', {}).get('score', 0.0)

    output.append(f"📄 Section: {section_type.upper()}")
    output.append(f"📊 Overall Score: {overall_score:.2f}/10")
    output.append(f"📝 Word Count: {evaluation.get('word_count', 0)}")
    output.append(f"🔍 Context: {evaluation.get('context_mode', 'standalone')}")
    output.append("")

    # Dimensional scores
    output.append("📋 Dimensional Scores:")
    for key, value in evaluation.items():
        if isinstance(value, dict) and 'score' in value and key != 'overall':
            output.append(f"   {key.replace('_', ' ').title()}: {value['score']:.2f}/10")
            if value.get('justification'):
                output.append(f"      → {value['justification']}")
    output.append("")

    # Strengths
    if evaluation.get('strengths'):
        output.append("✅ Strengths:")
        for strength in evaluation['strengths']:
            output.append(f"   • {strength}")
        output.append("")

    # Weaknesses
    if evaluation.get('weaknesses'):
        output.append("⚠️ Areas for Improvement:")
        for weakness in evaluation['weaknesses']:
            output.append(f"   • {weakness}")
        output.append("")

    # Suggestions
    if evaluation.get('suggestions'):
        output.append("💡 Improvement Suggestions:")
        for suggestion in evaluation['suggestions']:
            output.append(f"   • {suggestion}")

    return "\n".join(output)


if __name__ == "__main__":
    # Test section evaluation
    sample_abstract = """
    This paper introduces a novel deep learning approach for protein structure prediction.
    We developed a transformer-based architecture that achieves state-of-the-art accuracy
    on benchmark datasets, outperforming existing methods by 15%. Our model was trained
    on over 100,000 protein structures and validated using cross-validation. Results
    demonstrate significant improvements in prediction speed and accuracy, with potential
    applications in drug discovery and structural biology.
    """

    evaluator = SectionEvaluator()

    print("Testing Abstract Evaluation (Standalone Mode):\n")
    result = evaluator.evaluate_section(
        SectionType.ABSTRACT,
        sample_abstract,
        EvaluationContext.STANDALONE
    )

    print(format_section_evaluation(result))
