#!/usr/bin/env python3
"""Cross-section consistency checker for scientific papers.

This module checks for consistency and coherence across paper sections:
- Methods-Results alignment
- Introduction-Discussion coherence
- Abstract-Full paper consistency
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
import os
from dotenv import load_dotenv
from section_parser import SectionType, Section

load_dotenv()


class ConsistencyIssueType(Enum):
    """Types of consistency issues."""
    METHODS_RESULTS_MISMATCH = "methods_results_mismatch"
    CLAIMS_NOT_SUPPORTED = "claims_not_supported"
    MISSING_METHODOLOGY = "missing_methodology"
    ABSTRACT_BODY_INCONSISTENCY = "abstract_body_inconsistency"
    INTRO_DISCUSSION_DISCONNECT = "intro_discussion_disconnect"
    TERMINOLOGY_INCONSISTENCY = "terminology_inconsistency"


class IssueSeverity(Enum):
    """Severity levels for consistency issues."""
    CRITICAL = "critical"  # Must be fixed
    MAJOR = "major"  # Should be fixed
    MINOR = "minor"  # Nice to fix
    SUGGESTION = "suggestion"  # Optional improvement


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue between sections."""
    issue_type: ConsistencyIssueType
    severity: IssueSeverity
    sections_involved: List[SectionType]
    description: str
    suggestion: str
    evidence: str


class ConsistencyChecker:
    """Check consistency across paper sections."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize consistency checker.

        Args:
            api_key: Anthropic API key
        """
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def check_methods_results_alignment(
        self,
        methods_section: Section,
        results_section: Section
    ) -> List[ConsistencyIssue]:
        """Check if Results section aligns with Methods.

        Args:
            methods_section: Methods section
            results_section: Results section

        Returns:
            List of consistency issues
        """
        prompt = f"""Analyze consistency between Methods and Results sections.

METHODS SECTION:
{methods_section.content}

RESULTS SECTION:
{results_section.content}

Check for:
1. Are all methods described in Methods actually used/reported in Results?
2. Are all results in Results backed by methods described in Methods?
3. Are statistical methods mentioned in Methods applied in Results?
4. Are experimental parameters from Methods reflected in Results?

Respond in JSON format:
{{
    "issues": [
        {{
            "severity": "critical|major|minor|suggestion",
            "description": "<what's inconsistent>",
            "suggestion": "<how to fix>",
            "evidence": "<specific quote or reference>"
        }}
    ],
    "overall_alignment": <float 0-10>,
    "justification": "<brief overall assessment>"
}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                issues = []
                for issue_data in result.get('issues', []):
                    severity_str = issue_data.get('severity', 'minor')
                    severity = IssueSeverity(severity_str)

                    issue = ConsistencyIssue(
                        issue_type=ConsistencyIssueType.METHODS_RESULTS_MISMATCH,
                        severity=severity,
                        sections_involved=[SectionType.METHODS, SectionType.RESULTS],
                        description=issue_data.get('description', ''),
                        suggestion=issue_data.get('suggestion', ''),
                        evidence=issue_data.get('evidence', '')
                    )
                    issues.append(issue)

                return issues

        except Exception as e:
            print(f"Consistency check error: {e}")

        return []

    def check_claims_support(
        self,
        abstract_section: Section,
        results_section: Section,
        discussion_section: Optional[Section] = None
    ) -> List[ConsistencyIssue]:
        """Check if claims in Abstract are supported by Results/Discussion.

        Args:
            abstract_section: Abstract section
            results_section: Results section
            discussion_section: Optional discussion section

        Returns:
            List of consistency issues
        """
        discussion_content = discussion_section.content if discussion_section else "N/A"

        prompt = f"""Analyze if claims in Abstract are supported by Results and Discussion.

ABSTRACT:
{abstract_section.content}

RESULTS:
{results_section.content}

DISCUSSION:
{discussion_content}

Check for:
1. Are all major claims in Abstract backed by Results?
2. Are quantitative claims in Abstract supported by data in Results?
3. Are interpretations in Abstract consistent with Discussion?
4. Are any claims overstated or unsupported?

Respond in JSON format:
{{
    "issues": [
        {{
            "severity": "critical|major|minor|suggestion",
            "description": "<what claim is unsupported or overstated>",
            "suggestion": "<how to fix>",
            "evidence": "<specific quote>"
        }}
    ],
    "overall_support": <float 0-10>,
    "justification": "<brief assessment>"
}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                issues = []
                sections_involved = [SectionType.ABSTRACT, SectionType.RESULTS]
                if discussion_section:
                    sections_involved.append(SectionType.DISCUSSION)

                for issue_data in result.get('issues', []):
                    severity_str = issue_data.get('severity', 'minor')
                    severity = IssueSeverity(severity_str)

                    issue = ConsistencyIssue(
                        issue_type=ConsistencyIssueType.CLAIMS_NOT_SUPPORTED,
                        severity=severity,
                        sections_involved=sections_involved,
                        description=issue_data.get('description', ''),
                        suggestion=issue_data.get('suggestion', ''),
                        evidence=issue_data.get('evidence', '')
                    )
                    issues.append(issue)

                return issues

        except Exception as e:
            print(f"Claims support check error: {e}")

        return []

    def check_intro_discussion_coherence(
        self,
        intro_section: Section,
        discussion_section: Section
    ) -> List[ConsistencyIssue]:
        """Check coherence between Introduction and Discussion.

        Args:
            intro_section: Introduction section
            discussion_section: Discussion section

        Returns:
            List of consistency issues
        """
        prompt = f"""Analyze coherence between Introduction and Discussion.

INTRODUCTION:
{intro_section.content}

DISCUSSION:
{discussion_section.content}

Check for:
1. Are research questions from Introduction addressed in Discussion?
2. Are gaps identified in Introduction filled in Discussion?
3. Are objectives from Introduction achieved (as stated in Discussion)?
4. Is the narrative arc coherent from problem to solution?

Respond in JSON format:
{{
    "issues": [
        {{
            "severity": "critical|major|minor|suggestion",
            "description": "<what's disconnected>",
            "suggestion": "<how to improve coherence>",
            "evidence": "<specific references>"
        }}
    ],
    "overall_coherence": <float 0-10>,
    "justification": "<brief assessment>"
}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                issues = []
                for issue_data in result.get('issues', []):
                    severity_str = issue_data.get('severity', 'minor')
                    severity = IssueSeverity(severity_str)

                    issue = ConsistencyIssue(
                        issue_type=ConsistencyIssueType.INTRO_DISCUSSION_DISCONNECT,
                        severity=severity,
                        sections_involved=[SectionType.INTRODUCTION, SectionType.DISCUSSION],
                        description=issue_data.get('description', ''),
                        suggestion=issue_data.get('suggestion', ''),
                        evidence=issue_data.get('evidence', '')
                    )
                    issues.append(issue)

                return issues

        except Exception as e:
            print(f"Intro-discussion coherence check error: {e}")

        return []

    def check_terminology_consistency(
        self,
        all_sections: Dict[SectionType, Section]
    ) -> List[ConsistencyIssue]:
        """Check terminology consistency across all sections.

        Args:
            all_sections: All paper sections

        Returns:
            List of terminology inconsistencies
        """
        sections_text = "\n\n---\n\n".join([
            f"[{section_type.value.upper()}]\n{section.content}"
            for section_type, section in all_sections.items()
        ])

        prompt = f"""Analyze terminology consistency across all sections.

FULL PAPER SECTIONS:
{sections_text}

Check for:
1. Inconsistent use of key terms (e.g., "machine learning" vs "ML" vs "artificial intelligence")
2. Inconsistent capitalization of technical terms
3. Inconsistent abbreviations (first use should be spelled out)
4. Contradictory definitions or uses of same terms

Respond in JSON format:
{{
    "issues": [
        {{
            "severity": "minor|suggestion",
            "description": "<what's inconsistent>",
            "suggestion": "<recommended standard usage>",
            "evidence": "<examples of inconsistent usage>"
        }}
    ],
    "overall_consistency": <float 0-10>,
    "justification": "<brief assessment>"
}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                issues = []
                for issue_data in result.get('issues', []):
                    severity_str = issue_data.get('severity', 'minor')
                    severity = IssueSeverity(severity_str)

                    issue = ConsistencyIssue(
                        issue_type=ConsistencyIssueType.TERMINOLOGY_INCONSISTENCY,
                        severity=severity,
                        sections_involved=list(all_sections.keys()),
                        description=issue_data.get('description', ''),
                        suggestion=issue_data.get('suggestion', ''),
                        evidence=issue_data.get('evidence', '')
                    )
                    issues.append(issue)

                return issues

        except Exception as e:
            print(f"Terminology consistency check error: {e}")

        return []

    def comprehensive_consistency_check(
        self,
        sections: Dict[SectionType, Section]
    ) -> Dict[str, List[ConsistencyIssue]]:
        """Perform comprehensive consistency check across all sections.

        Args:
            sections: Dictionary of all paper sections

        Returns:
            Dictionary mapping check types to lists of issues
        """
        results = {}

        # Methods-Results alignment
        if SectionType.METHODS in sections and SectionType.RESULTS in sections:
            results['methods_results'] = self.check_methods_results_alignment(
                sections[SectionType.METHODS],
                sections[SectionType.RESULTS]
            )

        # Claims support
        if SectionType.ABSTRACT in sections and SectionType.RESULTS in sections:
            discussion = sections.get(SectionType.DISCUSSION)
            results['claims_support'] = self.check_claims_support(
                sections[SectionType.ABSTRACT],
                sections[SectionType.RESULTS],
                discussion
            )

        # Intro-Discussion coherence
        if SectionType.INTRODUCTION in sections and SectionType.DISCUSSION in sections:
            results['intro_discussion'] = self.check_intro_discussion_coherence(
                sections[SectionType.INTRODUCTION],
                sections[SectionType.DISCUSSION]
            )

        # Terminology consistency
        if len(sections) >= 2:
            results['terminology'] = self.check_terminology_consistency(sections)

        return results


def format_consistency_report(results: Dict[str, List[ConsistencyIssue]]) -> str:
    """Format consistency check results for display.

    Args:
        results: Dictionary of check results

    Returns:
        Formatted string report
    """
    output = []
    output.append("=" * 80)
    output.append("CONSISTENCY CHECK REPORT")
    output.append("=" * 80)

    total_issues = sum(len(issues) for issues in results.values())
    critical = sum(1 for issues in results.values() for i in issues if i.severity == IssueSeverity.CRITICAL)
    major = sum(1 for issues in results.values() for i in issues if i.severity == IssueSeverity.MAJOR)
    minor = sum(1 for issues in results.values() for i in issues if i.severity == IssueSeverity.MINOR)

    output.append(f"\nTotal Issues: {total_issues}")
    output.append(f"  🚨 Critical: {critical}")
    output.append(f"  ⚠️  Major: {major}")
    output.append(f"  ℹ️  Minor: {minor}")
    output.append("")

    for check_type, issues in results.items():
        if not issues:
            continue

        output.append(f"\n{'=' * 80}")
        output.append(f"{check_type.replace('_', ' ').upper()}")
        output.append(f"{'=' * 80}")

        for i, issue in enumerate(issues, 1):
            severity_symbol = {
                IssueSeverity.CRITICAL: "🚨",
                IssueSeverity.MAJOR: "⚠️",
                IssueSeverity.MINOR: "ℹ️",
                IssueSeverity.SUGGESTION: "💡"
            }[issue.severity]

            output.append(f"\n{severity_symbol} Issue #{i} [{issue.severity.value.upper()}]")
            output.append(f"Sections: {', '.join([s.value for s in issue.sections_involved])}")
            output.append(f"Description: {issue.description}")
            output.append(f"Suggestion: {issue.suggestion}")
            if issue.evidence:
                output.append(f"Evidence: {issue.evidence}")
            output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    print("Consistency Checker Module - Use via chatbot or section_evaluator")
