#!/usr/bin/env python3
"""Paper section parser with intelligent section detection.

This module provides functionality to:
1. Detect and extract paper sections (Abstract, Introduction, Methods, Results, Discussion)
2. Handle various section naming conventions
3. Support both complete and partial papers
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SectionType(Enum):
    """Standard paper section types."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    UNKNOWN = "unknown"


@dataclass
class Section:
    """Represents a paper section."""
    type: SectionType
    title: str
    content: str
    start_pos: int
    end_pos: int
    word_count: int

    def __str__(self):
        return f"{self.type.value}: {self.word_count} words"


class PaperSectionParser:
    """Parse and extract sections from scientific papers."""

    # Section heading patterns
    SECTION_PATTERNS = {
        SectionType.ABSTRACT: [
            r'\n\s*abstract\s*\n',
            r'\n\s*summary\s*\n',
        ],
        SectionType.INTRODUCTION: [
            r'\n\s*(?:1\.?\s*)?introduction\s*\n',
            r'\n\s*background\s*\n',
        ],
        SectionType.METHODS: [
            r'\n\s*(?:2\.?\s*)?methods?\s*\n',
            r'\n\s*(?:2\.?\s*)?materials?\s+and\s+methods?\s*\n',
            r'\n\s*(?:2\.?\s*)?methodology\s*\n',
            r'\n\s*(?:2\.?\s*)?experimental\s+(?:design|procedure|setup)\s*\n',
        ],
        SectionType.RESULTS: [
            r'\n\s*(?:3\.?\s*)?results?\s*\n',
            r'\n\s*(?:3\.?\s*)?findings?\s*\n',
        ],
        SectionType.DISCUSSION: [
            r'\n\s*(?:4\.?\s*)?discussion\s*\n',
            r'\n\s*(?:4\.?\s*)?interpretation\s*\n',
            r'\n\s*(?:4\.?\s*)?results?\s+and\s+discussion\s*\n',
        ],
        SectionType.CONCLUSION: [
            r'\n\s*(?:5\.?\s*)?conclusion\s*\n',
            r'\n\s*(?:5\.?\s*)?concluding\s+remarks?\s*\n',
        ],
        SectionType.REFERENCES: [
            r'\n\s*references?\s*\n',
            r'\n\s*bibliography\s*\n',
            r'\n\s*works?\s+cited\s*\n',
        ]
    }

    def __init__(self):
        """Initialize the parser."""
        self.sections: List[Section] = []

    def parse(self, text: str) -> Dict[SectionType, Section]:
        """Parse paper text and extract sections.

        Args:
            text: Full paper text

        Returns:
            Dictionary mapping section types to Section objects
        """
        text_lower = text.lower()

        # Find all section boundaries
        boundaries = self._find_section_boundaries(text_lower)

        # Extract sections
        sections = {}
        for i, (section_type, start, title) in enumerate(boundaries):
            # Determine end position
            if i < len(boundaries) - 1:
                end = boundaries[i + 1][1]
            else:
                end = len(text)

            # Extract content (skip the title line)
            content = text[start:end].strip()

            # Calculate word count
            word_count = len(content.split())

            # Create section object
            section = Section(
                type=section_type,
                title=title,
                content=content,
                start_pos=start,
                end_pos=end,
                word_count=word_count
            )

            sections[section_type] = section

        self.sections = list(sections.values())
        return sections

    def _find_section_boundaries(self, text_lower: str) -> List[Tuple[SectionType, int, str]]:
        """Find all section boundaries in the text.

        Args:
            text_lower: Lowercased paper text

        Returns:
            List of (section_type, position, title) tuples
        """
        boundaries = []

        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    pos = match.start()
                    title = match.group(0).strip()
                    boundaries.append((section_type, pos, title))

        # Sort by position
        boundaries.sort(key=lambda x: x[1])

        # Remove duplicates (keep first occurrence)
        seen_types = set()
        unique_boundaries = []
        for boundary in boundaries:
            if boundary[0] not in seen_types:
                unique_boundaries.append(boundary)
                seen_types.add(boundary[0])

        return unique_boundaries

    def get_section(self, section_type: SectionType) -> Optional[Section]:
        """Get a specific section.

        Args:
            section_type: Type of section to retrieve

        Returns:
            Section object or None if not found
        """
        for section in self.sections:
            if section.type == section_type:
                return section
        return None

    def has_section(self, section_type: SectionType) -> bool:
        """Check if paper has a specific section.

        Args:
            section_type: Type of section to check

        Returns:
            True if section exists
        """
        return self.get_section(section_type) is not None

    def get_completion_status(self) -> Dict[str, any]:
        """Get paper completion status.

        Returns:
            Dictionary with completion information
        """
        total_sections = 5  # Abstract, Intro, Methods, Results, Discussion
        core_sections = [
            SectionType.ABSTRACT,
            SectionType.INTRODUCTION,
            SectionType.METHODS,
            SectionType.RESULTS,
            SectionType.DISCUSSION
        ]

        completed = sum(1 for s in core_sections if self.has_section(s))

        status = {
            'total_sections': total_sections,
            'completed_sections': completed,
            'completion_percentage': (completed / total_sections) * 100,
            'sections_present': [s.value for s in core_sections if self.has_section(s)],
            'sections_missing': [s.value for s in core_sections if not self.has_section(s)],
            'word_counts': {
                s.type.value: s.word_count
                for s in self.sections
                if s.type in core_sections
            }
        }

        return status

    def detect_sections_simple(self, text: str) -> Dict[str, str]:
        """Simple section detection for chatbot use.

        Args:
            text: Paper text

        Returns:
            Dictionary mapping section names to content
        """
        sections_dict = self.parse(text)

        return {
            section_type.value: section.content
            for section_type, section in sections_dict.items()
        }


def parse_paper_sections(text: str) -> Dict[SectionType, Section]:
    """Convenience function to parse paper sections.

    Args:
        text: Paper text

    Returns:
        Dictionary of sections
    """
    parser = PaperSectionParser()
    return parser.parse(text)


def get_section_content(text: str, section_name: str) -> Optional[str]:
    """Get content of a specific section.

    Args:
        text: Paper text
        section_name: Name of section (e.g., 'abstract', 'methods')

    Returns:
        Section content or None if not found
    """
    parser = PaperSectionParser()
    sections = parser.parse(text)

    # Try to find matching section type
    try:
        section_type = SectionType(section_name.lower())
        if section_type in sections:
            return sections[section_type].content
    except ValueError:
        pass

    return None


if __name__ == "__main__":
    # Test the parser
    sample_paper = """
    Abstract

    This is the abstract of the paper. It contains background, methods, results.

    1. Introduction

    This is the introduction section with background information.

    2. Methods

    This section describes the methodology used in the study.

    3. Results

    Here are the results of our experiments.

    4. Discussion

    We discuss the implications of our findings.

    References

    List of references.
    """

    parser = PaperSectionParser()
    sections = parser.parse(sample_paper)

    print("Detected sections:")
    for section_type, section in sections.items():
        print(f"  {section}")

    print("\nCompletion status:")
    status = parser.get_completion_status()
    print(f"  {status['completed_sections']}/{status['total_sections']} sections "
          f"({status['completion_percentage']:.0f}%)")
    print(f"  Present: {', '.join(status['sections_present'])}")
    print(f"  Missing: {', '.join(status['sections_missing'])}")
