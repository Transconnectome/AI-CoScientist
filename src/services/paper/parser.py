"""Paper parsing service for extracting structure from academic papers."""

import json
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.llm.service import LLMService
from src.models.project import Paper


class PaperParser:
    """Service for parsing academic papers into structured sections."""

    def __init__(self, llm_service: LLMService):
        """Initialize parser with LLM service.

        Args:
            llm_service: LLM service for intelligent text parsing
        """
        self.llm = llm_service

    async def parse_text(self, text: str) -> dict[str, str]:
        """Parse paper text into structured sections.

        Uses LLM to intelligently identify and extract standard academic
        paper sections (Abstract, Introduction, Methods, Results, Discussion, etc.).

        Args:
            text: Full paper text content

        Returns:
            Dictionary mapping section names to content

        Example:
            {
                "abstract": "This paper explores...",
                "introduction": "Machine learning has...",
                "methods": "We collected data...",
                "results": "Our model achieved...",
                "discussion": "These findings...",
                "references": "1. Smith et al..."
            }
        """
        prompt = f"""
        Analyze the following academic paper text and extract its sections.
        Identify standard sections like Abstract, Introduction, Methods, Results, Discussion, Conclusion, and References.

        Return the result as a JSON object where keys are lowercase section names and values are the section content.
        If a section is not found, omit it from the result.

        Paper text:
        {text}

        Return only valid JSON, no additional text.
        Example format:
        {{
            "abstract": "content here",
            "introduction": "content here",
            "methods": "content here"
        }}
        """

        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.1  # Low temperature for structured extraction
            )

            # Parse LLM response as JSON
            sections = json.loads(result.content)

            # Validate structure
            if not isinstance(sections, dict):
                raise ValueError("LLM response is not a dictionary")

            return sections

        except json.JSONDecodeError as e:
            # Fallback: try to extract JSON from response
            content = result.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                sections = json.loads(content[start:end])
                return sections
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    async def extract_sections(self, text: str) -> list[dict]:
        """Extract sections as ordered list.

        Args:
            text: Full paper text

        Returns:
            List of section dictionaries with name, content, and order

        Example:
            [
                {"name": "abstract", "content": "...", "order": 0},
                {"name": "introduction", "content": "...", "order": 1},
                ...
            ]
        """
        parsed = await self.parse_text(text)

        # Define standard section order
        standard_order = [
            "abstract",
            "introduction",
            "related_work",
            "background",
            "methods",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references"
        ]

        sections = []
        order = 0

        # Add sections in standard order
        for section_name in standard_order:
            if section_name in parsed:
                sections.append({
                    "name": section_name,
                    "content": parsed[section_name],
                    "order": order
                })
                order += 1

        # Add any remaining sections not in standard order
        for section_name, content in parsed.items():
            if section_name not in standard_order:
                sections.append({
                    "name": section_name,
                    "content": content,
                    "order": order
                })
                order += 1

        return sections

    async def extract_metadata(self, text: str) -> dict:
        """Extract paper metadata (title, authors, abstract).

        Args:
            text: Full paper text

        Returns:
            Dictionary with metadata fields
        """
        prompt = f"""
        Extract metadata from this academic paper.

        Return as JSON with these fields:
        - title: Paper title
        - authors: List of author names
        - abstract: Abstract text (if present)
        - keywords: List of keywords (if present)

        Paper text:
        {text[:2000]}  # First 2000 chars usually contain metadata

        Return only valid JSON.
        """

        try:
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.1
            )

            metadata = json.loads(result.content)
            return metadata

        except json.JSONDecodeError:
            # Return minimal metadata
            return {
                "title": "Untitled",
                "authors": [],
                "abstract": "",
                "keywords": []
            }
