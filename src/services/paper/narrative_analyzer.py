"""LLM-based narrative analysis for research papers."""

import json
import re
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class NarrativeResult:
    """Result of narrative analysis."""
    hook_score: float
    tension_curve: list
    story_elements: Dict[str, str]
    feedback: str


class NarrativeAnalyzer:
    """LLM-based narrative analysis for research papers."""
    
    NARRATIVE_PROMPT_TEMPLATE = """You are a senior editor at Nature/Science journals. Your task is to evaluate the narrative structure of a research paper.

Abstract:
{abstract}

Introduction:
{introduction}

Evaluate the following aspects:

1. **Hook Strength (0-10)**: Does the opening immediately grab the reader's attention? Is there a compelling "why should I care?" element?

2. **Gap Definition**: Is the research problem clearly stated? Is there a well-defined gap in current knowledge?

3. **Tension Curve**: Analyze the narrative tension from problem statement to resolution. Rate tension at 4 key points:
   - Opening (0.0-1.0)
   - Problem escalation (0.0-1.0)
   - Climax/novelty reveal (0.0-1.0)
   - Resolution promise (0.0-1.0)

4. **Story Elements**: Extract key narrative components.

Respond ONLY with valid JSON in this exact format:
{{
    "hook_score": <float 0-10>,
    "tension_curve": [<float>, <float>, <float>, <float>],
    "story_elements": {{
        "hook": "<opening hook statement>",
        "gap": "<identified knowledge gap>",
        "resolution": "<proposed solution>"
    }},
    "feedback": "<1-2 sentence editorial feedback>"
}}

Be strict. Most papers are 5-7/10. Only truly exceptional hooks get 9-10."""

    async def analyze_narrative(
        self,
        abstract: str,
        introduction: str,
        llm_service
    ) -> Dict:
        """Analyze narrative structure using LLM.
        
        Args:
            abstract: Paper abstract
            introduction: Paper introduction
            llm_service: LLM service instance with get_response method
            
        Returns:
            Dict with hook_score, tension_curve, story_elements, feedback
        """
        # Format prompt
        prompt = self.NARRATIVE_PROMPT_TEMPLATE.format(
            abstract=abstract,
            introduction=introduction
        )
        
        # Get LLM response
        response = await llm_service.get_response(
            system_prompt="You are a Nature/Science editor evaluating narrative quality.",
            user_message=prompt
        )
        
        # Parse JSON response
        try:
            result = self._parse_response(response)
            return result
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to structured extraction if JSON parsing fails
            return self._fallback_extraction(response, abstract, introduction)
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed dictionary
        """
        # Extract JSON from response (sometimes LLMs add extra text)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
        else:
            data = json.loads(response)
        
        # Validate structure
        required_keys = ["hook_score", "tension_curve", "story_elements", "feedback"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key: {key}")
        
        # Normalize hook_score to 0-10 range
        data["hook_score"] = max(0.0, min(10.0, float(data["hook_score"])))
        
        # Normalize tension_curve values to 0-1 range
        data["tension_curve"] = [
            max(0.0, min(1.0, float(x))) for x in data["tension_curve"]
        ]
        
        return data
    
    def _fallback_extraction(
        self,
        response: str,
        abstract: str,
        introduction: str
    ) -> Dict:
        """Fallback extraction when JSON parsing fails.
        
        Uses simple heuristics as backup.
        """
        # Simple heuristic scoring
        hook_score = 5.0  # Default middle score
        
        # Check for power words
        power_words = ["novel", "first", "breakthrough", "critical", "fundamental"]
        combined_text = (abstract + " " + introduction).lower()
        
        power_word_count = sum(1 for word in power_words if word in combined_text)
        hook_score += min(2.0, power_word_count * 0.5)
        
        # Check for gap indicators
        gap_indicators = ["however", "although", "remains unknown", "challenge"]
        has_gap = any(indicator in combined_text for indicator in gap_indicators)
        
        if has_gap:
            hook_score += 1.0
        
        # Simple tension curve (increasing then resolving)
        tension_curve = [0.3, 0.6, 0.8, 0.5]
        
        return {
            "hook_score": min(10.0, hook_score),
            "tension_curve": tension_curve,
            "story_elements": {
                "hook": abstract[:100] if abstract else "Not extracted",
                "gap": "Gap indicators detected" if has_gap else "No clear gap",
                "resolution": "See full text"
            },
            "feedback": f"Fallback analysis. Hook score: {hook_score:.1f}. LLM parsing failed."
        }
