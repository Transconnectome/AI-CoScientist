from dataclasses import dataclass
from typing import List, Dict, Any
import re
import json

@dataclass
class StyleMetrics:
    avg_sentence_length: float
    avg_paragraph_length: float
    vocabulary_richness: float  # Type-Token Ratio (TTR)

class StyleExtractor:
    def __init__(self, llm):
        self.llm = llm

    async def analyze_style(self, text: str) -> StyleMetrics:
        """Analyze text structure metrics."""
        # Sentence splitting (simple regex)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Paragraph splitting
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Word tokenization (simple split)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Calculate metrics
        avg_sent_len = len(words) / len(sentences) if sentences else 0
        avg_para_len = len(words) / len(paragraphs) if paragraphs else 0
        
        # Type-Token Ratio
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        
        return StyleMetrics(
            avg_sentence_length=avg_sent_len,
            avg_paragraph_length=avg_para_len,
            vocabulary_richness=ttr
        )

    async def extract_transitions(self, text: str) -> List[str]:
        """Extract transition phrases using LLM."""
        prompt = f"""Identify the transition phrases and connecting words in the following text.
        Return ONLY a JSON list of strings.
        
        Text:
        {text[:2000]}
        """
        
        response, _ = await self.llm.generate(prompt)
        try:
            # Clean up response if needed (e.g. remove markdown code blocks)
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]
            elif json_str.startswith('```'):
                json_str = json_str[3:-3]
                
            return json.loads(json_str)
        except json.JSONDecodeError:
            return []

    async def analyze_tone(self, text: str) -> Dict[str, Any]:
        """Analyze tone and voice using LLM."""
        prompt = f"""Analyze the tone and voice of the following scientific text.
        Return ONLY a JSON object with keys: "tone", "voice", "confidence".
        
        Text:
        {text[:2000]}
        """
        
        response, _ = await self.llm.generate(prompt)
        try:
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]
            elif json_str.startswith('```'):
                json_str = json_str[3:-3]
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"tone": "unknown", "voice": "unknown", "confidence": "low"}
