from typing import List, Optional
from src.services.paper.style_extractor import StyleMetrics

class StyleTransferService:
    """Manages style guides based on Golden Reference papers."""

    def __init__(self):
        # Default Nature-style metrics (could be loaded from DB)
        self.default_metrics = StyleMetrics(
            avg_sentence_length=20.0,
            avg_paragraph_length=150.0,
            vocabulary_richness=0.6
        )
        self.default_transitions = ["However", "Furthermore", "In contrast", "Notably"]
        self.default_tone = "objective, concise, and authoritative"

    def construct_style_guide(self, metrics: Optional[StyleMetrics] = None) -> str:
        """Construct a style guide prompt from metrics."""
        m = metrics or self.default_metrics
        
        guide = f"""
        **Writing Style Guidelines (Nature-like):**
        1. **Sentence Structure**: Aim for an average sentence length of {m.avg_sentence_length:.1f} words. Mix short, punchy sentences with longer explanatory ones.
        2. **Vocabulary**: Use precise, high-level scientific vocabulary (Type-Token Ratio target: {m.vocabulary_richness:.2f}).
        3. **Tone**: Maintain a {self.default_tone} tone. Avoid hyperbole.
        4. **Transitions**: Use transition words effectively (e.g., {', '.join(self.default_transitions[:3])}) to ensure flow.
        5. **Paragraphs**: Keep paragraphs focused (approx. {m.avg_paragraph_length:.0f} words).
        """
        return guide.strip()
