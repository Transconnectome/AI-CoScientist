"""Extract handcrafted linguistic features for academic papers."""

from typing import List
import torch
import numpy as np
from collections import Counter


class LinguisticFeatureExtractor:
    """Extract 20 linguistic features from academic papers for hybrid model."""

    def __init__(self):
        """Initialize feature extractor with NLP tools."""
        self._nlp = None  # Lazy load spaCy
        self._textstat = None  # Lazy load textstat

        # Academic vocabulary (simplified AWL - top 100 words)
        self.academic_words = {
            "analyze", "approach", "area", "assess", "assume", "authority",
            "available", "benefit", "concept", "consist", "context", "contract",
            "create", "data", "define", "derive", "distribute", "economy",
            "environment", "establish", "estimate", "evidence", "export",
            "factor", "finance", "formula", "function", "identify", "income",
            "indicate", "individual", "interpret", "involve", "issue", "labor",
            "legal", "legislate", "major", "method", "occur", "percent",
            "period", "policy", "principle", "proceed", "process", "require",
            "research", "respond", "role", "section", "sector", "significant",
            "similar", "source", "specific", "structure", "theory", "vary",
            "hypothesis", "methodology", "analysis", "empirical", "quantitative",
            "qualitative", "inference", "correlation", "regression", "variable",
            "parameter", "coefficient", "significance", "validity", "reliability",
            "sample", "population", "distribution", "probability", "statistical",
            "experiment", "observation", "measurement", "instrument", "procedure",
            "intervention", "treatment", "control", "outcome", "effect",
            "implication", "limitation", "conclusion", "recommendation", "future"
        }

        # Technical term patterns
        self.tech_patterns = [
            "ology", "metric", "synthesis", "neural", "algorithm", "coefficient",
            "variance", "hypothesis", "methodology", "empirical", "quantitative"
        ]

    def _get_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except:
                # Fallback to None if spaCy not available
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    def _get_textstat(self):
        """Lazy load textstat."""
        if self._textstat is None:
            try:
                import textstat
                self._textstat = textstat
            except:
                self._textstat = False
        return self._textstat if self._textstat is not False else None

    def extract(self, text: str) -> torch.Tensor:
        """Extract all 20 linguistic features from text.

        Args:
            text: Paper content

        Returns:
            Tensor of 20 linguistic features (normalized 0-1)
        """
        features = []

        # Limit text length for processing
        text = text[:100000]  # First 100k chars

        # Category 1: Readability (4 features)
        features.extend(self._extract_readability(text))

        # Category 2: Vocabulary Richness (4 features)
        features.extend(self._extract_vocabulary(text))

        # Category 3: Syntactic Complexity (4 features)
        features.extend(self._extract_syntax(text))

        # Category 4: Academic Writing Indicators (4 features)
        features.extend(self._extract_academic(text))

        # Category 5: Discourse & Coherence (4 features)
        features.extend(self._extract_coherence(text))

        return torch.tensor(features, dtype=torch.float32)

    # ========== Category 1: Readability Features ==========

    def _extract_readability(self, text: str) -> List[float]:
        """Extract 4 readability features."""
        textstat = self._get_textstat()

        if textstat:
            flesch = textstat.flesch_reading_ease(text) / 100.0
            kincaid = min(textstat.flesch_kincaid_grade(text) / 20.0, 1.0)
            fog = min(textstat.gunning_fog(text) / 20.0, 1.0)
            smog = min(textstat.smog_index(text) / 20.0, 1.0)
        else:
            # Simple heuristic fallback
            words = text.split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            flesch = max(0, 1.0 - avg_word_len / 15.0)
            kincaid = min(avg_word_len / 15.0, 1.0)
            fog = kincaid
            smog = kincaid

        return [flesch, kincaid, fog, smog]

    # ========== Category 2: Vocabulary Richness Features ==========

    def _extract_vocabulary(self, text: str) -> List[float]:
        """Extract 4 vocabulary richness features."""
        words = [w.lower() for w in text.split() if w.isalpha()]

        if not words:
            return [0.0, 0.0, 0.0, 0.0]

        # Type-Token Ratio
        ttr = len(set(words)) / len(words)

        # Unique word ratio (words used only once)
        word_counts = Counter(words)
        unique_ratio = sum(1 for count in word_counts.values() if count == 1) / len(words)

        # Academic word ratio
        academic_ratio = sum(1 for w in words if w in self.academic_words) / len(words)

        # Lexical diversity (simple metric)
        diversity = min(len(set(words)) / 1000, 1.0)  # Normalize by 1000 unique words

        return [ttr, unique_ratio, academic_ratio, diversity]

    # ========== Category 3: Syntactic Complexity Features ==========

    def _extract_syntax(self, text: str) -> List[float]:
        """Extract 4 syntactic complexity features."""
        words = text.split()
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [0.0, 0.0, 0.0, 0.0]

        # Average sentence length
        avg_sent_len = np.mean([len(s.split()) for s in sentences])
        avg_sent_len_norm = min(avg_sent_len / 50.0, 1.0)

        # Average word length
        if words:
            avg_word_len = np.mean([len(w) for w in words if w.isalpha()])
            avg_word_len_norm = min(avg_word_len / 15.0, 1.0)
        else:
            avg_word_len_norm = 0.0

        # Sentence complexity (clause markers)
        clause_markers = ["if", "when", "while", "because", "although", "that", "which"]
        complexity = sum(
            s.lower().count(marker) for s in sentences for marker in clause_markers
        ) / len(sentences)
        complexity_norm = min(complexity / 3.0, 1.0)

        # Dependency depth (approximated by punctuation density)
        punct_density = sum(text.count(p) for p in [',', ';', ':', '-']) / len(words) if words else 0
        punct_density_norm = min(punct_density, 1.0)

        return [avg_sent_len_norm, avg_word_len_norm, complexity_norm, punct_density_norm]

    # ========== Category 4: Academic Writing Indicators ==========

    def _extract_academic(self, text: str) -> List[float]:
        """Extract 4 academic writing indicators."""
        import re

        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return [0.0, 0.0, 0.0, 0.0]

        # Citation density
        citations = len(re.findall(r'\([A-Z][a-z]+,?\s+\d{4}\)|\[\d+\]', text))
        citation_density = min(citations / word_count * 100, 1.0)

        # Technical term ratio
        tech_count = sum(
            1 for word in words
            if any(pattern in word.lower() for pattern in self.tech_patterns)
        )
        tech_ratio = tech_count / word_count

        # Passive voice ratio (simple heuristic: "is/are/was/were" + past participle pattern)
        passive_indicators = text.lower().count(" is ") + text.lower().count(" was ") + \
                             text.lower().count(" are ") + text.lower().count(" were ")
        passive_ratio = min(passive_indicators / word_count * 10, 1.0)

        # Nominalization ratio (words ending in -tion, -ment, -ness, etc.)
        nom_suffixes = ["tion", "sion", "ment", "ness", "ity", "ance", "ence"]
        nom_count = sum(
            1 for word in words
            if any(word.lower().endswith(suffix) for suffix in nom_suffixes)
        )
        nom_ratio = nom_count / word_count

        return [citation_density, tech_ratio, passive_ratio, nom_ratio]

    # ========== Category 5: Discourse & Coherence Features ==========

    def _extract_coherence(self, text: str) -> List[float]:
        """Extract 4 discourse and coherence features."""
        words = text.split()
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences or not words:
            return [0.0, 0.0, 0.0, 0.0]

        # Discourse markers ratio
        markers = [
            "however", "therefore", "furthermore", "moreover", "nevertheless",
            "consequently", "thus", "hence", "additionally", "similarly"
        ]
        marker_count = sum(text.lower().count(marker) for marker in markers)
        marker_ratio = min(marker_count / len(words) * 100, 1.0)

        # Topic consistency (noun repetition between sentences)
        # Simplified: count repeated words between consecutive sentences
        if len(sentences) > 1:
            overlaps = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    overlaps.append(overlap)
            topic_consistency = np.mean(overlaps) if overlaps else 0.0
        else:
            topic_consistency = 1.0

        # Entity coherence (capitalized word repetition)
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 1]
        if capitalized:
            entity_repetition = len(capitalized) / len(set(capitalized))
            entity_coherence = min(entity_repetition / 5.0, 1.0)  # Normalize
        else:
            entity_coherence = 0.0

        # Pronoun ratio
        pronouns = ["it", "they", "this", "these", "those", "which", "that"]
        pronoun_count = sum(text.lower().count(f" {p} ") for p in pronouns)
        pronoun_ratio = min(pronoun_count / len(words), 1.0)

        return [marker_ratio, topic_consistency, entity_coherence, pronoun_ratio]
