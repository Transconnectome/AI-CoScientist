#!/usr/bin/env python3
"""
Query Classification System for Hybrid DD Search
Classifies queries as clinical, technical, or mixed to optimize search weighting
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query type classification"""
    CLINICAL = "clinical"
    TECHNICAL = "technical"
    MIXED = "mixed"


@dataclass
class QueryClassification:
    """Query classification result"""
    query_type: QueryType
    clinical_score: float
    technical_score: float
    confidence: float
    keywords_matched: Dict[str, List[str]]
    reasoning: str


class QueryClassifier:
    """
    Intelligent query classifier for hybrid DD search.

    Determines whether a query is:
    - Clinical: Focus on DD research, diagnosis, treatment, behavioral studies
    - Technical: Focus on AI/ML architecture, models, training methods
    - Mixed: Combines both clinical and technical aspects
    """

    def __init__(self):
        # Clinical keywords (DD research focus)
        self.clinical_keywords = {
            'disorders': [
                'autism', 'asd', 'adhd', 'developmental disorder', 'neurodevelopmental',
                'asperger', 'learning disability', 'intellectual disability',
                'down syndrome', 'fragile x', 'rett syndrome'
            ],
            'symptoms': [
                'behavioral', 'social interaction', 'communication deficit',
                'repetitive behavior', 'sensory processing', 'motor coordination',
                'attention deficit', 'hyperactivity', 'impulsivity'
            ],
            'diagnosis': [
                'diagnosis', 'screening', 'assessment', 'clinical evaluation',
                'biomarker', 'phenotype', 'endophenotype', 'symptom severity'
            ],
            'treatment': [
                'intervention', 'therapy', 'treatment', 'medication',
                'behavioral therapy', 'cognitive training', 'rehabilitation'
            ],
            'neuroscience': [
                'brain imaging', 'fmri', 'eeg', 'dmri', 'mri', 'pet scan',
                'neural circuit', 'brain region', 'cortex', 'connectivity',
                'functional connectivity', 'structural connectivity'
            ],
            'population': [
                'pediatric', 'children', 'adolescent', 'adult', 'age group',
                'typically developing', 'patient', 'subject', 'cohort'
            ]
        }

        # Technical keywords (AI/ML focus)
        self.technical_keywords = {
            'architecture': [
                'transformer', 'attention mechanism', 'neural network', 'cnn', 'rnn',
                'lstm', 'gru', 'encoder', 'decoder', 'autoencoder', 'vae',
                'gan', 'diffusion model', 'architecture', 'layer', 'block'
            ],
            'models': [
                'foundation model', 'large language model', 'llm', 'multimodal model',
                'vision-language model', 'clip', 'blip', 'bert', 'gpt',
                'pretrained model', 'fine-tuning', 'transfer learning'
            ],
            'training': [
                'training', 'optimization', 'gradient descent', 'backpropagation',
                'loss function', 'learning rate', 'batch size', 'epoch',
                'convergence', 'overfitting', 'regularization', 'dropout'
            ],
            'data': [
                'embedding', 'representation', 'feature extraction', 'dimensionality reduction',
                'data augmentation', 'preprocessing', 'tokenization', 'encoding'
            ],
            'performance': [
                'accuracy', 'precision', 'recall', 'f1 score', 'auc', 'roc',
                'benchmark', 'evaluation metric', 'performance', 'efficiency'
            ],
            'methods': [
                'self-supervised learning', 'contrastive learning', 'few-shot learning',
                'zero-shot learning', 'meta-learning', 'reinforcement learning',
                'active learning', 'semi-supervised learning'
            ]
        }

        # Domain overlap keywords (both clinical and technical)
        self.overlap_keywords = {
            'ai_neuroscience': [
                'neuroimaging ai', 'deep learning for diagnosis', 'ml for autism',
                'ai-assisted diagnosis', 'automated screening', 'computer-aided diagnosis',
                'predictive modeling', 'classification model', 'detection algorithm'
            ],
            'multimodal': [
                'multimodal', 'multi-modal', 'cross-modal', 'fusion',
                'multimodal learning', 'multimodal integration'
            ],
            'analysis': [
                'analysis', 'classification', 'prediction', 'detection',
                'segmentation', 'clustering', 'pattern recognition'
            ]
        }

    def classify(self, query: str) -> QueryClassification:
        """
        Classify query into clinical, technical, or mixed type.

        Args:
            query: Search query string

        Returns:
            QueryClassification with scores and reasoning
        """
        query_lower = query.lower()

        # Count keyword matches
        clinical_matches = self._count_keyword_matches(query_lower, self.clinical_keywords)
        technical_matches = self._count_keyword_matches(query_lower, self.technical_keywords)
        overlap_matches = self._count_keyword_matches(query_lower, self.overlap_keywords)

        # Calculate scores
        total_clinical = clinical_matches['total'] + overlap_matches['total'] * 0.5
        total_technical = technical_matches['total'] + overlap_matches['total'] * 0.5
        total_matches = total_clinical + total_technical

        if total_matches == 0:
            # No keywords matched - default to mixed with low confidence
            return QueryClassification(
                query_type=QueryType.MIXED,
                clinical_score=0.5,
                technical_score=0.5,
                confidence=0.3,
                keywords_matched={},
                reasoning="No domain-specific keywords found. Using balanced search."
            )

        # Normalize scores
        clinical_score = total_clinical / total_matches if total_matches > 0 else 0
        technical_score = total_technical / total_matches if total_matches > 0 else 0

        # Determine query type and confidence
        threshold = 0.35  # If either score > 65%, it's dominant

        if clinical_score > (1 - threshold) and technical_score < threshold:
            query_type = QueryType.CLINICAL
            confidence = clinical_score
            reasoning = f"Strong clinical focus ({clinical_score:.2%}). Prioritizing DD papers."
        elif technical_score > (1 - threshold) and clinical_score < threshold:
            query_type = QueryType.TECHNICAL
            confidence = technical_score
            reasoning = f"Strong technical focus ({technical_score:.2%}). Prioritizing FM papers."
        else:
            query_type = QueryType.MIXED
            confidence = min(clinical_score, technical_score) / max(clinical_score, technical_score)
            reasoning = (
                f"Mixed query (clinical: {clinical_score:.2%}, technical: {technical_score:.2%}). "
                f"Using balanced search with adaptive weighting."
            )

        # Collect all matched keywords
        all_matches = {
            'clinical': clinical_matches['categories'],
            'technical': technical_matches['categories'],
            'overlap': overlap_matches['categories']
        }

        return QueryClassification(
            query_type=query_type,
            clinical_score=clinical_score,
            technical_score=technical_score,
            confidence=confidence,
            keywords_matched=all_matches,
            reasoning=reasoning
        )

    def _count_keyword_matches(
        self,
        query: str,
        keyword_dict: Dict[str, List[str]]
    ) -> Dict:
        """Count keyword matches in query."""
        total_matches = 0
        category_matches = {}

        for category, keywords in keyword_dict.items():
            matches = []
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, query):
                    matches.append(keyword)
                    total_matches += 1

            if matches:
                category_matches[category] = matches

        return {
            'total': total_matches,
            'categories': category_matches
        }

    def get_search_weights(self, classification: QueryClassification) -> Tuple[float, float]:
        """
        Get search weights for DD and FM databases based on classification.

        Returns:
            (dd_weight, fm_weight) tuple
        """
        if classification.query_type == QueryType.CLINICAL:
            # Strong clinical focus - prioritize DD papers
            dd_weight = 2.0 + classification.clinical_score
            fm_weight = 1.0
        elif classification.query_type == QueryType.TECHNICAL:
            # Strong technical focus - prioritize FM papers
            dd_weight = 1.0
            fm_weight = 2.0 + classification.technical_score
        else:
            # Mixed query - adaptive weighting based on scores
            dd_weight = 1.0 + classification.clinical_score
            fm_weight = 1.0 + classification.technical_score

        return dd_weight, fm_weight

    def explain_classification(self, classification: QueryClassification) -> str:
        """Generate detailed explanation of classification."""
        explanation = [
            f"Query Type: {classification.query_type.value.upper()}",
            f"Confidence: {classification.confidence:.2%}",
            f"Clinical Score: {classification.clinical_score:.2%}",
            f"Technical Score: {classification.technical_score:.2%}",
            f"\nReasoning: {classification.reasoning}",
        ]

        if classification.keywords_matched:
            explanation.append("\nKeywords Matched:")
            for domain, categories in classification.keywords_matched.items():
                if categories:
                    explanation.append(f"  {domain.upper()}:")
                    for category, keywords in categories.items():
                        explanation.append(f"    {category}: {', '.join(keywords)}")

        return '\n'.join(explanation)


def test_classifier():
    """Test the query classifier with sample queries."""
    classifier = QueryClassifier()

    test_queries = [
        "autism diagnosis using EEG signals",
        "ADHD treatment effectiveness",
        "transformer architecture for brain imaging",
        "foundation models for developmental disorders",
        "multimodal AI in neuroscience",
        "behavioral therapy for children with ASD",
        "attention mechanism in neural networks",
        "fMRI connectivity analysis in autism spectrum disorder",
        "large language models for medical diagnosis",
        "clinical assessment of neurodevelopmental disorders"
    ]

    print("Query Classification Test Results")
    print("=" * 80)

    for query in test_queries:
        classification = classifier.classify(query)
        weights = classifier.get_search_weights(classification)

        print(f"\nQuery: '{query}'")
        print(f"Type: {classification.query_type.value.upper()}")
        print(f"Scores: Clinical={classification.clinical_score:.2%}, Technical={classification.technical_score:.2%}")
        print(f"Weights: DD={weights[0]:.2f}, FM={weights[1]:.2f}")
        print(f"Reasoning: {classification.reasoning}")


if __name__ == "__main__":
    test_classifier()
