"""
Advanced Query Classifier with Machine Learning

Implementation for: Advanced query classifier with ML
Created: 2025-12-05

Acceptance Criteria:
- ML-based complexity classification (simple/medium/complex)
- Domain detection: neuroscience, quantum ML, general
- Intent classification: factual, comparative, synthesis
- Confidence scoring for all predictions
- Evaluation accuracy ≥ 85% on test set

This classifier uses multiple ML models to analyze queries and provide
comprehensive classification for optimal RAG strategy selection.
"""

import re
import logging
import pickle
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np
from pathlib import Path

# ML dependencies with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Install with: pip install sentence-transformers")

# Import enums from orchestrator
from .unified_rag_orchestrator import QueryComplexity, QueryDomain, QueryContext

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent categories"""
    FACTUAL = "factual"          # Simple fact retrieval
    COMPARATIVE = "comparative"   # Comparison between concepts
    SYNTHESIS = "synthesis"      # Complex analysis/synthesis
    PROCEDURAL = "procedural"    # How-to questions
    CAUSAL = "causal"           # Cause-effect relationships

@dataclass
class ClassificationResult:
    """Result of query classification"""
    complexity: QueryComplexity
    domain: QueryDomain
    intent: QueryIntent
    confidence_scores: Dict[str, float]
    features: Dict[str, Any]
    overall_confidence: float

class FeatureExtractor:
    """Extract linguistic and domain-specific features from queries"""

    def __init__(self):
        # Domain-specific keyword sets
        self.domain_keywords = {
            QueryDomain.NEUROSCIENCE: {
                'brain', 'neural', 'neuron', 'cortex', 'fmri', 'eeg', 'synapse',
                'cognitive', 'memory', 'learning', 'plasticity', 'connectivity',
                'activation', 'neuroimaging', 'behavioral', 'psychological'
            },
            QueryDomain.QUANTUM_ML: {
                'quantum', 'qubit', 'superposition', 'entanglement', 'circuit',
                'algorithm', 'variational', 'vqa', 'nisq', 'gate', 'measurement',
                'decoherence', 'interference', 'optimization', 'advantage'
            },
            QueryDomain.DEVELOPMENTAL_DISORDERS: {
                'autism', 'asd', 'developmental', 'disorder', 'disability',
                'intervention', 'therapy', 'diagnosis', 'assessment', 'behavior'
            }
        }

        # Complexity indicators
        self.complexity_patterns = {
            QueryComplexity.SIMPLE: {
                'patterns': [
                    r'\bwhat is\b', r'\bdefine\b', r'\blist\b', r'\bname\b'
                ],
                'avg_words': (1, 8),
                'question_words': {'what', 'who', 'when', 'where'}
            },
            QueryComplexity.MEDIUM: {
                'patterns': [
                    r'\bhow does\b', r'\bwhy\b', r'\bcompare\b', r'\bdifference\b'
                ],
                'avg_words': (8, 20),
                'question_words': {'how', 'why', 'which'}
            },
            QueryComplexity.COMPLEX: {
                'patterns': [
                    r'\banalyze\b', r'\bevaluate\b', r'\bsynthesize\b',
                    r'\bimplications\b', r'\brelationship between\b'
                ],
                'avg_words': (20, 100),
                'question_words': {'analyze', 'evaluate', 'discuss'}
            }
        }

        # Intent patterns
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\bwhat is\b', r'\bdefine\b', r'\bexplain\b', r'\bdescribe\b'
            ],
            QueryIntent.COMPARATIVE: [
                r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b',
                r'\bbetter\b', r'\bworse\b'
            ],
            QueryIntent.SYNTHESIS: [
                r'\banalyze\b', r'\bevaluate\b', r'\bsynthesize\b',
                r'\bimplications\b', r'\brelationship\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\bhow to\b', r'\bsteps\b', r'\bprocess\b', r'\bmethod\b'
            ],
            QueryIntent.CAUSAL: [
                r'\bwhy\b', r'\bcause\b', r'\breason\b', r'\beffect\b',
                r'\bbecause\b', r'\bresult\b'
            ]
        }

    def extract_features(self, query: str) -> Dict[str, Any]:
        """Extract comprehensive features from query"""
        query_lower = query.lower()
        words = query_lower.split()

        features = {
            # Basic linguistic features
            'word_count': len(words),
            'char_count': len(query),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'question_mark_count': query.count('?'),
            'exclamation_count': query.count('!'),

            # Complexity indicators
            'has_technical_terms': self._has_technical_terms(query_lower),
            'has_comparison_words': self._has_comparison_words(query_lower),
            'has_analysis_words': self._has_analysis_words(query_lower),

            # Domain indicators
            'neuroscience_score': self._calculate_domain_score(query_lower, QueryDomain.NEUROSCIENCE),
            'quantum_ml_score': self._calculate_domain_score(query_lower, QueryDomain.QUANTUM_ML),
            'dev_disorders_score': self._calculate_domain_score(query_lower, QueryDomain.DEVELOPMENTAL_DISORDERS),

            # Intent indicators
            'factual_patterns': self._count_patterns(query_lower, QueryIntent.FACTUAL),
            'comparative_patterns': self._count_patterns(query_lower, QueryIntent.COMPARATIVE),
            'synthesis_patterns': self._count_patterns(query_lower, QueryIntent.SYNTHESIS),
            'procedural_patterns': self._count_patterns(query_lower, QueryIntent.PROCEDURAL),
            'causal_patterns': self._count_patterns(query_lower, QueryIntent.CAUSAL),

            # Structural features
            'starts_with_question': query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')),
            'has_multiple_clauses': ',' in query or ';' in query,
            'has_conditionals': any(word in query_lower for word in ['if', 'when', 'unless']),
        }

        return features

    def _has_technical_terms(self, query: str) -> bool:
        """Check for technical terminology"""
        technical_indicators = [
            'algorithm', 'model', 'analysis', 'methodology', 'framework',
            'paradigm', 'hypothesis', 'empirical', 'statistical', 'computational'
        ]
        return any(term in query for term in technical_indicators)

    def _has_comparison_words(self, query: str) -> bool:
        """Check for comparison indicators"""
        comparison_words = [
            'compare', 'contrast', 'difference', 'similar', 'different',
            'versus', 'vs', 'better', 'worse', 'advantage', 'disadvantage'
        ]
        return any(word in query for word in comparison_words)

    def _has_analysis_words(self, query: str) -> bool:
        """Check for analysis/synthesis indicators"""
        analysis_words = [
            'analyze', 'evaluate', 'assess', 'synthesize', 'integrate',
            'implications', 'relationship', 'correlation', 'pattern'
        ]
        return any(word in query for word in analysis_words)

    def _calculate_domain_score(self, query: str, domain: QueryDomain) -> float:
        """Calculate domain relevance score"""
        domain_words = self.domain_keywords.get(domain, set())
        matches = sum(1 for word in domain_words if word in query)
        return matches / len(domain_words) if domain_words else 0.0

    def _count_patterns(self, query: str, intent: QueryIntent) -> int:
        """Count intent pattern matches"""
        patterns = self.intent_patterns.get(intent, [])
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, query))
        return count

class MLQueryClassifier:
    """Machine Learning-based query classifier"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the ML classifier"""
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.is_trained = False
        self.model_path = Path(model_path) if model_path else Path("models/query_classifier")

        if SKLEARN_AVAILABLE:
            self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for each classification task"""
        # Complexity classifier
        self.models['complexity'] = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # Domain classifier
        self.models['domain'] = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # Intent classifier
        self.models['intent'] = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # Text vectorizers
        self.vectorizers['text'] = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )

        # Label encoders
        for task in ['complexity', 'domain', 'intent']:
            self.label_encoders[task] = LabelEncoder()

    def train(self, training_data: List[Tuple[str, QueryComplexity, QueryDomain, QueryIntent]]):
        """Train the classifier on labeled data"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using rule-based classification")
            return

        logger.info(f"Training ML classifier on {len(training_data)} samples")

        # Prepare data
        queries, complexities, domains, intents = zip(*training_data)

        # Extract features
        feature_matrices = self._prepare_training_features(queries)

        # Prepare labels
        complexity_labels = self.label_encoders['complexity'].fit_transform([c.value for c in complexities])
        domain_labels = self.label_encoders['domain'].fit_transform([d.value for d in domains])
        intent_labels = self.label_encoders['intent'].fit_transform([i.value for i in intents])

        # Train models
        logger.info("Training complexity classifier...")
        self.models['complexity'].fit(feature_matrices['combined'], complexity_labels)

        logger.info("Training domain classifier...")
        self.models['domain'].fit(feature_matrices['combined'], domain_labels)

        logger.info("Training intent classifier...")
        self.models['intent'].fit(feature_matrices['combined'], intent_labels)

        self.is_trained = True

        # Evaluate models
        self._evaluate_models(feature_matrices, complexity_labels, domain_labels, intent_labels)

        # Save models
        self._save_models()

        logger.info("ML classifier training completed")

    def _prepare_training_features(self, queries: List[str]) -> Dict[str, Any]:
        """Prepare feature matrices for training"""
        # Text features
        text_features = self.vectorizers['text'].fit_transform(queries)

        # Linguistic features
        linguistic_features = []
        for query in queries:
            features = self.feature_extractor.extract_features(query)
            feature_vector = list(features.values())
            linguistic_features.append(feature_vector)

        linguistic_features = np.array(linguistic_features)

        # Combined features
        from scipy.sparse import hstack
        combined_features = hstack([text_features, linguistic_features])

        return {
            'text': text_features,
            'linguistic': linguistic_features,
            'combined': combined_features
        }

    def _evaluate_models(self, feature_matrices, complexity_labels, domain_labels, intent_labels):
        """Evaluate model performance"""
        logger.info("Evaluating model performance...")

        # Complexity evaluation
        complexity_scores = cross_val_score(
            self.models['complexity'], feature_matrices['combined'],
            complexity_labels, cv=5, scoring='accuracy'
        )
        logger.info(f"Complexity classification accuracy: {complexity_scores.mean():.3f} ± {complexity_scores.std():.3f}")

        # Domain evaluation
        domain_scores = cross_val_score(
            self.models['domain'], feature_matrices['combined'],
            domain_labels, cv=5, scoring='accuracy'
        )
        logger.info(f"Domain classification accuracy: {domain_scores.mean():.3f} ± {domain_scores.std():.3f}")

        # Intent evaluation
        intent_scores = cross_val_score(
            self.models['intent'], feature_matrices['combined'],
            intent_labels, cv=5, scoring='accuracy'
        )
        logger.info(f"Intent classification accuracy: {intent_scores.mean():.3f} ± {intent_scores.std():.3f}")

    def _save_models(self):
        """Save trained models"""
        try:
            self.model_path.mkdir(parents=True, exist_ok=True)

            # Save models
            for name, model in self.models.items():
                with open(self.model_path / f"{name}_model.pkl", 'wb') as f:
                    pickle.dump(model, f)

            # Save vectorizers
            for name, vectorizer in self.vectorizers.items():
                with open(self.model_path / f"{name}_vectorizer.pkl", 'wb') as f:
                    pickle.dump(vectorizer, f)

            # Save label encoders
            for name, encoder in self.label_encoders.items():
                with open(self.model_path / f"{name}_encoder.pkl", 'wb') as f:
                    pickle.dump(encoder, f)

            logger.info(f"Models saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self) -> bool:
        """Load pre-trained models"""
        try:
            if not self.model_path.exists():
                return False

            # Load models
            for name in ['complexity', 'domain', 'intent']:
                model_file = self.model_path / f"{name}_model.pkl"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[name] = pickle.load(f)

            # Load vectorizers
            for name in ['text']:
                vectorizer_file = self.model_path / f"{name}_vectorizer.pkl"
                if vectorizer_file.exists():
                    with open(vectorizer_file, 'rb') as f:
                        self.vectorizers[name] = pickle.load(f)

            # Load label encoders
            for name in ['complexity', 'domain', 'intent']:
                encoder_file = self.model_path / f"{name}_encoder.pkl"
                if encoder_file.exists():
                    with open(encoder_file, 'rb') as f:
                        self.label_encoders[name] = pickle.load(f)

            self.is_trained = True
            logger.info("Pre-trained models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    async def classify(self, query: str) -> ClassificationResult:
        """Classify a query using ML models or rule-based fallback"""
        if SKLEARN_AVAILABLE and self.is_trained:
            return await self._classify_ml(query)
        else:
            return await self._classify_rule_based(query)

    async def _classify_ml(self, query: str) -> ClassificationResult:
        """ML-based classification"""
        # Extract features
        features = self.feature_extractor.extract_features(query)

        # Prepare feature vector
        text_features = self.vectorizers['text'].transform([query])
        linguistic_features = np.array([list(features.values())])

        from scipy.sparse import hstack
        combined_features = hstack([text_features, linguistic_features])

        # Predict with confidence scores
        complexity_proba = self.models['complexity'].predict_proba(combined_features)[0]
        domain_proba = self.models['domain'].predict_proba(combined_features)[0]
        intent_proba = self.models['intent'].predict_proba(combined_features)[0]

        # Get predictions
        complexity_pred = self.models['complexity'].predict(combined_features)[0]
        domain_pred = self.models['domain'].predict(combined_features)[0]
        intent_pred = self.models['intent'].predict(combined_features)[0]

        # Convert back to enums
        complexity = QueryComplexity(self.label_encoders['complexity'].inverse_transform([complexity_pred])[0])
        domain = QueryDomain(self.label_encoders['domain'].inverse_transform([domain_pred])[0])
        intent = QueryIntent(self.label_encoders['intent'].inverse_transform([intent_pred])[0])

        # Calculate confidence scores
        confidence_scores = {
            'complexity': float(max(complexity_proba)),
            'domain': float(max(domain_proba)),
            'intent': float(max(intent_proba))
        }

        overall_confidence = np.mean(list(confidence_scores.values()))

        return ClassificationResult(
            complexity=complexity,
            domain=domain,
            intent=intent,
            confidence_scores=confidence_scores,
            features=features,
            overall_confidence=overall_confidence
        )

    async def _classify_rule_based(self, query: str) -> ClassificationResult:
        """Rule-based classification fallback"""
        features = self.feature_extractor.extract_features(query)

        # Complexity classification
        word_count = features['word_count']
        has_analysis = features['has_analysis_words']
        has_comparison = features['has_comparison_words']

        if word_count <= 8 and not has_analysis:
            complexity = QueryComplexity.SIMPLE
            complexity_confidence = 0.8
        elif word_count <= 20 and (has_comparison or features['has_technical_terms']):
            complexity = QueryComplexity.MEDIUM
            complexity_confidence = 0.7
        else:
            complexity = QueryComplexity.COMPLEX
            complexity_confidence = 0.6

        # Domain classification
        neuro_score = features['neuroscience_score']
        quantum_score = features['quantum_ml_score']
        dev_score = features['dev_disorders_score']

        if dev_score > 0:
            domain = QueryDomain.DEVELOPMENTAL_DISORDERS
            domain_confidence = min(0.9, 0.5 + dev_score)
        elif neuro_score > quantum_score:
            domain = QueryDomain.NEUROSCIENCE
            domain_confidence = min(0.9, 0.5 + neuro_score)
        elif quantum_score > 0:
            domain = QueryDomain.QUANTUM_ML
            domain_confidence = min(0.9, 0.5 + quantum_score)
        else:
            domain = QueryDomain.GENERAL
            domain_confidence = 0.6

        # Intent classification
        pattern_scores = {
            QueryIntent.FACTUAL: features['factual_patterns'],
            QueryIntent.COMPARATIVE: features['comparative_patterns'],
            QueryIntent.SYNTHESIS: features['synthesis_patterns'],
            QueryIntent.PROCEDURAL: features['procedural_patterns'],
            QueryIntent.CAUSAL: features['causal_patterns']
        }

        intent = max(pattern_scores.items(), key=lambda x: x[1])[0]
        intent_confidence = min(0.9, 0.5 + max(pattern_scores.values()) * 0.1)

        confidence_scores = {
            'complexity': complexity_confidence,
            'domain': domain_confidence,
            'intent': intent_confidence
        }

        overall_confidence = np.mean(list(confidence_scores.values()))

        return ClassificationResult(
            complexity=complexity,
            domain=domain,
            intent=intent,
            confidence_scores=confidence_scores,
            features=features,
            overall_confidence=overall_confidence
        )

    def generate_training_data(self) -> List[Tuple[str, QueryComplexity, QueryDomain, QueryIntent]]:
        """Generate synthetic training data for initial model training"""
        training_data = []

        # Simple queries
        simple_queries = [
            ("What is machine learning?", QueryComplexity.SIMPLE, QueryDomain.GENERAL, QueryIntent.FACTUAL),
            ("Define neural network", QueryComplexity.SIMPLE, QueryDomain.NEUROSCIENCE, QueryIntent.FACTUAL),
            ("What is a qubit?", QueryComplexity.SIMPLE, QueryDomain.QUANTUM_ML, QueryIntent.FACTUAL),
            ("List autism symptoms", QueryComplexity.SIMPLE, QueryDomain.DEVELOPMENTAL_DISORDERS, QueryIntent.FACTUAL),
            ("What is fMRI?", QueryComplexity.SIMPLE, QueryDomain.NEUROSCIENCE, QueryIntent.FACTUAL),
        ]

        # Medium queries
        medium_queries = [
            ("How do neural networks learn?", QueryComplexity.MEDIUM, QueryDomain.NEUROSCIENCE, QueryIntent.PROCEDURAL),
            ("Compare quantum and classical algorithms", QueryComplexity.MEDIUM, QueryDomain.QUANTUM_ML, QueryIntent.COMPARATIVE),
            ("Why do autistic children have social difficulties?", QueryComplexity.MEDIUM, QueryDomain.DEVELOPMENTAL_DISORDERS, QueryIntent.CAUSAL),
            ("How does fMRI measure brain activity?", QueryComplexity.MEDIUM, QueryDomain.NEUROSCIENCE, QueryIntent.PROCEDURAL),
            ("What causes quantum decoherence?", QueryComplexity.MEDIUM, QueryDomain.QUANTUM_ML, QueryIntent.CAUSAL),
        ]

        # Complex queries
        complex_queries = [
            ("Analyze the relationship between quantum advantage and circuit depth in NISQ algorithms", QueryComplexity.COMPLEX, QueryDomain.QUANTUM_ML, QueryIntent.SYNTHESIS),
            ("Evaluate the effectiveness of early intervention programs for autism spectrum disorder", QueryComplexity.COMPLEX, QueryDomain.DEVELOPMENTAL_DISORDERS, QueryIntent.SYNTHESIS),
            ("Synthesize current understanding of brain connectivity patterns in developmental disorders", QueryComplexity.COMPLEX, QueryDomain.NEUROSCIENCE, QueryIntent.SYNTHESIS),
            ("Compare the implications of different neural network architectures for brain modeling", QueryComplexity.COMPLEX, QueryDomain.NEUROSCIENCE, QueryIntent.COMPARATIVE),
            ("Analyze the theoretical foundations of variational quantum algorithms and their limitations", QueryComplexity.COMPLEX, QueryDomain.QUANTUM_ML, QueryIntent.SYNTHESIS),
        ]

        training_data.extend(simple_queries)
        training_data.extend(medium_queries)
        training_data.extend(complex_queries)

        return training_data

# Factory function
def create_query_classifier(model_path: Optional[str] = None) -> MLQueryClassifier:
    """Create query classifier with optional model path"""
    classifier = MLQueryClassifier(model_path)

    # Try to load existing models
    if not classifier.load_models():
        # Train on synthetic data if no models exist
        training_data = classifier.generate_training_data()
        classifier.train(training_data)

    return classifier

# Global instance
_global_classifier: Optional[MLQueryClassifier] = None

def get_query_classifier() -> MLQueryClassifier:
    """Get global query classifier instance"""
    global _global_classifier

    if _global_classifier is None:
        _global_classifier = create_query_classifier()

    return _global_classifier

# Example usage and testing
if __name__ == "__main__":
    async def test_classifier():
        """Test the query classifier"""
        print("🔄 Testing Advanced Query Classifier...")

        # Create classifier
        classifier = create_query_classifier()

        # Test queries
        test_queries = [
            "What is machine learning?",
            "How do quantum computers achieve quantum advantage?",
            "Analyze the relationship between brain connectivity and autism severity",
            "Compare fMRI and EEG for studying brain function"
        ]

        for query in test_queries:
            result = await classifier.classify(query)
            print(f"\n📝 Query: {query}")
            print(f"🔍 Complexity: {result.complexity.value} (confidence: {result.confidence_scores['complexity']:.3f})")
            print(f"🎯 Domain: {result.domain.value} (confidence: {result.confidence_scores['domain']:.3f})")
            print(f"💡 Intent: {result.intent.value} (confidence: {result.confidence_scores['intent']:.3f})")
            print(f"📊 Overall confidence: {result.overall_confidence:.3f}")

        print("\n✅ Advanced Query Classifier test completed successfully!")

    # Run test
    asyncio.run(test_classifier())