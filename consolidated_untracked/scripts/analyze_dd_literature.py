#!/usr/bin/env python3
"""
Comprehensive Scientific Literature Analysis of DD-RAPTOR Knowledge Base

This script conducts rigorous analysis of 26 developmental disorder papers to identify:
1. State-of-the-art diagnostic methods, biomarkers, and interventions
2. Critical research gaps and unsolved questions
3. Methodological limitations
4. Future directions and emerging approaches

Usage:
    poetry run python scripts/analyze_dd_literature.py
"""

import chromadb
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import re

from sentence_transformers import SentenceTransformer, CrossEncoder


class DDLiteratureAnalyzer:
    """Comprehensive literature analyzer for developmental disorder research"""

    def __init__(self, db_path: str = "chromadb_data_dd"):
        self.db_path = Path(db_path)
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.cross_encoder = None

        # Analysis queries organized by theme
        self.query_themes = {
            "biomarkers_diagnostics": [
                "early biomarkers autism spectrum disorder prediction",
                "machine learning diagnostic accuracy ASD ADHD",
                "digital biomarkers behavioral analysis"
            ],
            "neuroimaging_methods": [
                "multimodal neuroimaging developmental disorders",
                "longitudinal brain development trajectories",
                "structural functional connectivity autism"
            ],
            "interventions_precision": [
                "precision medicine developmental disorders",
                "personalized interventions autism treatment",
                "therapeutic outcomes developmental disabilities"
            ],
            "methodologies": [
                "deep learning neural networks autism classification",
                "sample size statistical power developmental disorders",
                "replication reproducibility neuroimaging studies"
            ]
        }

        # Evidence synthesis framework
        self.evidence_levels = {
            "strong": [],
            "moderate": [],
            "weak": [],
            "conflicting": []
        }

        # Track findings
        self.all_findings = defaultdict(list)
        self.paper_metadata = {}

    def initialize(self):
        """Initialize database connection and models"""
        print("Initializing DD Literature Analyzer...")
        print(f"Database path: {self.db_path}")

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {self.db_path}. "
                "Please run 'poetry run python scripts/load_json_to_chromadb_dd.py' first."
            )

        # Connect to ChromaDB
        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_collection(name="dd_papers_L0")

        # Load models
        print("Loading SciBERT embedding model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')

        print("Loading Cross-Encoder for re-ranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print("Initialization complete!\n")

    def query_with_reranking(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query with cross-encoder re-ranking for highest relevance"""

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Retrieve top 50 candidates
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=50
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        if not documents:
            return []

        # Re-rank with cross-encoder
        pairs = [[query, doc] for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        # Sort by relevance score
        scored_results = []
        for i, score in enumerate(scores):
            scored_results.append({
                'document': documents[i],
                'metadata': metadatas[i],
                'score': float(score),
                'query': query
            })

        scored_results.sort(key=lambda x: x['score'], reverse=True)

        return scored_results[:n_results]

    def extract_statistical_info(self, text: str) -> Dict[str, Any]:
        """Extract statistical information from text"""
        stats = {
            'sample_sizes': [],
            'effect_sizes': [],
            'accuracy_metrics': [],
            'p_values': [],
            'confidence_intervals': []
        }

        # Sample size patterns (n=X, N=X, X participants, X subjects)
        n_patterns = [
            r'[nN]\s*=\s*(\d+)',
            r'(\d+)\s+(?:participants|subjects|patients|children|individuals)',
            r'sample\s+(?:of|size)\s+(\d+)'
        ]

        for pattern in n_patterns:
            matches = re.findall(pattern, text)
            stats['sample_sizes'].extend([int(m) for m in matches])

        # Effect sizes (Cohen's d, r, eta-squared)
        effect_patterns = [
            r"(?:Cohen'?s?\s+)?d\s*=\s*([\d.]+)",
            r"r\s*=\s*(0\.\d+)",
            r"η²\s*=\s*(0\.\d+)",
            r"effect\s+size[:\s]+([\d.]+)"
        ]

        for pattern in effect_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['effect_sizes'].extend([float(m) for m in matches])

        # Accuracy/performance metrics
        accuracy_patterns = [
            r'accuracy[:\s]+([\d.]+)%?',
            r'AUC[:\s]+(0\.\d+)',
            r'sensitivity[:\s]+([\d.]+)%?',
            r'specificity[:\s]+([\d.]+)%?',
            r'F1[:\s]+(0\.\d+)'
        ]

        for pattern in accuracy_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['accuracy_metrics'].extend([float(m) if '.' in m else float(m)/100 for m in matches])

        # P-values
        p_patterns = [
            r'p\s*[<>=]\s*(0\.\d+)',
            r'p\s*=\s*\.(\d+)',
        ]

        for pattern in p_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if pattern.endswith(r'(\d+)'):  # Handle .001 format
                stats['p_values'].extend([float(f'0.{m}') for m in matches])
            else:
                stats['p_values'].extend([float(m) for m in matches])

        return stats

    def categorize_evidence_strength(self, results: List[Dict]) -> str:
        """Categorize evidence strength based on convergence and statistics"""

        if len(results) < 2:
            return "weak"

        # Extract statistics from all results
        all_stats = []
        for res in results:
            stats = self.extract_statistical_info(res['document'])
            all_stats.append(stats)

        # Check for convergent findings
        avg_score = sum(r['score'] for r in results) / len(results)

        # Count papers with substantial sample sizes (n > 50)
        large_samples = sum(1 for stats in all_stats
                          if stats['sample_sizes'] and max(stats['sample_sizes']) > 50)

        # Count papers with significant results (p < 0.05)
        significant_results = sum(1 for stats in all_stats
                                 if stats['p_values'] and min(stats['p_values']) < 0.05)

        # Categorize
        if avg_score > 0.7 and large_samples >= 3 and significant_results >= 2:
            return "strong"
        elif avg_score > 0.5 and (large_samples >= 2 or significant_results >= 1):
            return "moderate"
        else:
            return "weak"

    def identify_methodological_innovations(self, results: List[Dict]) -> List[str]:
        """Identify novel methodological approaches"""
        innovations = []

        innovation_keywords = [
            'novel', 'new approach', 'first study', 'innovative',
            'machine learning', 'deep learning', 'convolutional neural',
            'transformer', 'attention mechanism',
            'multimodal', 'multi-modal', 'integration',
            'longitudinal', 'prospective',
            'real-time', 'digital phenotyping',
            'explainable AI', 'interpretable',
            'transfer learning', 'pre-trained'
        ]

        for res in results:
            text_lower = res['document'].lower()
            for keyword in innovation_keywords:
                if keyword.lower() in text_lower:
                    # Extract context around keyword
                    idx = text_lower.find(keyword.lower())
                    context = res['document'][max(0, idx-100):min(len(res['document']), idx+200)]
                    innovations.append({
                        'keyword': keyword,
                        'context': context.strip(),
                        'paper': res['metadata'].get('paper_title', 'Unknown'),
                        'score': res['score']
                    })

        # Deduplicate and sort by relevance
        unique_innovations = {}
        for innov in innovations:
            key = (innov['keyword'], innov['paper'])
            if key not in unique_innovations or unique_innovations[key]['score'] < innov['score']:
                unique_innovations[key] = innov

        return sorted(unique_innovations.values(), key=lambda x: x['score'], reverse=True)

    def identify_research_gaps(self, results: List[Dict]) -> List[str]:
        """Identify explicitly stated research gaps and limitations"""
        gaps = []

        gap_indicators = [
            'limitation', 'future research', 'future work', 'future studies',
            'need for', 'lack of', 'absence of', 'insufficient',
            'challenge', 'barrier', 'difficulty',
            'not yet', 'remain unclear', 'unknown',
            'further investigation', 'warrant', 'require'
        ]

        for res in results:
            text = res['document']
            text_lower = text.lower()

            for indicator in gap_indicators:
                if indicator in text_lower:
                    # Extract sentence containing gap indicator
                    sentences = re.split(r'[.!?]+', text)
                    for sent in sentences:
                        if indicator in sent.lower() and len(sent) > 20:
                            gaps.append({
                                'gap': sent.strip(),
                                'indicator': indicator,
                                'paper': res['metadata'].get('paper_title', 'Unknown'),
                                'section': res['metadata'].get('section', 'Unknown')
                            })

        return gaps

    def analyze_theme(self, theme_name: str, queries: List[str]) -> Dict[str, Any]:
        """Comprehensive analysis of a research theme"""

        print(f"\n{'='*80}")
        print(f"ANALYZING THEME: {theme_name.upper().replace('_', ' ')}")
        print(f"{'='*80}\n")

        theme_results = {
            'theme': theme_name,
            'queries': {},
            'convergent_findings': [],
            'divergent_findings': [],
            'methodological_innovations': [],
            'research_gaps': [],
            'evidence_strength': None,
            'key_papers': set()
        }

        all_theme_results = []

        for query in queries:
            print(f"\nQuery: {query}")
            print("-" * 80)

            # Get top 5 results
            results = self.query_with_reranking(query, n_results=5)

            if not results:
                print("No results found.")
                continue

            all_theme_results.extend(results)

            # Extract key findings
            findings = []
            for i, res in enumerate(results):
                paper_title = res['metadata'].get('paper_title', 'Unknown')
                section = res['metadata'].get('section', 'Unknown')
                score = res['score']

                print(f"\n[{i+1}] {paper_title}")
                print(f"    Section: {section} | Relevance: {score:.3f}")

                # Extract statistics
                stats = self.extract_statistical_info(res['document'])

                finding = {
                    'paper': paper_title,
                    'section': section,
                    'score': score,
                    'content': res['document'][:500],  # First 500 chars
                    'statistics': stats
                }

                findings.append(finding)
                theme_results['key_papers'].add(paper_title)

                # Print key statistics if found
                if stats['sample_sizes']:
                    print(f"    Sample sizes: {stats['sample_sizes'][:3]}")
                if stats['accuracy_metrics']:
                    print(f"    Accuracy metrics: {[f'{x:.3f}' for x in stats['accuracy_metrics'][:3]]}")
                if stats['p_values']:
                    print(f"    P-values: {[f'{x:.4f}' for x in stats['p_values'][:3]]}")

            theme_results['queries'][query] = findings

            # Identify innovations
            innovations = self.identify_methodological_innovations(results)
            theme_results['methodological_innovations'].extend(innovations[:5])

            # Identify gaps
            gaps = self.identify_research_gaps(results)
            theme_results['research_gaps'].extend(gaps[:5])

        # Categorize evidence strength across all queries in theme
        if all_theme_results:
            theme_results['evidence_strength'] = self.categorize_evidence_strength(all_theme_results)

        # Convert set to list for JSON serialization
        theme_results['key_papers'] = list(theme_results['key_papers'])

        print(f"\n{'='*80}")
        print(f"THEME SUMMARY: {theme_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        print(f"Evidence Strength: {theme_results['evidence_strength'].upper()}")
        print(f"Key Papers: {len(theme_results['key_papers'])}")
        print(f"Methodological Innovations: {len(theme_results['methodological_innovations'])}")
        print(f"Research Gaps Identified: {len(theme_results['research_gaps'])}")

        return theme_results

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive analysis across all themes"""

        print("\n" + "="*80)
        print("DD-RAPTOR COMPREHENSIVE LITERATURE ANALYSIS")
        print("Analyzing 26 Developmental Disorder Papers")
        print("="*80 + "\n")

        analysis_report = {
            'metadata': {
                'total_papers': 26,
                'database_path': str(self.db_path),
                'embedding_model': 'allenai/scibert_scivocab_uncased',
                'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            },
            'themes': {},
            'overall_findings': {
                'state_of_the_art': [],
                'critical_gaps': [],
                'methodological_limitations': [],
                'future_directions': []
            }
        }

        # Analyze each theme
        for theme_name, queries in self.query_themes.items():
            theme_analysis = self.analyze_theme(theme_name, queries)
            analysis_report['themes'][theme_name] = theme_analysis

        # Synthesize overall findings
        self._synthesize_overall_findings(analysis_report)

        return analysis_report

    def _synthesize_overall_findings(self, report: Dict[str, Any]):
        """Synthesize findings across all themes"""

        print("\n" + "="*80)
        print("CROSS-THEME SYNTHESIS")
        print("="*80 + "\n")

        # Aggregate innovations (state-of-the-art)
        all_innovations = []
        for theme in report['themes'].values():
            all_innovations.extend(theme['methodological_innovations'])

        # Sort by relevance and deduplicate
        unique_innovations = {}
        for innov in all_innovations:
            key = innov['keyword']
            if key not in unique_innovations or unique_innovations[key]['score'] < innov['score']:
                unique_innovations[key] = innov

        top_innovations = sorted(unique_innovations.values(), key=lambda x: x['score'], reverse=True)[:10]
        report['overall_findings']['state_of_the_art'] = top_innovations

        print(f"Top 10 State-of-the-Art Methods:")
        for i, innov in enumerate(top_innovations):
            print(f"  {i+1}. {innov['keyword']} (relevance: {innov['score']:.3f})")
            print(f"     Paper: {innov['paper'][:80]}")

        # Aggregate research gaps
        all_gaps = []
        for theme in report['themes'].values():
            all_gaps.extend(theme['research_gaps'])

        # Deduplicate gaps
        unique_gaps = {}
        for gap in all_gaps:
            key = gap['gap'][:100]  # Use first 100 chars as key
            if key not in unique_gaps:
                unique_gaps[key] = gap

        top_gaps = list(unique_gaps.values())[:15]
        report['overall_findings']['critical_gaps'] = top_gaps

        print(f"\n\nTop 15 Critical Research Gaps:")
        for i, gap in enumerate(top_gaps):
            print(f"  {i+1}. {gap['gap'][:150]}...")
            print(f"     [{gap['indicator']}] - {gap['paper'][:60]}")

        # Evidence strength summary
        print(f"\n\nEvidence Strength by Theme:")
        for theme_name, theme_data in report['themes'].items():
            strength = theme_data.get('evidence_strength', 'unknown')
            print(f"  {theme_name.replace('_', ' ').title()}: {strength.upper()}")

    def save_report(self, report: Dict[str, Any], output_path: str = "dd_literature_analysis_report.json"):
        """Save analysis report to JSON file"""

        output_file = Path(output_path)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n\n{'='*80}")
        print(f"Analysis report saved to: {output_file.absolute()}")
        print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
        print(f"{'='*80}\n")


def main():
    """Main execution function"""

    try:
        # Initialize analyzer
        analyzer = DDLiteratureAnalyzer(db_path="chromadb_data_dd")
        analyzer.initialize()

        # Run comprehensive analysis
        report = analyzer.run_comprehensive_analysis()

        # Save report
        analyzer.save_report(report)

        print("\n✓ Analysis complete!")

        return 0

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
