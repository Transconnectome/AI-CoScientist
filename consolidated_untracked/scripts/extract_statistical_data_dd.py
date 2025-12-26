#!/usr/bin/env python3
"""
Extract Statistical Data from DD-RAPTOR ChromaDB for Meta-Analysis
=====================================================================

This script systematically queries the DD-RAPTOR knowledge base to extract
quantitative data for statistical meta-analysis, including:
- Sample sizes
- Effect sizes (Cohen's d, odds ratios, etc.)
- Confidence intervals
- Diagnostic accuracy metrics (sensitivity, specificity, AUC)
- Performance metrics for ML models

Usage:
    python scripts/extract_statistical_data_dd.py --output statistical_data_extraction.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import numpy as np

# Statistical data patterns
PATTERNS = {
    'sample_size': [
        r'n\s*=\s*(\d+)',
        r'N\s*=\s*(\d+)',
        r'sample\s+size\s+of\s+(\d+)',
        r'(\d+)\s+participants',
        r'(\d+)\s+patients',
        r'(\d+)\s+subjects',
        r'cohort\s+of\s+(\d+)',
    ],
    'sensitivity': [
        r'sensitivity\s*[:=]\s*([0-9.]+)%?',
        r'sens\.\s*[:=]\s*([0-9.]+)%?',
        r'true\s+positive\s+rate\s*[:=]\s*([0-9.]+)%?',
    ],
    'specificity': [
        r'specificity\s*[:=]\s*([0-9.]+)%?',
        r'spec\.\s*[:=]\s*([0-9.]+)%?',
        r'true\s+negative\s+rate\s*[:=]\s*([0-9.]+)%?',
    ],
    'accuracy': [
        r'accuracy\s*[:=]\s*([0-9.]+)%?',
        r'acc\.\s*[:=]\s*([0-9.]+)%?',
        r'correct\s+classification\s*[:=]\s*([0-9.]+)%?',
    ],
    'auc': [
        r'AUC\s*[:=]\s*([0-9.]+)',
        r'area\s+under\s+the\s+curve\s*[:=]\s*([0-9.]+)',
        r'AUROC\s*[:=]\s*([0-9.]+)',
    ],
    'confidence_interval': [
        r'95%?\s*CI\s*[:=]?\s*\[?([0-9.]+)\s*[-–to]\s*([0-9.]+)\]?',
        r'\(([0-9.]+)\s*[-–to]\s*([0-9.]+)\)',
        r'CI\s*[:=]?\s*([0-9.]+)\s*[-–]\s*([0-9.]+)',
    ],
    'p_value': [
        r'p\s*[<=]\s*([0-9.]+)',
        r'p-value\s*[:=]\s*([0-9.]+)',
    ],
    'effect_size': [
        r"Cohen's\s+d\s*[:=]\s*([0-9.]+)",
        r'd\s*=\s*([0-9.]+)',
        r'effect\s+size\s*[:=]\s*([0-9.]+)',
        r'η²\s*[:=]\s*([0-9.]+)',
        r'eta\s+squared\s*[:=]\s*([0-9.]+)',
    ],
    'odds_ratio': [
        r'OR\s*[:=]\s*([0-9.]+)',
        r'odds\s+ratio\s*[:=]\s*([0-9.]+)',
    ],
}


class StatisticalDataExtractor:
    """Extract statistical data from DD-RAPTOR knowledge base."""

    def __init__(self, db_path: str = "chromadb_data_dd"):
        """Initialize extractor with ChromaDB connection."""
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="dd_papers_L0")

        # Load models
        print("Loading SciBERT embedding model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')

        print("Loading Cross-Encoder for re-ranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Results storage
        self.extracted_data = {
            'queries': [],
            'papers_analyzed': set(),
            'statistical_findings': [],
            'summary_statistics': {},
        }

    def query_and_extract(self, query: str, n_results: int = 50, category: str = "general") -> Dict:
        """Query DD-RAPTOR and extract statistical data."""
        print(f"\n🔍 Querying: '{query}'")

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Retrieve candidates
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        if not documents:
            print(f"  ⚠️  No results found for query: {query}")
            return {'query': query, 'findings': []}

        # Re-rank with cross-encoder
        pairs = [[query, doc] for doc in documents]
        scores = self.cross_encoder.predict(pairs)

        # Sort by relevance
        scored_results = []
        for i, score in enumerate(scores):
            scored_results.append({
                'document': documents[i],
                'metadata': metadatas[i],
                'score': float(score)
            })

        scored_results.sort(key=lambda x: x['score'], reverse=True)

        # Extract statistical data from top results
        findings = []
        for result in scored_results[:20]:  # Top 20 most relevant
            extracted = self.extract_statistical_info(
                result['document'],
                result['metadata'],
                category
            )
            if extracted:
                extracted['relevance_score'] = result['score']
                findings.append(extracted)

                # Track papers analyzed
                paper_id = result['metadata'].get('paper_id', 'unknown')
                self.extracted_data['papers_analyzed'].add(paper_id)

        query_result = {
            'query': query,
            'category': category,
            'n_documents_retrieved': len(documents),
            'n_documents_reranked': len(scored_results),
            'n_findings_extracted': len(findings),
            'findings': findings
        }

        self.extracted_data['queries'].append(query_result)
        self.extracted_data['statistical_findings'].extend(findings)

        print(f"  ✅ Extracted {len(findings)} statistical findings")

        return query_result

    def extract_statistical_info(self, document: str, metadata: Dict, category: str) -> Optional[Dict]:
        """Extract statistical information from a document."""
        extracted = {
            'paper_title': metadata.get('paper_title', 'Unknown'),
            'section': metadata.get('section', 'Unknown'),
            'category': category,
            'statistics': {},
            'raw_text': document[:500]  # Store snippet for verification
        }

        found_any = False

        # Extract each type of statistical data
        for stat_type, patterns in PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, document, re.IGNORECASE)
                if matches:
                    found_any = True
                    if stat_type == 'confidence_interval':
                        # Special handling for CIs (tuple)
                        extracted['statistics'][stat_type] = [
                            {'lower': float(m[0]), 'upper': float(m[1])}
                            for m in matches if len(m) == 2
                        ]
                    else:
                        # Single value
                        extracted['statistics'][stat_type] = [float(m) if isinstance(m, str) else float(m[0])
                                                               for m in matches]
                    break  # Found match, move to next stat type

        return extracted if found_any else None

    def run_comprehensive_extraction(self):
        """Run comprehensive statistical data extraction across multiple query categories."""

        print("\n" + "=" * 70)
        print("COMPREHENSIVE STATISTICAL DATA EXTRACTION FROM DD-RAPTOR")
        print("=" * 70)

        # Define systematic queries for different statistical domains
        query_categories = {
            'diagnostic_accuracy': [
                "diagnostic accuracy sensitivity specificity AUC autism",
                "machine learning classification performance autism ADHD",
                "ROC curve diagnostic performance developmental disorders",
                "predictive accuracy early detection autism",
            ],
            'sample_sizes': [
                "sample size participants cohort study autism",
                "number of subjects neuroimaging study developmental disorders",
                "patient cohort longitudinal study ADHD",
            ],
            'effect_sizes': [
                "effect size Cohen's d statistical power autism",
                "treatment effect intervention outcome developmental disorders",
                "odds ratio risk factor genetic association autism",
            ],
            'biomarker_performance': [
                "biomarker accuracy prediction early diagnosis autism",
                "genetic biomarker sensitivity specificity developmental disorders",
                "neuroimaging biomarker diagnostic performance ASD",
                "EEG biomarker classification accuracy ADHD",
            ],
            'multimodal_fusion': [
                "multimodal accuracy performance imaging genomics autism",
                "fusion approach classification accuracy developmental disorders",
                "combined features diagnostic performance autism ADHD",
            ],
            'longitudinal_studies': [
                "longitudinal study sample size follow-up autism",
                "prospective cohort developmental trajectory ADHD",
                "attrition rate retention longitudinal study developmental disorders",
            ],
            'meta_analysis': [
                "meta-analysis pooled estimate sensitivity specificity autism",
                "systematic review combined accuracy diagnostic ASD",
                "meta-analytic results effect size developmental disorders",
            ],
        }

        # Execute queries for each category
        for category, queries in query_categories.items():
            print(f"\n{'='*70}")
            print(f"CATEGORY: {category.upper()}")
            print(f"{'='*70}")

            for query in queries:
                self.query_and_extract(query, n_results=50, category=category)

        # Calculate summary statistics
        self.calculate_summary_statistics()

        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"Papers analyzed: {len(self.extracted_data['papers_analyzed'])}")
        print(f"Total queries: {len(self.extracted_data['queries'])}")
        print(f"Statistical findings: {len(self.extracted_data['statistical_findings'])}")
        print("=" * 70)

    def calculate_summary_statistics(self):
        """Calculate summary statistics from extracted data."""

        summary = {
            'sample_sizes': [],
            'sensitivities': [],
            'specificities': [],
            'accuracies': [],
            'aucs': [],
            'effect_sizes': [],
            'p_values': [],
        }

        # Aggregate all findings
        for finding in self.extracted_data['statistical_findings']:
            stats = finding['statistics']

            if 'sample_size' in stats:
                summary['sample_sizes'].extend(stats['sample_size'])

            if 'sensitivity' in stats:
                # Convert percentages to proportions if needed
                values = [v/100 if v > 1 else v for v in stats['sensitivity']]
                summary['sensitivities'].extend(values)

            if 'specificity' in stats:
                values = [v/100 if v > 1 else v for v in stats['specificity']]
                summary['specificities'].extend(values)

            if 'accuracy' in stats:
                values = [v/100 if v > 1 else v for v in stats['accuracy']]
                summary['accuracies'].extend(values)

            if 'auc' in stats:
                summary['aucs'].extend(stats['auc'])

            if 'effect_size' in stats:
                summary['effect_sizes'].extend(stats['effect_size'])

            if 'p_value' in stats:
                summary['p_values'].extend(stats['p_value'])

        # Calculate descriptive statistics
        descriptive_stats = {}

        for key, values in summary.items():
            if values:
                descriptive_stats[key] = {
                    'n': len(values),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'quartiles': {
                        'q25': float(np.percentile(values, 25)),
                        'q50': float(np.percentile(values, 50)),
                        'q75': float(np.percentile(values, 75)),
                    }
                }

        self.extracted_data['summary_statistics'] = descriptive_stats

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)

        for stat_name, stats in descriptive_stats.items():
            print(f"\n{stat_name.upper().replace('_', ' ')}:")
            print(f"  n = {stats['n']}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  SD: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  IQR: [{stats['quartiles']['q25']:.4f}, {stats['quartiles']['q75']:.4f}]")

    def save_results(self, output_file: str):
        """Save extracted data to JSON file."""

        # Convert set to list for JSON serialization
        output_data = self.extracted_data.copy()
        output_data['papers_analyzed'] = list(output_data['papers_analyzed'])

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Extract statistical data from DD-RAPTOR ChromaDB"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='statistical_data_extraction.json',
        help='Output JSON file for extracted data'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='chromadb_data_dd',
        help='Path to ChromaDB directory'
    )

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"❌ Error: ChromaDB not found at {args.db_path}")
        print("Please run 'python scripts/load_json_to_chromadb_dd.py' first.")
        sys.exit(1)

    # Run extraction
    extractor = StatisticalDataExtractor(db_path=args.db_path)
    extractor.run_comprehensive_extraction()
    extractor.save_results(args.output)

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review extracted data in:", args.output)
    print("2. Use data for meta-analysis and statistical power calculations")
    print("3. Generate publication-ready tables and forest plots")
    print("=" * 70)


if __name__ == "__main__":
    main()
