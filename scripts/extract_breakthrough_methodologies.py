#!/usr/bin/env python3
"""
Breakthrough Methodology Extraction from DD-RAPTOR ChromaDB
Systematic query system to extract cutting-edge research approaches
for enhancing proposal success rate.
"""

import chromadb
import json
import sys
from collections import defaultdict
from typing import List, Dict, Any
import re

class BreakthroughMethodologyExtractor:
    """Extract breakthrough research methodologies from DD literature database"""

    def __init__(self, db_path: str = '/home/juke/git/AI-CoScientist/chromadb_data_dd'):
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.results = defaultdict(list)

    def initialize(self):
        """Initialize ChromaDB connection"""
        print('[INIT] Connecting to ChromaDB...', flush=True)
        self.client = chromadb.PersistentClient(path=self.db_path)

        collections = self.client.list_collections()
        if not collections:
            raise RuntimeError('No collections found in ChromaDB!')

        col_name = collections[0].name
        print(f'[INIT] Using collection: {col_name}', flush=True)
        self.collection = self.client.get_collection(name=col_name)

        # Get collection stats
        count = self.collection.count()
        print(f'[INIT] Collection contains {count} documents', flush=True)

    def execute_strategic_query(self, category: str, query_text: str, n_results: int = 10) -> List[Dict]:
        """Execute a single strategic query and return results"""
        print(f'\n[QUERY] Category: {category}', flush=True)
        print(f'[QUERY] Search: "{query_text}"', flush=True)

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            if not results['documents'] or not results['documents'][0]:
                print(f'[WARN] No results found for query: {query_text}', flush=True)
                return []

            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            query_results = []
            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                # Extract key information
                result = {
                    'rank': i + 1,
                    'relevance_score': 1.0 - dist,  # Convert distance to similarity
                    'paper_title': meta.get('paper_title', 'Unknown'),
                    'authors': meta.get('authors', 'Unknown'),
                    'year': meta.get('year', 'Unknown'),
                    'document_snippet': doc[:500],  # First 500 chars
                    'full_document': doc,
                    'metadata': meta
                }
                query_results.append(result)

            print(f'[RESULT] Found {len(query_results)} results (top relevance: {query_results[0]["relevance_score"]:.3f})', flush=True)
            return query_results

        except Exception as e:
            print(f'[ERROR] Query failed: {e}', flush=True)
            return []

    def extract_methodology_details(self, doc: str, category: str) -> Dict[str, Any]:
        """Extract specific methodology details from document"""
        details = {
            'methods': [],
            'metrics': [],
            'innovations': [],
            'technical_specs': []
        }

        # Extract performance metrics (percentages, accuracies, etc.)
        metric_patterns = [
            r'(\d+\.?\d*)\s*%\s*(accuracy|precision|recall|F1|AUC|sensitivity|specificity)',
            r'(accuracy|precision|recall|F1|AUC)\s*of\s*(\d+\.?\d*)\s*%',
            r'improved?\s+by\s+(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*%\s*improvement'
        ]

        for pattern in metric_patterns:
            matches = re.finditer(pattern, doc, re.IGNORECASE)
            for match in matches:
                details['metrics'].append(match.group(0))

        # Extract method keywords based on category
        if category == 'Novel Diagnostic Methods':
            keywords = ['biomarker', 'phenotyping', 'MRI', 'EEG', 'fMRI', 'dMRI',
                       'connectome', 'multimodal', 'digital', 'wearable']
        elif category == 'Advanced AI/ML':
            keywords = ['transformer', 'BERT', 'GPT', 'attention', 'foundation model',
                       'neural network', 'CNN', 'RNN', 'LSTM', 'GNN', 'GCN',
                       'self-supervised', 'contrastive', 'few-shot', 'zero-shot']
        elif category == 'Data Integration':
            keywords = ['fusion', 'integration', 'multimodal', 'genomic', 'neuroimaging',
                       'behavioral', 'phenotype', 'genotype', 'cross-modal']
        elif category == 'Treatment Personalization':
            keywords = ['reinforcement learning', 'adaptive', 'personalized', 'precision',
                       'intervention', 'treatment', 'therapy', 'RL', 'DRL']
        elif category == 'Methodological Innovation':
            keywords = ['causal inference', 'counterfactual', 'RCT', 'longitudinal',
                       'cross-sectional', 'meta-analysis', 'systematic review']
        else:
            keywords = []

        # Find keyword mentions with context
        for keyword in keywords:
            pattern = r'.{0,100}' + re.escape(keyword) + r'.{0,100}'
            matches = re.finditer(pattern, doc, re.IGNORECASE)
            for match in matches:
                context = match.group(0).strip()
                if context not in details['methods']:
                    details['methods'].append(context)
                    if len(details['methods']) >= 5:  # Limit to top 5
                        break

        return details

    def run_comprehensive_extraction(self):
        """Execute all strategic queries and compile results"""

        # Define strategic queries by category
        query_plan = {
            'Novel Diagnostic Methods': [
                'foundation model autism diagnosis biomarkers',
                'multimodal biomarkers developmental disorders early detection',
                'digital phenotyping autism spectrum disorder',
                'wearable sensors neurodevelopmental assessment',
                'neuroimaging biomarkers MRI EEG fMRI precision diagnosis'
            ],
            'Advanced AI/ML': [
                'transformer neural network autism classification',
                'foundation model developmental disability diagnosis',
                'self-supervised learning brain imaging analysis',
                'graph neural network brain connectome autism',
                'attention mechanism multimodal fusion neurodevelopmental'
            ],
            'Data Integration': [
                'multimodal fusion genomics neuroimaging behavioral data',
                'cross-modal integration developmental disorders',
                'genomic neuroimaging integration autism',
                'phenotype genotype integration neurodevelopmental',
                'multi-scale data fusion brain disorders'
            ],
            'Treatment Personalization': [
                'reinforcement learning personalized treatment autism',
                'adaptive intervention developmental disorders',
                'precision medicine neurodevelopmental therapy',
                'deep reinforcement learning behavioral intervention',
                'personalized treatment optimization machine learning'
            ],
            'Methodological Innovation': [
                'causal inference neurodevelopmental disorders',
                'federated learning brain imaging privacy',
                'counterfactual analysis autism intervention',
                'longitudinal analysis developmental trajectories',
                'transfer learning small sample developmental disorders'
            ]
        }

        print('\n' + '='*80)
        print('BREAKTHROUGH METHODOLOGY EXTRACTION - COMPREHENSIVE ANALYSIS')
        print('='*80)

        all_results = {}

        for category, queries in query_plan.items():
            print(f'\n\n{"="*80}')
            print(f'CATEGORY: {category}')
            print(f'{"="*80}')

            category_results = []

            for query in queries:
                results = self.execute_strategic_query(category, query, n_results=5)

                for result in results:
                    # Extract methodology details
                    method_details = self.extract_methodology_details(
                        result['full_document'],
                        category
                    )
                    result['extracted_details'] = method_details
                    category_results.append(result)

            # Deduplicate by paper title
            unique_papers = {}
            for result in category_results:
                title = result['paper_title']
                if title not in unique_papers or result['relevance_score'] > unique_papers[title]['relevance_score']:
                    unique_papers[title] = result

            all_results[category] = list(unique_papers.values())

            # Print summary
            print(f'\n[SUMMARY] Found {len(unique_papers)} unique papers for {category}')
            for i, result in enumerate(sorted(unique_papers.values(),
                                             key=lambda x: x['relevance_score'],
                                             reverse=True)[:3]):
                print(f'  Top {i+1}: {result["paper_title"]} (relevance: {result["relevance_score"]:.3f})')

        return all_results

    def generate_proposal_recommendations(self, all_results: Dict) -> Dict:
        """Generate actionable recommendations for proposal enhancement"""

        recommendations = {
            'executive_summary': {},
            'breakthrough_methods': {},
            'integration_strategies': {},
            'performance_metrics': {}
        }

        for category, results in all_results.items():
            # Top 3 papers per category
            top_papers = sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:3]

            methods_found = []
            metrics_found = []

            for paper in top_papers:
                details = paper.get('extracted_details', {})
                methods_found.extend(details.get('methods', []))
                metrics_found.extend(details.get('metrics', []))

            recommendations['breakthrough_methods'][category] = {
                'top_papers': [
                    {
                        'title': p['paper_title'],
                        'authors': p['authors'],
                        'year': p['year'],
                        'relevance': p['relevance_score']
                    } for p in top_papers
                ],
                'key_methods': methods_found[:10],  # Top 10 methods
                'performance_metrics': list(set(metrics_found))[:5]  # Top 5 unique metrics
            }

        return recommendations

    def save_results(self, all_results: Dict, recommendations: Dict, output_file: str):
        """Save extracted results and recommendations to JSON"""

        output = {
            'metadata': {
                'database_path': self.db_path,
                'total_categories': len(all_results),
                'total_unique_papers': sum(len(results) for results in all_results.values())
            },
            'raw_results': all_results,
            'recommendations': recommendations
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f'\n[SAVE] Results saved to: {output_file}', flush=True)

        # Also create a human-readable markdown report
        md_file = output_file.replace('.json', '.md')
        self.generate_markdown_report(recommendations, md_file)
        print(f'[SAVE] Markdown report saved to: {md_file}', flush=True)

    def generate_markdown_report(self, recommendations: Dict, output_file: str):
        """Generate human-readable markdown report"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('# Breakthrough Methodology Extraction Report\n\n')
            f.write('## Executive Summary\n\n')
            f.write('This report identifies cutting-edge research methodologies from 26 developmental disability papers ')
            f.write('that can enhance proposal success rate for QuantERA 2025 and Samsung grants.\n\n')

            for category, data in recommendations.get('breakthrough_methods', {}).items():
                f.write(f'\n## {category}\n\n')

                f.write('### Top Papers\n\n')
                for i, paper in enumerate(data['top_papers'], 1):
                    f.write(f'{i}. **{paper["title"]}**\n')
                    f.write(f'   - Authors: {paper["authors"]}\n')
                    f.write(f'   - Year: {paper["year"]}\n')
                    f.write(f'   - Relevance Score: {paper["relevance"]:.3f}\n\n')

                if data['key_methods']:
                    f.write('### Key Methodologies Identified\n\n')
                    for i, method in enumerate(data['key_methods'], 1):
                        f.write(f'{i}. {method}\n')
                    f.write('\n')

                if data['performance_metrics']:
                    f.write('### Performance Metrics\n\n')
                    for metric in data['performance_metrics']:
                        f.write(f'- {metric}\n')
                    f.write('\n')


def main():
    """Main execution function"""

    extractor = BreakthroughMethodologyExtractor()

    try:
        # Initialize
        extractor.initialize()

        # Run comprehensive extraction
        all_results = extractor.run_comprehensive_extraction()

        # Generate recommendations
        print('\n\n' + '='*80)
        print('GENERATING PROPOSAL RECOMMENDATIONS')
        print('='*80)
        recommendations = extractor.generate_proposal_recommendations(all_results)

        # Save results
        output_file = '/home/juke/git/AI-CoScientist/breakthrough_methodologies_2025.json'
        extractor.save_results(all_results, recommendations, output_file)

        print('\n' + '='*80)
        print('EXTRACTION COMPLETE')
        print('='*80)
        print(f'\nTotal unique papers analyzed: {sum(len(r) for r in all_results.values())}')
        print(f'Categories covered: {len(all_results)}')

        return 0

    except Exception as e:
        print(f'\n[FATAL ERROR] {e}', flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
