#!/usr/bin/env python3
"""
Breakthrough Methodology Extraction from DD-RAPTOR ChromaDB (v2)
Uses direct document retrieval without re-embedding
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
        self.collection_l0 = None
        self.collection_l1 = None
        self.collection_l2 = None

    def initialize(self):
        """Initialize ChromaDB connection"""
        print('[INIT] Connecting to ChromaDB...', flush=True)
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Get all three collections
        self.collection_l0 = self.client.get_collection(name="dd_papers_L0")
        self.collection_l1 = self.client.get_collection(name="dd_papers_L1")
        self.collection_l2 = self.client.get_collection(name="dd_papers_L2")

        print(f'[INIT] L0 chunks: {self.collection_l0.count()}', flush=True)
        print(f'[INIT] L1 summaries: {self.collection_l1.count()}', flush=True)
        print(f'[INIT] L2 papers: {self.collection_l2.count()}', flush=True)

    def retrieve_all_documents(self) -> Dict[str, List[Dict]]:
        """Retrieve all documents from ChromaDB"""
        print('\n[RETRIEVE] Fetching all documents...', flush=True)

        all_docs = {
            'L0': [],
            'L1': [],
            'L2': []
        }

        # Get L0 chunks
        l0_results = self.collection_l0.get(limit=10000, include=['documents', 'metadatas'])
        for i, (doc, meta) in enumerate(zip(l0_results['documents'], l0_results['metadatas'])):
            all_docs['L0'].append({
                'id': l0_results['ids'][i],
                'content': doc,
                'metadata': meta
            })

        # Get L1 summaries
        l1_results = self.collection_l1.get(limit=10000, include=['documents', 'metadatas'])
        for i, (doc, meta) in enumerate(zip(l1_results['documents'], l1_results['metadatas'])):
            all_docs['L1'].append({
                'id': l1_results['ids'][i],
                'content': doc,
                'metadata': meta
            })

        # Get L2 papers
        l2_results = self.collection_l2.get(limit=10000, include=['documents', 'metadatas'])
        for i, (doc, meta) in enumerate(zip(l2_results['documents'], l2_results['metadatas'])):
            all_docs['L2'].append({
                'id': l2_results['ids'][i],
                'content': doc,
                'metadata': meta
            })

        print(f'[RETRIEVE] Retrieved {len(all_docs["L0"])} L0, {len(all_docs["L1"])} L1, {len(all_docs["L2"])} L2 docs', flush=True)
        return all_docs

    def search_by_keywords(self, all_docs: Dict, keywords: List[str], level: str = 'L1') -> List[Dict]:
        """Search documents by keywords using simple text matching"""
        results = []

        for doc in all_docs[level]:
            content_lower = doc['content'].lower()
            # Score based on keyword matches
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                results.append({
                    'document': doc['content'],
                    'metadata': doc['metadata'],
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'paper_title': doc['metadata'].get('paper_title', 'Unknown')
                })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def extract_methodology_details(self, doc: str, category: str) -> Dict[str, Any]:
        """Extract specific methodology details from document"""
        details = {
            'methods': [],
            'metrics': [],
            'innovations': [],
            'technical_specs': []
        }

        # Extract performance metrics
        metric_patterns = [
            r'(\d+\.?\d*)\s*%\s*(accuracy|precision|recall|F1|AUC|sensitivity|specificity)',
            r'(accuracy|precision|recall|F1|AUC|sensitivity|specificity)[:\s]+(\d+\.?\d*)\s*%?',
            r'improved?\s+by\s+(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*%\s*improvement',
            r'achieved\s+(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*%\s+(better|higher|superior)'
        ]

        for pattern in metric_patterns:
            matches = re.finditer(pattern, doc, re.IGNORECASE)
            for match in matches:
                metric_text = match.group(0)
                if metric_text not in details['metrics']:
                    details['metrics'].append(metric_text)

        # Extract method descriptions (context around key terms)
        method_keywords = {
            'Novel Diagnostic Methods': [
                'biomarker', 'phenotyping', 'MRI', 'EEG', 'fMRI', 'dMRI', 'DTI',
                'connectome', 'multimodal', 'digital', 'wearable', 'sensor'
            ],
            'Advanced AI/ML': [
                'transformer', 'BERT', 'GPT', 'attention', 'foundation model',
                'neural network', 'CNN', 'RNN', 'LSTM', 'GNN', 'GCN',
                'self-supervised', 'contrastive', 'few-shot', 'zero-shot',
                'deep learning', 'machine learning'
            ],
            'Data Integration': [
                'fusion', 'integration', 'multimodal', 'genomic', 'neuroimaging',
                'behavioral', 'phenotype', 'genotype', 'cross-modal', 'multi-scale',
                'heterogeneous data'
            ],
            'Treatment Personalization': [
                'reinforcement learning', 'adaptive', 'personalized', 'precision',
                'intervention', 'treatment', 'therapy', 'RL', 'DRL', 'individualized',
                'tailored'
            ],
            'Methodological Innovation': [
                'causal inference', 'counterfactual', 'RCT', 'longitudinal',
                'cross-sectional', 'meta-analysis', 'systematic review', 'federated',
                'transfer learning', 'domain adaptation'
            ]
        }

        keywords = method_keywords.get(category, [])

        # Find sentences containing keywords
        sentences = re.split(r'[.!?]\s+', doc)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in keywords:
                if keyword.lower() in sentence_lower and len(sentence) > 20:
                    if sentence not in details['methods'] and len(details['methods']) < 10:
                        details['methods'].append(sentence.strip())
                        break

        return details

    def run_comprehensive_extraction(self):
        """Execute keyword-based extraction across all categories"""

        # Retrieve all documents
        all_docs = self.retrieve_all_documents()

        # Define keyword sets by category
        query_keywords = {
            'Novel Diagnostic Methods': [
                ['foundation model', 'autism', 'diagnosis'],
                ['multimodal', 'biomarker', 'detection'],
                ['digital phenotyping', 'autism'],
                ['wearable', 'sensor', 'neurodevelopmental'],
                ['neuroimaging', 'MRI', 'EEG', 'biomarker'],
                ['fMRI', 'connectome', 'diagnosis'],
                ['early detection', 'screening', 'ASD']
            ],
            'Advanced AI/ML': [
                ['transformer', 'neural network', 'autism'],
                ['foundation model', 'developmental'],
                ['self-supervised', 'learning', 'brain'],
                ['graph neural network', 'connectome'],
                ['attention mechanism', 'multimodal'],
                ['deep learning', 'classification', 'ASD'],
                ['machine learning', 'diagnosis', 'developmental']
            ],
            'Data Integration': [
                ['multimodal', 'fusion', 'neuroimaging'],
                ['cross-modal', 'integration'],
                ['genomic', 'neuroimaging', 'integration'],
                ['phenotype', 'genotype'],
                ['heterogeneous data', 'integration'],
                ['multi-scale', 'fusion']
            ],
            'Treatment Personalization': [
                ['reinforcement learning', 'treatment'],
                ['adaptive intervention', 'autism'],
                ['precision medicine', 'developmental'],
                ['personalized', 'therapy', 'individualized'],
                ['tailored treatment', 'ASD']
            ],
            'Methodological Innovation': [
                ['causal inference', 'neurodevelopmental'],
                ['federated learning', 'privacy'],
                ['counterfactual', 'intervention'],
                ['longitudinal', 'developmental'],
                ['transfer learning', 'small sample'],
                ['meta-analysis', 'autism'],
                ['systematic review', 'developmental disorders']
            ]
        }

        print('\n' + '='*80)
        print('BREAKTHROUGH METHODOLOGY EXTRACTION - COMPREHENSIVE ANALYSIS')
        print('='*80)

        all_results = {}

        for category, keyword_sets in query_keywords.items():
            print(f'\n{"="*80}')
            print(f'CATEGORY: {category}')
            print(f'{"="*80}')

            category_results = []

            for keywords in keyword_sets:
                print(f'\n[SEARCH] Keywords: {", ".join(keywords)}', flush=True)

                # Search in L1 (section summaries) for better quality
                results = self.search_by_keywords(all_docs, keywords, level='L1')

                if results:
                    print(f'[FOUND] {len(results)} documents (top score: {results[0]["score"]})', flush=True)

                    # Take top 5 results
                    for result in results[:5]:
                        # Extract methodology details
                        method_details = self.extract_methodology_details(
                            result['document'],
                            category
                        )
                        result['extracted_details'] = method_details
                        category_results.append(result)

            # Deduplicate by paper title
            unique_papers = {}
            for result in category_results:
                title = result['paper_title']
                if title not in unique_papers or result['score'] > unique_papers[title]['score']:
                    unique_papers[title] = result

            all_results[category] = list(unique_papers.values())

            # Print summary
            print(f'\n[SUMMARY] Found {len(unique_papers)} unique papers for {category}')
            for i, result in enumerate(sorted(unique_papers.values(),
                                             key=lambda x: x['score'],
                                             reverse=True)[:3]):
                print(f'  Top {i+1}: {result["paper_title"]} (score: {result["score"]}, keywords: {", ".join(result["matched_keywords"][:3])})')

        return all_results

    def generate_proposal_recommendations(self, all_results: Dict) -> Dict:
        """Generate actionable recommendations for proposal enhancement"""

        recommendations = {
            'executive_summary': {
                'total_papers_analyzed': len(set(
                    r['paper_title'] for results in all_results.values() for r in results
                )),
                'categories_covered': len(all_results),
                'breakthrough_methods_identified': sum(
                    len(r.get('extracted_details', {}).get('methods', []))
                    for results in all_results.values() for r in results
                ),
                'performance_metrics_found': sum(
                    len(r.get('extracted_details', {}).get('metrics', []))
                    for results in all_results.values() for r in results
                )
            },
            'breakthrough_methods': {},
            'integration_strategies': {},
            'proposal_enhancement_opportunities': []
        }

        for category, results in all_results.items():
            # Top 5 papers per category
            top_papers = sorted(results, key=lambda x: x['score'], reverse=True)[:5]

            methods_found = []
            metrics_found = []

            for paper in top_papers:
                details = paper.get('extracted_details', {})
                methods_found.extend(details.get('methods', []))
                metrics_found.extend(details.get('metrics', []))

            # Deduplicate metrics
            unique_metrics = list(set(metrics_found))

            recommendations['breakthrough_methods'][category] = {
                'top_papers': [
                    {
                        'title': p['paper_title'],
                        'relevance_score': p['score'],
                        'matched_keywords': p.get('matched_keywords', []),
                        'snippet': p['document'][:300]
                    } for p in top_papers
                ],
                'key_methods': methods_found[:15],  # Top 15 methods
                'performance_metrics': unique_metrics[:10],  # Top 10 unique metrics
                'total_relevant_papers': len(results)
            }

        # Generate integration strategies
        recommendations['integration_strategies'] = self._generate_integration_strategies(all_results)

        # Generate proposal enhancement opportunities
        recommendations['proposal_enhancement_opportunities'] = self._generate_enhancement_opportunities(all_results)

        return recommendations

    def _generate_integration_strategies(self, all_results: Dict) -> Dict:
        """Generate strategies for integrating findings into proposal"""
        strategies = {}

        for category, results in all_results.items():
            if not results:
                continue

            top_paper = sorted(results, key=lambda x: x['score'], reverse=True)[0]
            details = top_paper.get('extracted_details', {})

            strategies[category] = {
                'primary_approach': details.get('methods', ['Not identified'])[0] if details.get('methods') else 'Not identified',
                'expected_performance': details.get('metrics', ['Performance metrics not specified'])[0] if details.get('metrics') else 'Performance metrics not specified',
                'integration_with_neurox_fusion': f'Leverage {category.lower()} approaches with INCITE NeuroX-Fusion 130B model',
                'samsung_proposal_angle': f'Emphasize {category.lower()} innovation for developmental disability research'
            }

        return strategies

    def _generate_enhancement_opportunities(self, all_results: Dict) -> List[Dict]:
        """Identify specific opportunities to enhance proposals"""
        opportunities = []

        # Check which categories have strong findings
        for category, results in all_results.items():
            if len(results) >= 3:  # At least 3 relevant papers
                top_result = sorted(results, key=lambda x: x['score'], reverse=True)[0]
                opportunities.append({
                    'category': category,
                    'opportunity': f'Integrate {category.lower()} from {top_result["paper_title"]}',
                    'evidence_strength': 'Strong' if len(results) >= 5 else 'Moderate',
                    'implementation_priority': 'High' if top_result['score'] >= 3 else 'Medium',
                    'matched_keywords': top_result.get('matched_keywords', [])
                })

        return sorted(opportunities, key=lambda x: (
            x['evidence_strength'] == 'Strong',
            x['implementation_priority'] == 'High'
        ), reverse=True)

    def save_results(self, all_results: Dict, recommendations: Dict, output_file: str):
        """Save extracted results and recommendations to JSON"""

        output = {
            'metadata': {
                'database_path': self.db_path,
                'extraction_method': 'Keyword-based search (direct retrieval)',
                'total_categories': len(all_results),
                'total_unique_papers': len(set(
                    r['paper_title'] for results in all_results.values() for r in results
                ))
            },
            'raw_results': all_results,
            'recommendations': recommendations
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f'\n[SAVE] Results saved to: {output_file}', flush=True)

        # Also create a human-readable markdown report
        md_file = output_file.replace('.json', '.md')
        self.generate_markdown_report(all_results, recommendations, md_file)
        print(f'[SAVE] Markdown report saved to: {md_file}', flush=True)

    def generate_markdown_report(self, all_results: Dict, recommendations: Dict, output_file: str):
        """Generate human-readable markdown report"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('# Breakthrough Methodology Extraction Report\n\n')
            f.write('**Generated from DD-RAPTOR ChromaDB Analysis**\n\n')

            # Executive Summary
            f.write('## Executive Summary\n\n')
            exec_sum = recommendations.get('executive_summary', {})
            f.write(f'- **Total Unique Papers Analyzed**: {exec_sum.get("total_papers_analyzed", 0)}\n')
            f.write(f'- **Categories Covered**: {exec_sum.get("categories_covered", 0)}\n')
            f.write(f'- **Breakthrough Methods Identified**: {exec_sum.get("breakthrough_methods_identified", 0)}\n')
            f.write(f'- **Performance Metrics Found**: {exec_sum.get("performance_metrics_found", 0)}\n\n')

            f.write('This report identifies cutting-edge research methodologies from developmental disability papers ')
            f.write('to enhance proposal success rate for QuantERA 2025 and Samsung grants.\n\n')

            # Proposal Enhancement Opportunities
            f.write('## Top Proposal Enhancement Opportunities\n\n')
            opportunities = recommendations.get('proposal_enhancement_opportunities', [])
            for i, opp in enumerate(opportunities[:10], 1):
                f.write(f'### {i}. {opp["category"]}\n\n')
                f.write(f'- **Opportunity**: {opp["opportunity"]}\n')
                f.write(f'- **Evidence Strength**: {opp["evidence_strength"]}\n')
                f.write(f'- **Priority**: {opp["implementation_priority"]}\n')
                f.write(f'- **Key Terms**: {", ".join(opp["matched_keywords"])}\n\n')

            # Detailed Category Analysis
            f.write('---\n\n')
            f.write('## Detailed Category Analysis\n\n')

            for category, data in recommendations.get('breakthrough_methods', {}).items():
                f.write(f'\n## {category}\n\n')
                f.write(f'**Total Relevant Papers**: {data["total_relevant_papers"]}\n\n')

                f.write('### Top Papers\n\n')
                for i, paper in enumerate(data['top_papers'], 1):
                    f.write(f'#### {i}. {paper["title"]}\n\n')
                    f.write(f'- **Relevance Score**: {paper["relevance_score"]}\n')
                    f.write(f'- **Matched Keywords**: {", ".join(paper["matched_keywords"])}\n')
                    f.write(f'- **Snippet**: {paper["snippet"]}...\n\n')

                if data['key_methods']:
                    f.write('### Key Methodologies Identified\n\n')
                    for i, method in enumerate(data['key_methods'][:10], 1):
                        f.write(f'{i}. {method}\n')
                    f.write('\n')

                if data['performance_metrics']:
                    f.write('### Performance Metrics\n\n')
                    for metric in data['performance_metrics'][:8]:
                        f.write(f'- {metric}\n')
                    f.write('\n')

            # Integration Strategies
            f.write('---\n\n')
            f.write('## Integration Strategies for Proposals\n\n')

            strategies = recommendations.get('integration_strategies', {})
            for category, strategy in strategies.items():
                f.write(f'### {category}\n\n')
                f.write(f'- **Primary Approach**: {strategy["primary_approach"]}\n')
                f.write(f'- **Expected Performance**: {strategy["expected_performance"]}\n')
                f.write(f'- **NeuroX-Fusion Integration**: {strategy["integration_with_neurox_fusion"]}\n')
                f.write(f'- **Samsung Proposal Angle**: {strategy["samsung_proposal_angle"]}\n\n')


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

        print(f'\n[INFO] Total unique papers: {recommendations["executive_summary"]["total_papers_analyzed"]}')
        print(f'[INFO] Methods identified: {recommendations["executive_summary"]["breakthrough_methods_identified"]}')
        print(f'[INFO] Metrics found: {recommendations["executive_summary"]["performance_metrics_found"]}')

        # Save results
        output_file = '/home/juke/git/AI-CoScientist/breakthrough_methodologies_2025.json'
        extractor.save_results(all_results, recommendations, output_file)

        print('\n' + '='*80)
        print('EXTRACTION COMPLETE')
        print('='*80)

        return 0

    except Exception as e:
        print(f'\n[FATAL ERROR] {e}', flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
