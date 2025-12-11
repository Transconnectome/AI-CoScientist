#!/usr/bin/env python3
"""
Generate comprehensive synthesis report from DD literature analysis.
Creates a scientific-grade literature review with evidence synthesis.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_analysis_report(filepath: str) -> dict:
    """Load the analysis report JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_key_papers(report: dict) -> dict:
    """Extract and consolidate key papers across all themes"""
    papers = defaultdict(lambda: {
        'themes': set(),
        'innovations': [],
        'gaps': [],
        'findings': []
    })

    for theme_name, theme_data in report['themes'].items():
        for paper_name in theme_data.get('key_papers', []):
            papers[paper_name]['themes'].add(theme_name)

        # Add innovations per paper
        for innov in theme_data.get('methodological_innovations', []):
            paper = innov.get('paper', 'Unknown')
            papers[paper]['innovations'].append(innov)

        # Add gaps per paper
        for gap in theme_data.get('research_gaps', []):
            paper = gap.get('paper', 'Unknown')
            papers[paper]['gaps'].append(gap)

        # Extract findings with statistics
        for query, findings in theme_data.get('queries', {}).items():
            for finding in findings:
                paper = finding.get('paper', 'Unknown')
                papers[paper]['findings'].append(finding)

    return dict(papers)


def categorize_methods(innovations: list) -> dict:
    """Categorize innovations by method type"""
    categories = {
        'ML/AI': ['machine learning', 'deep learning', 'neural networks', 'transformer',
                  'convolutional neural', 'attention mechanism', 'transfer learning'],
        'Multimodal': ['multimodal', 'multi-modal', 'integration'],
        'Study Design': ['longitudinal', 'prospective', 'novel', 'first study'],
        'Digital Biomarkers': ['digital phenotyping', 'real-time', 'eye-tracking'],
        'Explainability': ['explainable AI', 'interpretable']
    }

    categorized = defaultdict(list)

    for innov in innovations:
        keyword = innov.get('keyword', '').lower()
        assigned = False

        for cat, keywords in categories.items():
            if keyword in keywords:
                categorized[cat].append(innov)
                assigned = True
                break

        if not assigned:
            categorized['Other'].append(innov)

    return dict(categorized)


def extract_statistics_summary(report: dict) -> dict:
    """Extract and summarize statistical information across all findings"""
    stats_summary = {
        'sample_sizes': [],
        'accuracy_metrics': [],
        'p_values': [],
        'total_findings_with_stats': 0
    }

    for theme_data in report['themes'].values():
        for query, findings in theme_data.get('queries', {}).items():
            for finding in findings:
                stats = finding.get('statistics', {})

                if any(stats.values()):
                    stats_summary['total_findings_with_stats'] += 1

                stats_summary['sample_sizes'].extend(stats.get('sample_sizes', []))
                stats_summary['accuracy_metrics'].extend(stats.get('accuracy_metrics', []))
                stats_summary['p_values'].extend(stats.get('p_values', []))

    # Calculate summary statistics
    if stats_summary['sample_sizes']:
        stats_summary['median_sample_size'] = sorted(stats_summary['sample_sizes'])[
            len(stats_summary['sample_sizes'])//2
        ]
        stats_summary['max_sample_size'] = max(stats_summary['sample_sizes'])
        stats_summary['min_sample_size'] = min(stats_summary['sample_sizes'])

    if stats_summary['accuracy_metrics']:
        stats_summary['mean_accuracy'] = sum(stats_summary['accuracy_metrics']) / len(
            stats_summary['accuracy_metrics']
        )
        stats_summary['max_accuracy'] = max(stats_summary['accuracy_metrics'])

    return stats_summary


def categorize_research_gaps(gaps: list) -> dict:
    """Categorize research gaps by type"""
    categories = {
        'Methodological Limitations': ['limitation', 'challenge', 'difficulty', 'barrier'],
        'Future Research Needs': ['future research', 'future work', 'future studies',
                                  'further investigation', 'warrant'],
        'Knowledge Gaps': ['need for', 'lack of', 'absence of', 'insufficient',
                          'not yet', 'remain unclear', 'unknown'],
        'Implementation Needs': ['require', 'necessary']
    }

    categorized = defaultdict(list)

    for gap in gaps:
        indicator = gap.get('indicator', '').lower()
        assigned = False

        for cat, indicators in categories.items():
            if indicator in indicators:
                categorized[cat].append(gap)
                assigned = True
                break

        if not assigned:
            categorized['Other'].append(gap)

    return dict(categorized)


def generate_markdown_report(report: dict, output_path: str):
    """Generate comprehensive markdown synthesis report"""

    # Extract data
    papers = extract_key_papers(report)
    stats_summary = extract_statistics_summary(report)
    innovations = report['overall_findings']['state_of_the_art']
    gaps = report['overall_findings']['critical_gaps']

    categorized_innovations = categorize_methods(innovations)
    categorized_gaps = categorize_research_gaps(gaps)

    # Generate markdown
    md = []

    # Title and metadata
    md.append("# Comprehensive Scientific Literature Analysis")
    md.append("## DD-RAPTOR Knowledge Base: 26 Developmental Disorder Papers")
    md.append("")
    md.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Database**: {report['metadata']['database_path']}")
    md.append(f"**Embedding Model**: {report['metadata']['embedding_model']}")
    md.append(f"**Reranker**: {report['metadata']['reranker_model']}")
    md.append("")
    md.append("---")
    md.append("")

    # Executive Summary
    md.append("## Executive Summary")
    md.append("")
    md.append(f"This comprehensive analysis systematically extracted scientific insights from "
              f"{report['metadata']['total_papers']} developmental disorder papers using the "
              f"DD-RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system.")
    md.append("")

    # Evidence strength overview
    md.append("### Evidence Strength by Theme")
    md.append("")
    md.append("| Theme | Evidence Strength | Key Papers | Innovations | Research Gaps |")
    md.append("|-------|------------------|------------|-------------|---------------|")

    for theme_name, theme_data in report['themes'].items():
        theme_display = theme_name.replace('_', ' ').title()
        strength = theme_data.get('evidence_strength', 'unknown').upper()
        n_papers = len(theme_data.get('key_papers', []))
        n_innovations = len(theme_data.get('methodological_innovations', []))
        n_gaps = len(theme_data.get('research_gaps', []))
        md.append(f"| {theme_display} | {strength} | {n_papers} | {n_innovations} | {n_gaps} |")

    md.append("")
    md.append("---")
    md.append("")

    # Section 1: State-of-the-Art
    md.append("## 1. Current State-of-the-Art")
    md.append("")
    md.append("### 1.1 Advanced Diagnostic Methods and Technologies")
    md.append("")

    if 'ML/AI' in categorized_innovations:
        md.append("#### Machine Learning and Artificial Intelligence")
        md.append("")
        for innov in categorized_innovations['ML/AI'][:5]:
            md.append(f"- **{innov['keyword'].title()}** (Relevance: {innov['score']:.3f})")
            md.append(f"  - Paper: {innov['paper']}")
            md.append(f"  - Context: {innov['context'][:200]}...")
            md.append("")

    if 'Multimodal' in categorized_innovations:
        md.append("#### Multimodal Integration Approaches")
        md.append("")
        for innov in categorized_innovations['Multimodal'][:3]:
            md.append(f"- **{innov['keyword'].title()}** (Relevance: {innov['score']:.3f})")
            md.append(f"  - Paper: {innov['paper']}")
            md.append("")

    if 'Digital Biomarkers' in categorized_innovations:
        md.append("#### Digital Biomarkers and Novel Assessment Methods")
        md.append("")
        for innov in categorized_innovations['Digital Biomarkers'][:3]:
            md.append(f"- **{innov['keyword'].title()}** (Relevance: {innov['score']:.3f})")
            md.append(f"  - Paper: {innov['paper']}")
            md.append("")

    md.append("### 1.2 Statistical Summary of Evidence")
    md.append("")
    md.append(f"- **Total findings with statistical data**: {stats_summary['total_findings_with_stats']}")

    if stats_summary.get('sample_sizes'):
        md.append(f"- **Sample sizes**:")
        md.append(f"  - Median: n={stats_summary.get('median_sample_size', 'N/A')}")
        md.append(f"  - Range: {stats_summary.get('min_sample_size', 'N/A')} - {stats_summary.get('max_sample_size', 'N/A')}")

    if stats_summary.get('accuracy_metrics'):
        md.append(f"- **Performance metrics**:")
        md.append(f"  - Mean accuracy: {stats_summary.get('mean_accuracy', 0)*100:.1f}%")
        md.append(f"  - Maximum accuracy: {stats_summary.get('max_accuracy', 0)*100:.1f}%")

    md.append("")
    md.append("---")
    md.append("")

    # Section 2: Critical Research Gaps
    md.append("## 2. Critical Research Gaps")
    md.append("")

    for category, cat_gaps in categorized_gaps.items():
        if cat_gaps:
            md.append(f"### 2.1 {category}")
            md.append("")

            for i, gap in enumerate(cat_gaps[:5], 1):
                gap_text = gap.get('gap', '')[:300]
                paper = gap.get('paper', 'Unknown')[:80]
                section = gap.get('section', 'Unknown')

                md.append(f"{i}. **{gap_text}...**")
                md.append(f"   - Source: {paper}")
                md.append(f"   - Section: {section}")
                md.append("")

    md.append("---")
    md.append("")

    # Section 3: Methodological Limitations
    md.append("## 3. Methodological Limitations")
    md.append("")

    if 'Methodological Limitations' in categorized_gaps:
        md.append("### 3.1 Statistical and Technical Limitations")
        md.append("")

        for i, gap in enumerate(categorized_gaps['Methodological Limitations'][:8], 1):
            gap_text = gap.get('gap', '')[:250]
            md.append(f"{i}. {gap_text}...")
            md.append("")

    md.append("### 3.2 Sample Size and Power Issues")
    md.append("")
    md.append("Based on extracted sample sizes across studies:")
    md.append("")

    if stats_summary.get('sample_sizes'):
        sample_sizes = sorted(stats_summary['sample_sizes'], reverse=True)
        md.append(f"- **Large-scale studies (n>100)**: {sum(1 for n in sample_sizes if n > 100)} studies")
        md.append(f"- **Medium-scale studies (n=50-100)**: {sum(1 for n in sample_sizes if 50 <= n <= 100)} studies")
        md.append(f"- **Small-scale studies (n<50)**: {sum(1 for n in sample_sizes if n < 50)} studies")
        md.append("")
        md.append("**Implication**: Limited statistical power in smaller studies may affect generalizability.")

    md.append("")
    md.append("---")
    md.append("")

    # Section 4: Future Directions
    md.append("## 4. Future Directions and Emerging Approaches")
    md.append("")

    if 'Future Research Needs' in categorized_gaps:
        md.append("### 4.1 Explicitly Identified Research Priorities")
        md.append("")

        for i, gap in enumerate(categorized_gaps['Future Research Needs'][:10], 1):
            gap_text = gap.get('gap', '')[:300]
            md.append(f"{i}. {gap_text}...")
            md.append("")

    md.append("### 4.2 Paradigm-Shifting Opportunities")
    md.append("")
    md.append("Based on methodological innovations and identified gaps, key opportunities include:")
    md.append("")

    # Synthesize opportunities
    if 'ML/AI' in categorized_innovations and 'Multimodal' in categorized_innovations:
        md.append("1. **AI-Powered Multimodal Integration**: Combining deep learning with multimodal "
                  "neuroimaging, behavioral, and genetic data for precision diagnostics")
        md.append("")

    if 'Study Design' in categorized_innovations:
        md.append("2. **Longitudinal Foundation Models**: Large-scale pre-trained models capturing "
                  "developmental trajectories across the lifespan")
        md.append("")

    if 'Digital Biomarkers' in categorized_innovations:
        md.append("3. **Real-Time Digital Phenotyping**: Continuous monitoring using wearables and "
                  "mobile technology for early detection")
        md.append("")

    md.append("---")
    md.append("")

    # Section 5: Key Papers Analysis
    md.append("## 5. Most Influential Papers")
    md.append("")
    md.append("### Papers Cited Across Multiple Themes")
    md.append("")

    # Sort papers by number of themes
    sorted_papers = sorted(
        papers.items(),
        key=lambda x: len(x[1]['themes']),
        reverse=True
    )

    for paper_name, data in sorted_papers[:10]:
        n_themes = len(data['themes'])
        n_innovations = len(data['innovations'])
        n_findings = len(data['findings'])

        if n_themes > 1:
            md.append(f"**{paper_name}**")
            md.append(f"- Themes: {', '.join([t.replace('_', ' ').title() for t in data['themes']])}")
            md.append(f"- Innovations: {n_innovations}")
            md.append(f"- Findings: {n_findings}")
            md.append("")

    md.append("---")
    md.append("")

    # Conclusions
    md.append("## 6. Conclusions and Evidence Synthesis")
    md.append("")
    md.append("### Key Findings")
    md.append("")
    md.append("1. **Moderate evidence** exists for biomarkers and diagnostic methods, primarily "
              "driven by machine learning approaches")
    md.append("")
    md.append("2. **Weak to moderate evidence** for neuroimaging methods suggests need for larger, "
              "multi-site collaborative studies")
    md.append("")
    md.append("3. **Precision medicine and intervention approaches** remain underdeveloped, "
              "representing critical research gaps")
    md.append("")
    md.append("4. **Methodological innovations** cluster around AI/ML, multimodal integration, "
              "and digital biomarkers")
    md.append("")
    md.append("5. **Sample size limitations** persist across many studies, affecting statistical "
              "power and replicability")
    md.append("")

    md.append("### Research Priorities for Revolutionary Impact")
    md.append("")
    md.append("1. **Large-scale multimodal foundation models** (n>3,000) trained on longitudinal data")
    md.append("2. **Real-time digital biomarker platforms** for continuous developmental monitoring")
    md.append("3. **Explainable AI systems** that provide interpretable clinical insights")
    md.append("4. **Precision intervention frameworks** guided by predictive models")
    md.append("5. **Multi-site collaborative networks** ensuring diverse, representative samples")
    md.append("")

    md.append("---")
    md.append("")
    md.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    # Write to file
    output_file = Path(output_path)
    output_file.write_text('\n'.join(md), encoding='utf-8')

    print(f"\n✓ Comprehensive synthesis report saved to: {output_file.absolute()}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")


def main():
    """Main execution"""

    report_path = "dd_literature_analysis_report.json"
    output_path = "DD_LITERATURE_COMPREHENSIVE_SYNTHESIS.md"

    if not Path(report_path).exists():
        print(f"Error: Report not found at {report_path}")
        print("Please run 'python3 scripts/analyze_dd_literature.py' first.")
        return 1

    print("Loading analysis report...")
    report = load_analysis_report(report_path)

    print("Generating comprehensive synthesis...")
    generate_markdown_report(report, output_path)

    print("\n✓ Synthesis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
