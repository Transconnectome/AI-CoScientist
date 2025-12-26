#!/usr/bin/env python3
"""
Enhanced DD-RAPTOR Query Tool
============================

고급 검색 기능과 citation-ready 출력을 제공하는 DD-RAPTOR 쿼리 도구

Features:
- Advanced search with filters and ranking
- Citation-ready output format
- Systematic literature review mode
- Export capabilities (JSON, CSV, BibTeX)
- Multi-query batch processing

Usage:
    # Basic search
    poetry run python scripts/enhanced_dd_query.py \
        --query "korean developmental disorder foundation model" \
        --n_results 10

    # Systematic review mode
    poetry run python scripts/enhanced_dd_query.py \
        --mode systematic_review \
        --topic "brain imaging autism" \
        --output_format citation_ready

    # Batch processing
    poetry run python scripts/enhanced_dd_query.py \
        --batch queries.txt \
        --export results.json
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
from tqdm import tqdm
import re

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    paper_title: str
    section: str
    text: str
    relevance_score: float
    paper_id: str
    section_id: str
    chunk_id: str
    citation: str
    keywords: List[str]
    confidence: str  # "high", "medium", "low"

@dataclass
class SearchSummary:
    """Search session summary"""
    query: str
    total_results: int
    high_confidence: int
    medium_confidence: int
    low_confidence: int
    top_papers: List[str]
    key_findings: List[str]
    suggested_citations: List[str]

class EnhancedDDQuery:
    """Enhanced DD-RAPTOR query engine"""

    def __init__(self, db_path: str = "chromadb_data_dd"):
        self.db_path = db_path

        print("🚀 Initializing Enhanced DD Query Engine...")
        print("   Loading SciBERT embedding model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')

        print("   Loading Cross-Encoder for re-ranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print("   Connecting to DD-RAPTOR ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_path)

        # Try different collection names
        collections = self.client.list_collections()
        collection_names = [c.name for c in collections]

        if "dd_papers_L0" in collection_names:
            self.collection = self.client.get_collection("dd_papers_L0")
        elif "dd_papers" in collection_names:
            self.collection = self.client.get_collection("dd_papers")
        else:
            raise ValueError(f"DD papers collection not found. Available: {collection_names}")

        print(f"   ✅ Connected to collection: {self.collection.name}")

        # Search filters and enhancements
        self.paper_quality_weights = {
            "nature": 1.0,
            "science": 1.0,
            "cell": 0.9,
            "lancet": 0.9,
            "nejm": 0.9,
            "neurips": 0.8,
            "icml": 0.8,
            "iclr": 0.8,
            "default": 0.7
        }

        self.domain_keywords = {
            "foundation_models": ["foundation model", "large language model", "transformer", "pre-trained"],
            "brain_imaging": ["fMRI", "DTI", "EEG", "MEG", "neuroimaging", "brain scan"],
            "developmental_disorders": ["autism", "ADHD", "developmental delay", "neurodevelopment"],
            "genomics": ["genetic", "genomic", "SNP", "GWAS", "heritability", "mutation"],
            "korean": ["korean", "korea", "asian", "east asian", "korean population"]
        }

        print("   🎯 Engine ready for enhanced search!\n")

    def search(self, query: str, n_results: int = 10,
               filters: Optional[Dict[str, Any]] = None,
               rerank: bool = True) -> List[SearchResult]:
        """Enhanced search with filtering and ranking"""

        print(f"🔍 Searching: '{query}'")
        print(f"   Results requested: {n_results}")

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Perform initial search with more results for reranking
        search_n = min(n_results * 3, 50) if rerank else n_results

        try:
            raw_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_n,
                include=["documents", "metadatas", "distances"]
            )

            if not raw_results['documents'][0]:
                print("   ❌ No results found")
                return []

            print(f"   📚 Retrieved {len(raw_results['documents'][0])} initial results")

        except Exception as e:
            print(f"   ❌ Search error: {e}")
            return []

        # Process results
        results = []
        documents = raw_results['documents'][0]
        metadatas = raw_results['metadatas'][0]
        distances = raw_results['distances'][0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # Convert distance to similarity score
            similarity = 1.0 - distance if distance is not None else 0.0

            # Apply filters
            if filters and not self._passes_filters(meta, filters):
                continue

            # Create search result
            result = SearchResult(
                paper_title=meta.get('paper_title', 'Unknown'),
                section=meta.get('section', 'Unknown'),
                text=doc,
                relevance_score=similarity,
                paper_id=meta.get('paper_id', ''),
                section_id=meta.get('section_id', ''),
                chunk_id=meta.get('chunk_id', ''),
                citation=self._generate_citation(meta),
                keywords=self._extract_keywords(doc, query),
                confidence=self._calculate_confidence(similarity, meta)
            )
            results.append(result)

        # Re-rank with cross-encoder if requested
        if rerank and len(results) > 1:
            print("   🎯 Re-ranking with cross-encoder...")
            results = self._rerank_results(query, results)

        # Apply paper quality weighting
        results = self._apply_quality_weighting(results)

        # Sort by final score and return top n
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        final_results = results[:n_results]
        print(f"   ✅ Returning {len(final_results)} results")

        return final_results

    def systematic_review(self, topic: str, n_results: int = 20) -> Dict[str, Any]:
        """Perform systematic literature review on topic"""

        print(f"📊 SYSTEMATIC LITERATURE REVIEW")
        print(f"   Topic: {topic}")
        print("=" * 60)

        # Generate multiple search queries for comprehensive coverage
        search_queries = self._generate_systematic_queries(topic)

        all_results = []
        unique_papers = set()

        for query in tqdm(search_queries, desc="Searching"):
            results = self.search(query, n_results=10, rerank=True)

            # Deduplicate by paper title
            for result in results:
                paper_id = result.paper_title
                if paper_id not in unique_papers:
                    all_results.append(result)
                    unique_papers.add(paper_id)

        # Analyze results
        analysis = self._analyze_systematic_results(all_results, topic)

        # Generate systematic review report
        report = {
            "topic": topic,
            "search_strategy": search_queries,
            "total_papers": len(unique_papers),
            "total_chunks": len(all_results),
            "analysis": analysis,
            "results": [asdict(r) for r in all_results[:n_results]]
        }

        return report

    def batch_search(self, queries: List[str], n_results: int = 5) -> Dict[str, List[SearchResult]]:
        """Process multiple queries in batch"""

        print(f"📦 BATCH PROCESSING")
        print(f"   Queries: {len(queries)}")
        print(f"   Results per query: {n_results}")

        batch_results = {}

        for query in tqdm(queries, desc="Processing queries"):
            try:
                results = self.search(query, n_results=n_results)
                batch_results[query] = results
            except Exception as e:
                print(f"   ❌ Error processing '{query}': {e}")
                batch_results[query] = []

        return batch_results

    def _passes_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if result passes filters"""

        # Journal filter
        if "journals" in filters:
            paper_title = metadata.get('paper_title', '').lower()
            if not any(journal.lower() in paper_title for journal in filters["journals"]):
                return False

        # Year filter
        if "min_year" in filters:
            # Try to extract year from metadata or paper title
            year = self._extract_year(metadata)
            if year and year < filters["min_year"]:
                return False

        # Section filter
        if "sections" in filters:
            section = metadata.get('section', '').lower()
            if not any(sec.lower() in section for sec in filters["sections"]):
                return False

        return True

    def _generate_citation(self, metadata: Dict[str, Any]) -> str:
        """Generate citation string"""
        paper_title = metadata.get('paper_title', 'Unknown')
        section = metadata.get('section', '')

        if section and section != 'Unknown':
            return f"{paper_title} ({section})"
        else:
            return paper_title

    def _extract_keywords(self, text: str, query: str) -> List[str]:
        """Extract relevant keywords from text"""

        # Combine query terms with domain-specific keywords
        query_terms = query.lower().split()

        all_keywords = set(query_terms)

        # Add domain keywords if relevant
        text_lower = text.lower()
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    all_keywords.add(keyword)

        return list(all_keywords)

    def _calculate_confidence(self, score: float, metadata: Dict[str, Any]) -> str:
        """Calculate confidence level"""

        # Base score threshold
        if score >= 0.8:
            base_confidence = "high"
        elif score >= 0.6:
            base_confidence = "medium"
        else:
            base_confidence = "low"

        # Adjust for paper quality
        paper_title = metadata.get('paper_title', '').lower()
        for journal, weight in self.paper_quality_weights.items():
            if journal in paper_title and journal != "default":
                if weight >= 0.9:
                    if base_confidence == "medium":
                        return "high"
                break

        return base_confidence

    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results with cross-encoder"""

        if len(results) <= 1:
            return results

        # Prepare pairs for cross-encoder
        pairs = [[query, result.text] for result in results]

        try:
            scores = self.cross_encoder.predict(pairs)

            # Update relevance scores
            for result, new_score in zip(results, scores):
                result.relevance_score = float(new_score)

        except Exception as e:
            print(f"   ⚠️  Re-ranking failed: {e}")
            # Keep original scores

        return results

    def _apply_quality_weighting(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply paper quality weighting"""

        for result in results:
            paper_title = result.paper_title.lower()

            # Find matching journal weight
            weight = self.paper_quality_weights.get("default")
            for journal, journal_weight in self.paper_quality_weights.items():
                if journal != "default" and journal in paper_title:
                    weight = journal_weight
                    break

            # Apply weight
            result.relevance_score *= weight

        return results

    def _generate_systematic_queries(self, topic: str) -> List[str]:
        """Generate multiple queries for systematic review"""

        base_queries = [topic]

        # Add domain-specific variations
        if "foundation model" in topic.lower():
            base_queries.extend([
                f"{topic} transformer",
                f"{topic} pre-trained",
                f"{topic} large language model"
            ])

        if "brain" in topic.lower() or "neuro" in topic.lower():
            base_queries.extend([
                f"{topic} neuroimaging",
                f"{topic} fMRI",
                f"{topic} brain imaging"
            ])

        if "developmental" in topic.lower() or "autism" in topic.lower():
            base_queries.extend([
                f"{topic} autism spectrum",
                f"{topic} ADHD",
                f"{topic} neurodevelopment"
            ])

        # Add methodological variations
        base_queries.extend([
            f"{topic} deep learning",
            f"{topic} machine learning",
            f"{topic} artificial intelligence"
        ])

        # Remove duplicates and limit
        unique_queries = list(set(base_queries))
        return unique_queries[:8]  # Limit to 8 queries

    def _analyze_systematic_results(self, results: List[SearchResult], topic: str) -> Dict[str, Any]:
        """Analyze systematic review results"""

        if not results:
            return {"error": "No results to analyze"}

        # Paper statistics
        papers = list(set(r.paper_title for r in results))
        high_conf = len([r for r in results if r.confidence == "high"])
        medium_conf = len([r for r in results if r.confidence == "medium"])
        low_conf = len([r for r in results if r.confidence == "low"])

        # Most cited papers (by frequency in results)
        paper_counts = {}
        for result in results:
            paper_counts[result.paper_title] = paper_counts.get(result.paper_title, 0) + 1

        top_papers = sorted(paper_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Extract key findings
        high_conf_results = [r for r in results if r.confidence == "high"][:10]
        key_findings = [r.text[:200] + "..." for r in high_conf_results]

        # Generate suggested citations
        suggested_citations = list(set(r.citation for r in high_conf_results))

        return {
            "total_papers": len(papers),
            "confidence_breakdown": {
                "high": high_conf,
                "medium": medium_conf,
                "low": low_conf
            },
            "top_papers": [{"title": title, "frequency": count} for title, count in top_papers],
            "key_findings": key_findings,
            "suggested_citations": suggested_citations[:10]
        }

    def _extract_year(self, metadata: Dict[str, Any]) -> Optional[int]:
        """Extract publication year from metadata"""

        # Try metadata first
        if 'year' in metadata:
            try:
                return int(metadata['year'])
            except (ValueError, TypeError):
                pass

        # Try extracting from paper title
        paper_title = metadata.get('paper_title', '')
        year_match = re.search(r'20\d{2}', paper_title)
        if year_match:
            try:
                return int(year_match.group())
            except ValueError:
                pass

        return None

    def export_results(self, results: List[SearchResult],
                      format: str, output_file: str):
        """Export results in various formats"""

        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

        elif format == "csv":
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Paper Title', 'Section', 'Relevance Score',
                    'Confidence', 'Citation', 'Keywords', 'Text'
                ])
                for r in results:
                    writer.writerow([
                        r.paper_title, r.section, r.relevance_score,
                        r.confidence, r.citation, '; '.join(r.keywords),
                        r.text[:500]  # Truncate text for CSV
                    ])

        elif format == "bibtex":
            # Simple BibTeX export (basic format)
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, r in enumerate(results):
                    f.write(f"@article{{dd_paper_{i+1},\n")
                    f.write(f"  title={{{r.paper_title}}},\n")
                    f.write(f"  note={{Relevance: {r.relevance_score:.3f}, Confidence: {r.confidence}}}\n")
                    f.write(f"}}\n\n")

        print(f"📄 Results exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced DD-RAPTOR query tool with advanced features"
    )

    parser.add_argument("--query", help="Search query")
    parser.add_argument("--n_results", type=int, default=10, help="Number of results")
    parser.add_argument("--mode", choices=["search", "systematic_review"],
                       default="search", help="Search mode")
    parser.add_argument("--topic", help="Topic for systematic review")
    parser.add_argument("--batch", help="File with multiple queries (one per line)")
    parser.add_argument("--output_format", choices=["simple", "detailed", "citation_ready"],
                       default="detailed", help="Output format")
    parser.add_argument("--export", help="Export file (JSON/CSV/BibTeX)")
    parser.add_argument("--db_path", default="chromadb_data_dd", help="ChromaDB path")
    parser.add_argument("--no_rerank", action="store_true", help="Skip cross-encoder re-ranking")

    args = parser.parse_args()

    # Validate inputs
    if args.mode == "search" and not args.query and not args.batch:
        print("❌ Search mode requires --query or --batch")
        return

    if args.mode == "systematic_review" and not args.topic:
        print("❌ Systematic review mode requires --topic")
        return

    if not Path(args.db_path).exists():
        print(f"❌ DD-RAPTOR database not found: {args.db_path}")
        print("Run: poetry run python scripts/load_json_to_chromadb_dd.py")
        return

    try:
        # Initialize query engine
        engine = EnhancedDDQuery(args.db_path)

        if args.mode == "systematic_review":
            # Systematic review mode
            print("\n📊 SYSTEMATIC LITERATURE REVIEW MODE")
            print("=" * 60)

            report = engine.systematic_review(args.topic, args.n_results)

            # Print summary
            analysis = report["analysis"]
            print(f"📚 Papers found: {analysis['total_papers']}")
            print(f"🎯 High confidence results: {analysis['confidence_breakdown']['high']}")

            print(f"\n🏆 TOP PAPERS:")
            for paper in analysis["top_papers"][:5]:
                print(f"   • {paper['title']} (appears {paper['frequency']} times)")

            print(f"\n📝 KEY FINDINGS:")
            for finding in analysis["key_findings"][:3]:
                print(f"   • {finding}")

            # Export if requested
            if args.export:
                with open(args.export, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Full report exported to: {args.export}")

        elif args.batch:
            # Batch processing mode
            print("\n📦 BATCH PROCESSING MODE")
            print("=" * 60)

            with open(args.batch, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]

            batch_results = engine.batch_search(queries, args.n_results)

            # Print summary
            total_results = sum(len(results) for results in batch_results.values())
            print(f"📝 Processed {len(queries)} queries")
            print(f"📚 Total results: {total_results}")

            # Show top result per query
            for query, results in batch_results.items():
                if results:
                    top = results[0]
                    print(f"\n🔍 '{query}' → {top.paper_title} (score: {top.relevance_score:.3f})")

            # Export if requested
            if args.export:
                export_data = {q: [asdict(r) for r in results] for q, results in batch_results.items()}
                with open(args.export, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Results exported to: {args.export}")

        else:
            # Single query search mode
            print(f"\n🔍 ENHANCED SEARCH MODE")
            print("=" * 60)

            results = engine.search(
                args.query,
                n_results=args.n_results,
                rerank=not args.no_rerank
            )

            if not results:
                print("❌ No results found")
                return

            # Display results based on format
            if args.output_format == "simple":
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result.paper_title} (score: {result.relevance_score:.3f})")

            elif args.output_format == "citation_ready":
                print(f"\n📚 CITATION-READY RESULTS:")
                print("=" * 60)
                for i, result in enumerate(results, 1):
                    print(f"[{i}] {result.citation}")
                    print(f"    Relevance: {result.relevance_score:.3f} | Confidence: {result.confidence}")
                    print(f"    Keywords: {', '.join(result.keywords[:5])}")
                    print()

            else:  # detailed format
                print(f"\n📄 DETAILED RESULTS:")
                print("=" * 60)
                for i, result in enumerate(results, 1):
                    print(f"📄 Result {i} - Score: {result.relevance_score:.3f} | Confidence: {result.confidence}")
                    print(f"Title:   {result.paper_title}")
                    print(f"Section: {result.section}")
                    print(f"Text:    {result.text[:200]}...")
                    print(f"Keywords: {', '.join(result.keywords[:5])}")
                    print("-" * 60)

            # Export if requested
            if args.export:
                # Determine format from file extension
                if args.export.endswith('.json'):
                    format_type = 'json'
                elif args.export.endswith('.csv'):
                    format_type = 'csv'
                elif args.export.endswith('.bib'):
                    format_type = 'bibtex'
                else:
                    format_type = 'json'

                engine.export_results(results, format_type, args.export)

    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()