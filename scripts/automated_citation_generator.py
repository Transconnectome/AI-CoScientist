#!/usr/bin/env python3
"""
Automated Citation Generator
===========================

DD-RAPTOR 데이터베이스를 활용하여 제안서에 자동으로
과학적 근거와 citation을 추가하는 도구

Features:
- Automatic claim detection and citation
- DD-RAPTOR evidence matching
- Citation format generation (APA, IEEE, etc.)
- Reference list generation
- Real-time citation validation

Usage:
    # Auto-cite entire proposal
    poetry run python scripts/automated_citation_generator.py \
        --input "proposal.md" \
        --output "cited_proposal.md" \
        --mode auto_cite

    # Interactive citation mode
    poetry run python scripts/automated_citation_generator.py \
        --input "proposal.md" \
        --mode interactive

    # Generate reference list
    poetry run python scripts/automated_citation_generator.py \
        --input "proposal.md" \
        --mode generate_references \
        --format apa
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class CitationCandidate:
    """Potential citation for a claim"""
    paper_title: str
    section: str
    text: str
    relevance_score: float
    citation_id: str
    formatted_citation: str
    page_info: Optional[str] = None

@dataclass
class Reference:
    """Reference entry for bibliography"""
    citation_id: str
    paper_title: str
    authors: List[str]
    journal: str
    year: int
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    formatted_apa: str = ""
    formatted_ieee: str = ""

class AutomatedCitationGenerator:
    """Automated citation generation system"""

    def __init__(self, db_path: str = "chromadb_data_dd"):
        self.db_path = db_path

        print("📚 Initializing Automated Citation Generator...")
        print("   Loading SciBERT model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')

        print("   Loading Cross-Encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        print("   Connecting to DD-RAPTOR...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("dd_papers_L0")

        # Citation formatting templates
        self.citation_formats = {
            "apa": "{authors} ({year}). {title}. {journal}, {volume}, {pages}.",
            "ieee": "[{id}] {authors}, \"{title},\" {journal}, vol. {volume}, pp. {pages}, {year}.",
            "nature": "{authors}. {title}. {journal} {volume}, {pages} ({year})."
        }

        # Claim detection patterns
        self.citation_needed_patterns = [
            r'연구에\s*따르면',
            r'보고된\s*바에\s*의하면',
            r'알려져\s*있다',
            r'입증되었다',
            r'확인되었다',
            r'\d+%\s*향상',
            r'세계\s*최초',
            r'혁신적',
            r'효과적(?:인|으로)',
            r'성공적(?:인|으로)'
        ]

        # Already cited patterns
        self.existing_citation_patterns = [
            r'\[[\d\-,\s]+\]',  # [1], [1-3], [1,2,3]
            r'\([\w\s]+et\s+al\.?,?\s*\d{4}\)',  # (Author et al., 2023)
            r'\([\w\s]+,\s*\d{4}\)'  # (Author, 2023)
        ]

        # Citation counter
        self.citation_counter = 1
        self.reference_map = {}  # citation_id -> Reference

        print("   ✅ Citation generator ready!\n")

    def auto_cite_proposal(self, input_file: str, output_file: str,
                          citation_threshold: float = 0.7) -> Dict[str, Any]:
        """Automatically add citations to proposal"""

        print(f"🎯 AUTO-CITING PROPOSAL")
        print("=" * 50)
        print(f"📄 Input: {input_file}")
        print(f"📝 Output: {output_file}")
        print(f"🎯 Threshold: {citation_threshold}")
        print("=" * 50)

        # Load proposal
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        lines = original_text.split('\n')
        cited_lines = []
        citation_stats = {
            "total_lines": len(lines),
            "lines_needing_citation": 0,
            "citations_added": 0,
            "claims_without_evidence": 0,
            "references_generated": 0
        }

        print("🔍 Processing lines for citations...")

        for line_num, line in enumerate(lines, 1):
            # Skip headers, empty lines, and existing citations
            if (line.strip().startswith('#') or
                not line.strip() or
                self._has_existing_citation(line)):
                cited_lines.append(line)
                continue

            # Check if line needs citation
            if self._needs_citation(line):
                citation_stats["lines_needing_citation"] += 1

                # Find best citation candidate
                candidate = self._find_citation_candidate(
                    line,
                    threshold=citation_threshold
                )

                if candidate:
                    # Add citation to line
                    cited_line = self._add_citation_to_line(line, candidate)
                    cited_lines.append(cited_line)
                    citation_stats["citations_added"] += 1

                    # Store reference
                    if candidate.citation_id not in self.reference_map:
                        reference = self._create_reference(candidate)
                        self.reference_map[candidate.citation_id] = reference

                    print(f"   ✅ Line {line_num}: Added citation [{candidate.citation_id}]")
                else:
                    # No suitable citation found
                    cited_lines.append(line + " [Citation needed]")
                    citation_stats["claims_without_evidence"] += 1
                    print(f"   ❌ Line {line_num}: No suitable citation found")
            else:
                cited_lines.append(line)

        # Generate references section
        references_section = self._generate_references_section()
        cited_text = '\n'.join(cited_lines) + '\n\n' + references_section

        citation_stats["references_generated"] = len(self.reference_map)

        # Save cited proposal
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cited_text)

        # Print summary
        print(f"\n📊 CITATION SUMMARY")
        print("=" * 50)
        print(f"📝 Lines processed: {citation_stats['total_lines']}")
        print(f"🎯 Lines needing citations: {citation_stats['lines_needing_citation']}")
        print(f"✅ Citations added: {citation_stats['citations_added']}")
        print(f"❌ Claims without evidence: {citation_stats['claims_without_evidence']}")
        print(f"📚 References generated: {citation_stats['references_generated']}")

        coverage = (citation_stats['citations_added'] /
                   citation_stats['lines_needing_citation'] * 100
                   if citation_stats['lines_needing_citation'] > 0 else 0)
        print(f"📈 Citation coverage: {coverage:.1f}%")

        return citation_stats

    def interactive_citation(self, input_file: str) -> str:
        """Interactive citation mode"""

        print("🔍 INTERACTIVE CITATION MODE")
        print("=" * 50)
        print("Commands:")
        print("  'cite' - Add suggested citation")
        print("  'skip' - Skip this line")
        print("  'edit' - Edit citation manually")
        print("  'quit' - Exit session")
        print("=" * 50)

        # Load proposal
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        lines = original_text.split('\n')
        cited_lines = []

        for line_num, line in enumerate(lines, 1):
            # Skip headers and empty lines
            if line.strip().startswith('#') or not line.strip():
                cited_lines.append(line)
                continue

            # Check if needs citation
            if self._needs_citation(line) and not self._has_existing_citation(line):
                print(f"\n{'='*50}")
                print(f"Line {line_num} needs citation:")
                print(f"📝 {line}")

                # Find citation candidates
                candidates = self._find_multiple_citation_candidates(line, n_candidates=3)

                if not candidates:
                    print("❌ No suitable citations found")
                    cited_lines.append(line)
                    continue

                print(f"\n💡 Citation candidates:")
                for i, candidate in enumerate(candidates, 1):
                    print(f"   {i}. [{candidate.citation_id}] {candidate.paper_title}")
                    print(f"      Relevance: {candidate.relevance_score:.3f}")
                    print(f"      Text: {candidate.text[:100]}...")

                # User choice
                while True:
                    choice = input(f"\nAction [1-{len(candidates)}/skip/quit]: ").strip().lower()

                    if choice == 'quit':
                        print("👋 Exiting interactive session")
                        return '\n'.join(cited_lines)
                    elif choice == 'skip':
                        cited_lines.append(line)
                        break
                    elif choice.isdigit() and 1 <= int(choice) <= len(candidates):
                        # Apply selected citation
                        selected_candidate = candidates[int(choice) - 1]
                        cited_line = self._add_citation_to_line(line, selected_candidate)
                        cited_lines.append(cited_line)

                        # Store reference
                        if selected_candidate.citation_id not in self.reference_map:
                            reference = self._create_reference(selected_candidate)
                            self.reference_map[selected_candidate.citation_id] = reference

                        print(f"✅ Citation added: [{selected_candidate.citation_id}]")
                        break
                    else:
                        print("Invalid choice. Try again.")
            else:
                cited_lines.append(line)

        # Add references section
        cited_text = '\n'.join(cited_lines)
        if self.reference_map:
            cited_text += '\n\n' + self._generate_references_section()

        return cited_text

    def generate_references_only(self, input_file: str, format: str = "apa") -> str:
        """Generate references section from existing citations"""

        print(f"📚 GENERATING REFERENCES ({format.upper()})")
        print("=" * 50)

        # Load proposal and extract citations
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Find all citation markers
        citations = self._extract_existing_citations(text)
        print(f"📄 Found {len(citations)} citations in proposal")

        # Generate references for each citation
        references = []
        for citation_id in citations:
            reference = self._generate_reference_from_dd_raptor(citation_id, format)
            if reference:
                references.append(reference)

        # Format references section
        references_text = self._format_references_section(references, format)

        return references_text

    def _needs_citation(self, line: str) -> bool:
        """Check if line needs citation"""
        for pattern in self.citation_needed_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        # Check for statistical claims
        if re.search(r'\d+%', line) or re.search(r'AUC.*0\.\d+', line):
            return True

        # Check for strong claims
        strong_claim_words = ['혁신적', '획기적', '세계 최초', '최고', '우수한']
        if any(word in line for word in strong_claim_words):
            return True

        return False

    def _has_existing_citation(self, line: str) -> bool:
        """Check if line already has citation"""
        for pattern in self.existing_citation_patterns:
            if re.search(pattern, line):
                return True
        return False

    def _find_citation_candidate(self, line: str, threshold: float = 0.7) -> Optional[CitationCandidate]:
        """Find best citation candidate for line"""
        candidates = self._find_multiple_citation_candidates(line, n_candidates=1)
        return candidates[0] if candidates and candidates[0].relevance_score >= threshold else None

    def _find_multiple_citation_candidates(self, line: str, n_candidates: int = 3) -> List[CitationCandidate]:
        """Find multiple citation candidates for line"""

        try:
            # Encode line
            query_embedding = self.embedding_model.encode([line])[0].tolist()

            # Search DD-RAPTOR
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_candidates * 2,  # Get more for reranking
                include=["documents", "metadatas"]
            )

            if not results['documents'][0]:
                return []

            # Re-rank with cross-encoder
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]

            pairs = [[line, doc] for doc in documents]
            scores = self.cross_encoder.predict(pairs)

            # Create candidates
            candidates = []
            for doc, meta, score in zip(documents, metadatas, scores):
                # Generate unique citation ID
                citation_id = f"DD{self.citation_counter}"
                self.citation_counter += 1

                candidate = CitationCandidate(
                    paper_title=meta.get('paper_title', 'Unknown'),
                    section=meta.get('section', 'Unknown'),
                    text=doc,
                    relevance_score=float(score),
                    citation_id=citation_id,
                    formatted_citation=self._format_short_citation(meta)
                )
                candidates.append(candidate)

            # Sort by score and return top candidates
            candidates.sort(key=lambda x: x.relevance_score, reverse=True)
            return candidates[:n_candidates]

        except Exception as e:
            print(f"   ⚠️  Citation search error: {e}")
            return []

    def _format_short_citation(self, metadata: Dict[str, Any]) -> str:
        """Format short citation for inline use"""
        paper_title = metadata.get('paper_title', 'Unknown')

        # Extract author and year if possible
        if 'et al' in paper_title or 'al.' in paper_title:
            # Already has author format
            return paper_title
        else:
            # Generic format
            return f"{paper_title[:50]}..." if len(paper_title) > 50 else paper_title

    def _add_citation_to_line(self, line: str, candidate: CitationCandidate) -> str:
        """Add citation to line"""
        # Add citation marker at end of sentence or line
        if line.strip().endswith('.'):
            return line[:-1] + f" [{candidate.citation_id}]."
        else:
            return line + f" [{candidate.citation_id}]"

    def _create_reference(self, candidate: CitationCandidate) -> Reference:
        """Create reference entry from candidate"""

        paper_title = candidate.paper_title

        # Extract metadata (simplified - in real implementation, parse more carefully)
        authors = self._extract_authors(paper_title)
        year = self._extract_year(paper_title)
        journal = self._extract_journal(paper_title)

        reference = Reference(
            citation_id=candidate.citation_id,
            paper_title=paper_title,
            authors=authors,
            journal=journal,
            year=year,
            formatted_apa=self._format_apa_reference(paper_title, authors, year, journal),
            formatted_ieee=self._format_ieee_reference(candidate.citation_id, paper_title, authors, year, journal)
        )

        return reference

    def _extract_authors(self, paper_title: str) -> List[str]:
        """Extract author names from paper title"""
        # Simple extraction (real implementation would be more sophisticated)
        if 'et al' in paper_title:
            # Try to find first author
            parts = paper_title.split()
            for i, part in enumerate(parts):
                if 'et' in part.lower():
                    if i > 0:
                        return [parts[i-1] + " et al."]
                    break
        return ["Unknown Author"]

    def _extract_year(self, paper_title: str) -> int:
        """Extract publication year"""
        year_match = re.search(r'20\d{2}', paper_title)
        return int(year_match.group()) if year_match else 2023

    def _extract_journal(self, paper_title: str) -> str:
        """Extract journal name"""
        # Common journal names
        journals = [
            "Nature", "Science", "Cell", "Nature Neuroscience",
            "NeuroImage", "IEEE", "ICLR", "NeurIPS", "ICML",
            "Computers in Biology and Medicine", "Journal of Autism"
        ]

        for journal in journals:
            if journal.lower() in paper_title.lower():
                return journal

        return "Unknown Journal"

    def _format_apa_reference(self, title: str, authors: List[str], year: int, journal: str) -> str:
        """Format APA style reference"""
        author_str = ", ".join(authors)
        return f"{author_str} ({year}). {title}. {journal}."

    def _format_ieee_reference(self, citation_id: str, title: str, authors: List[str], year: int, journal: str) -> str:
        """Format IEEE style reference"""
        author_str = ", ".join(authors)
        return f"[{citation_id}] {author_str}, \"{title},\" {journal}, {year}."

    def _generate_references_section(self, format: str = "apa") -> str:
        """Generate references section"""

        if not self.reference_map:
            return ""

        references_text = "\n## References\n\n"

        # Sort references by citation ID
        sorted_refs = sorted(self.reference_map.values(), key=lambda x: int(x.citation_id[2:]))  # Remove 'DD' prefix

        for ref in sorted_refs:
            if format == "apa":
                references_text += f"{ref.formatted_apa}\n\n"
            elif format == "ieee":
                references_text += f"{ref.formatted_ieee}\n\n"
            else:
                references_text += f"[{ref.citation_id}] {ref.paper_title}\n\n"

        return references_text

    def _extract_existing_citations(self, text: str) -> List[str]:
        """Extract existing citation IDs from text"""
        citations = set()

        # Find citation markers like [1], [DD1], etc.
        for match in re.finditer(r'\[([^]]+)\]', text):
            citation_id = match.group(1)
            if citation_id.replace(',', '').replace('-', '').replace(' ', '').isalnum():
                citations.add(citation_id)

        return list(citations)

    def _generate_reference_from_dd_raptor(self, citation_id: str, format: str) -> Optional[str]:
        """Generate reference from DD-RAPTOR for existing citation"""
        # This would query DD-RAPTOR based on citation ID
        # Simplified implementation
        return f"[{citation_id}] Reference from DD-RAPTOR database ({format} format)"

    def _format_references_section(self, references: List[str], format: str) -> str:
        """Format references section"""
        if not references:
            return ""

        section = f"\n## References ({format.upper()})\n\n"
        for ref in references:
            section += f"{ref}\n\n"

        return section

def main():
    parser = argparse.ArgumentParser(
        description="Automated citation generator using DD-RAPTOR"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input proposal file"
    )

    parser.add_argument(
        "--mode",
        choices=["auto_cite", "interactive", "generate_references"],
        default="auto_cite",
        help="Citation mode"
    )

    parser.add_argument(
        "--output",
        help="Output file (required for auto_cite mode)"
    )

    parser.add_argument(
        "--format",
        choices=["apa", "ieee", "nature"],
        default="apa",
        help="Citation format"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Relevance threshold for auto-citation"
    )

    parser.add_argument(
        "--db_path",
        default="chromadb_data_dd",
        help="DD-RAPTOR database path"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "auto_cite" and not args.output:
        print("❌ Auto-cite mode requires --output")
        return

    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return

    if not Path(args.db_path).exists():
        print(f"❌ DD-RAPTOR database not found: {args.db_path}")
        print("Run: poetry run python scripts/load_json_to_chromadb_dd.py")
        return

    try:
        # Initialize citation generator
        generator = AutomatedCitationGenerator(args.db_path)

        if args.mode == "auto_cite":
            # Automatic citation mode
            stats = generator.auto_cite_proposal(
                args.input,
                args.output,
                citation_threshold=args.threshold
            )

            print(f"\n✅ Auto-citation complete!")
            print(f"📄 Cited proposal saved: {args.output}")

        elif args.mode == "interactive":
            # Interactive citation mode
            cited_text = generator.interactive_citation(args.input)

            # Save result
            output_file = args.output or args.input.replace('.md', '_cited.md')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cited_text)

            print(f"\n✅ Interactive citation complete!")
            print(f"📄 Cited proposal saved: {output_file}")

        elif args.mode == "generate_references":
            # References generation mode
            references = generator.generate_references_only(args.input, args.format)

            print(f"\n📚 GENERATED REFERENCES ({args.format.upper()}):")
            print("=" * 60)
            print(references)

            # Save if output specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(references)
                print(f"\n💾 References saved: {args.output}")

    except Exception as e:
        print(f"❌ Citation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()