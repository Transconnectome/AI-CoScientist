#!/usr/bin/env python3
"""
QuantERA 2025: Batch Process 31 QML Papers
Processes all quantum machine learning papers for systematic analysis
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.ingest import QuantERAIngestor
from src.raptor import QuantERARAGTOR
from src.graph import QMLKnowledgeGraph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QMLBatchProcessor:
    """Batch processor for QML papers with progress tracking"""

    def __init__(self):
        self.papers_dir = Path("Papers")
        self.output_dir = Path("processed_output")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.ingestor = QuantERAIngestor()
        self.raptor = QuantERARAGTOR()
        self.knowledge_graph = QMLKnowledgeGraph()

        # Progress tracking
        self.results = {
            "processing_start": datetime.now().isoformat(),
            "papers_processed": [],
            "failed_papers": [],
            "statistics": {},
            "knowledge_graph_stats": {},
        }

    def get_paper_list(self) -> List[Path]:
        """Get list of PDF papers to process"""
        if not self.papers_dir.exists():
            logger.error(f"Papers directory {self.papers_dir} not found!")
            return []

        pdf_files = list(self.papers_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF papers to process")
        return sorted(pdf_files)

    def process_single_paper(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single paper through the QML-RAPTOR pipeline"""
        logger.info(f"Processing: {pdf_path.name}")

        try:
            # Step 1: Document ingestion
            logger.info(f"  Step 1/3: Ingesting {pdf_path.name}")
            doc = self.ingestor.process_paper(str(pdf_path))

            if not doc or not hasattr(doc, 'chunks') or len(doc.chunks) == 0:
                raise ValueError("No chunks extracted from document")

            # Step 2: RAPTOR tree construction
            logger.info(f"  Step 2/3: Building RAPTOR tree")
            source_metadata = {
                "paper_name": pdf_path.name,
                "paper_path": str(pdf_path),
                "processing_timestamp": datetime.now().isoformat()
            }
            raptor_tree = self.raptor.build_tree_from_chunks(doc.chunks, source_metadata)

            # Step 3: Knowledge graph integration
            logger.info(f"  Step 3/3: Adding to knowledge graph")
            # Extract text from all chunks for knowledge graph processing
            full_text = " ".join([chunk.get('content', '') for chunk in doc.chunks])
            concepts = self.knowledge_graph.extractor.extract_entities(full_text, pdf_path.stem)

            # Compile results
            result = {
                "paper_name": pdf_path.name,
                "status": "success",
                "chunks_count": len(doc.chunks),
                "math_elements": len(doc.mathematical_elements),
                "circuit_elements": len(doc.circuit_descriptions),
                "raptor_tree_levels": {
                    "L0": "processed",
                    "L1": "processed",
                    "L2": "processed"
                },
                "extracted_entities": len(concepts),
                "processing_time": datetime.now().isoformat()
            }

            # Save individual paper results
            output_file = self.output_dir / f"{pdf_path.stem}_processed.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "document_info": result,
                    "chunks_sample": doc.chunks[:3],  # First 3 chunks for inspection
                    "math_sample": doc.mathematical_elements[:5],
                    "entities_sample": [vars(entity) for entity in concepts[:10]]
                }, f, indent=2)

            logger.info(f"  ✅ Success: {result['chunks_count']} chunks, {result['extracted_entities']} entities")
            return result

        except Exception as e:
            logger.error(f"  ❌ Failed to process {pdf_path.name}: {str(e)}")
            return {
                "paper_name": pdf_path.name,
                "status": "failed",
                "error": str(e),
                "processing_time": datetime.now().isoformat()
            }

    def process_all_papers(self):
        """Process all papers in the Papers directory"""
        logger.info("🚀 Starting batch processing of QML papers")

        papers = self.get_paper_list()
        if not papers:
            logger.error("No papers found to process!")
            return

        total_papers = len(papers)
        logger.info(f"Processing {total_papers} papers...")

        for i, paper_path in enumerate(papers, 1):
            logger.info(f"\n📄 [{i}/{total_papers}] Processing: {paper_path.name}")

            result = self.process_single_paper(paper_path)

            if result["status"] == "success":
                self.results["papers_processed"].append(result)
            else:
                self.results["failed_papers"].append(result)

            # Save progress after each paper
            self.save_progress()

            # Progress update
            success_count = len(self.results["papers_processed"])
            failed_count = len(self.results["failed_papers"])
            logger.info(f"Progress: {i}/{total_papers} | Success: {success_count} | Failed: {failed_count}")

    def compile_statistics(self):
        """Compile final statistics"""
        successful_papers = self.results["papers_processed"]

        if not successful_papers:
            logger.warning("No papers processed successfully!")
            return

        # Aggregate statistics
        total_chunks = sum(p["chunks_count"] for p in successful_papers)
        total_math = sum(p["math_elements"] for p in successful_papers)
        total_circuits = sum(p["circuit_elements"] for p in successful_papers)
        total_concepts = sum(p["extracted_entities"] for p in successful_papers)

        self.results["statistics"] = {
            "total_papers_attempted": len(self.results["papers_processed"]) + len(self.results["failed_papers"]),
            "papers_successfully_processed": len(successful_papers),
            "papers_failed": len(self.results["failed_papers"]),
            "success_rate": len(successful_papers) / (len(successful_papers) + len(self.results["failed_papers"])) * 100,
            "total_chunks_extracted": total_chunks,
            "total_mathematical_elements": total_math,
            "total_circuit_descriptions": total_circuits,
            "total_concepts_extracted": total_concepts,
            "average_chunks_per_paper": total_chunks / len(successful_papers) if successful_papers else 0,
            "average_concepts_per_paper": total_concepts / len(successful_papers) if successful_papers else 0
        }

        # Knowledge graph statistics
        try:
            kg_stats = self.knowledge_graph.get_statistics()
            self.results["knowledge_graph_stats"] = kg_stats
        except Exception as e:
            logger.warning(f"Could not get knowledge graph stats: {e}")

        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"  Papers processed: {self.results['statistics']['papers_successfully_processed']}")
        logger.info(f"  Success rate: {self.results['statistics']['success_rate']:.1f}%")
        logger.info(f"  Total chunks: {self.results['statistics']['total_chunks_extracted']}")
        logger.info(f"  Total concepts: {self.results['statistics']['total_concepts_extracted']}")

    def save_progress(self):
        """Save current progress to file"""
        self.results["last_updated"] = datetime.now().isoformat()

        # Save main results
        results_file = self.output_dir / "batch_processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.debug(f"Progress saved to {results_file}")

    def generate_summary_report(self):
        """Generate a human-readable summary report"""
        report_lines = [
            "# QuantERA 2025 QML Papers Processing Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- Papers attempted: {self.results['statistics']['total_papers_attempted']}",
            f"- Successfully processed: {self.results['statistics']['papers_successfully_processed']}",
            f"- Failed: {self.results['statistics']['papers_failed']}",
            f"- **Success rate: {self.results['statistics']['success_rate']:.1f}%**",
            "",
            "## Content Analysis",
            f"- Total text chunks extracted: {self.results['statistics']['total_chunks_extracted']:,}",
            f"- Mathematical elements found: {self.results['statistics']['total_mathematical_elements']:,}",
            f"- Quantum circuit descriptions: {self.results['statistics']['total_circuit_descriptions']:,}",
            f"- QML concepts identified: {self.results['statistics']['total_concepts_extracted']:,}",
            "",
            "## Averages per Paper",
            f"- Chunks per paper: {self.results['statistics']['average_chunks_per_paper']:.1f}",
            f"- Concepts per paper: {self.results['statistics']['average_concepts_per_paper']:.1f}",
            "",
            "## Successful Papers",
        ]

        for paper in self.results["papers_processed"]:
            report_lines.append(f"- ✅ {paper['paper_name']} ({paper['chunks_count']} chunks, {paper['extracted_entities']} entities)")

        if self.results["failed_papers"]:
            report_lines.extend([
                "",
                "## Failed Papers",
            ])
            for paper in self.results["failed_papers"]:
                report_lines.append(f"- ❌ {paper['paper_name']} - {paper.get('error', 'Unknown error')}")

        report_content = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "processing_summary_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        logger.info(f"📄 Summary report saved to: {report_file}")

        return report_content

def main():
    """Main execution function"""
    logger.info("🎯 QuantERA 2025: QML Papers Batch Processing")
    logger.info("=" * 60)

    processor = QMLBatchProcessor()

    try:
        # Step 1: Process all papers
        processor.process_all_papers()

        # Step 2: Compile statistics
        processor.compile_statistics()

        # Step 3: Generate reports
        summary = processor.generate_summary_report()

        # Step 4: Final save
        processor.save_progress()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 Batch processing completed successfully!")
        logger.info(f"📁 Results saved in: {processor.output_dir}")
        logger.info("=" * 60)

        # Print summary to console
        print("\n" + summary)

    except KeyboardInterrupt:
        logger.info("\n⏸️ Processing interrupted by user")
        processor.save_progress()
    except Exception as e:
        logger.error(f"💥 Fatal error during processing: {e}")
        processor.save_progress()
        raise

if __name__ == "__main__":
    main()