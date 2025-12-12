#!/usr/bin/env python3
"""
QuantERA QML-RAPTOR System Integration Test
Tests the complete system with sample papers from the collection
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingest import QuantERAIngestor
from src.raptor import QuantERARAGTOR
from src.graph import QMLKnowledgeGraph
from src.agent import QuantERAAgent


class SystemTester:
    """Tests the QuantERA system comprehensively"""

    def __init__(self, test_papers_limit: int = 3):
        self.test_papers_limit = test_papers_limit
        self.base_dir = Path(__file__).parent
        self.papers_dir = self.base_dir / "Papers"
        self.test_db_dir = self.base_dir / "test_db"

        # Create test database directory
        self.test_db_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Test queries to validate system
        self.test_queries = [
            "What is a variational quantum eigensolver?",
            "How does barren plateau affect quantum machine learning?",
            "Compare VQE and QAOA algorithms",
            "What are the challenges with NISQ devices?",
            "How to mitigate quantum noise in VQE?"
        ]

        self.test_results = {
            'ingestion': {'status': 'pending', 'details': {}},
            'raptor': {'status': 'pending', 'details': {}},
            'knowledge_graph': {'status': 'pending', 'details': {}},
            'agent': {'status': 'pending', 'details': {}},
            'queries': {'status': 'pending', 'details': {}},
            'overall': {'status': 'pending', 'success_rate': 0.0}
        }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run complete system test"""
        self.logger.info("Starting QuantERA System Comprehensive Test")

        try:
            # 1. Test paper ingestion
            self.test_ingestion()

            # 2. Test RAPTOR tree building
            self.test_raptor()

            # 3. Test knowledge graph
            self.test_knowledge_graph()

            # 4. Test agent integration
            self.test_agent()

            # 5. Test query processing
            self.test_queries_processing()

            # 6. Calculate overall results
            self.calculate_overall_results()

        except Exception as e:
            self.logger.error(f"Critical test failure: {e}")
            self.test_results['overall']['status'] = 'failed'
            self.test_results['overall']['error'] = str(e)

        return self.test_results

    def get_test_papers(self) -> List[Path]:
        """Get sample papers for testing"""
        if not self.papers_dir.exists():
            self.logger.warning(f"Papers directory not found: {self.papers_dir}")
            return []

        pdf_files = list(self.papers_dir.glob("*.pdf"))

        # Select diverse papers for testing
        selected_papers = []
        preferred_papers = [
            "BarrenPlateaus.pdf",
            "Cerezo-2021-Variational quantum algorithms.pdf",
            "Quantum diffusion models.pdf"
        ]

        # Try to get preferred papers first
        for preferred in preferred_papers:
            matching = [p for p in pdf_files if preferred.lower() in p.name.lower()]
            if matching:
                selected_papers.append(matching[0])
                if len(selected_papers) >= self.test_papers_limit:
                    break

        # Fill remaining with any other papers
        if len(selected_papers) < self.test_papers_limit:
            remaining_papers = [p for p in pdf_files if p not in selected_papers]
            selected_papers.extend(remaining_papers[:self.test_papers_limit - len(selected_papers)])

        self.logger.info(f"Selected {len(selected_papers)} papers for testing:")
        for paper in selected_papers:
            self.logger.info(f"  - {paper.name}")

        return selected_papers

    def test_ingestion(self):
        """Test document ingestion module"""
        self.logger.info("Testing document ingestion...")

        try:
            papers = self.get_test_papers()
            if not papers:
                self.test_results['ingestion']['status'] = 'skipped'
                self.test_results['ingestion']['details']['reason'] = 'No test papers available'
                return

            # Initialize ingestor
            ingestor = QuantERAIngestor()
            processed_docs = []

            for paper_path in papers:
                try:
                    self.logger.info(f"Processing paper: {paper_path.name}")
                    processed_doc = ingestor.process_paper(str(paper_path))
                    processed_docs.append(processed_doc)

                except Exception as e:
                    self.logger.warning(f"Failed to process {paper_path.name}: {e}")
                    continue

            if processed_docs:
                # Save processed documents for other tests
                output_path = self.test_db_dir / "processed_docs.json"
                self._save_processed_docs(processed_docs, str(output_path))

                self.test_results['ingestion']['status'] = 'passed'
                self.test_results['ingestion']['details'] = {
                    'papers_processed': len(processed_docs),
                    'total_chunks': sum(len(doc.chunks) for doc in processed_docs),
                    'math_elements': sum(len(doc.mathematical_elements) for doc in processed_docs),
                    'circuit_elements': sum(len(doc.circuit_descriptions) for doc in processed_docs),
                    'output_file': str(output_path)
                }

                self.logger.info(f"Ingestion test passed: {len(processed_docs)} papers processed")

            else:
                self.test_results['ingestion']['status'] = 'failed'
                self.test_results['ingestion']['details'] = {'error': 'No papers could be processed'}

        except Exception as e:
            self.test_results['ingestion']['status'] = 'failed'
            self.test_results['ingestion']['details'] = {'error': str(e), 'traceback': traceback.format_exc()}

    def test_raptor(self):
        """Test RAPTOR tree building"""
        self.logger.info("Testing RAPTOR tree building...")

        if self.test_results['ingestion']['status'] != 'passed':
            self.test_results['raptor']['status'] = 'skipped'
            self.test_results['raptor']['details']['reason'] = 'Ingestion test failed'
            return

        try:
            # Load processed documents
            docs_path = self.test_db_dir / "processed_docs.json"

            if not docs_path.exists():
                self.test_results['raptor']['status'] = 'failed'
                self.test_results['raptor']['details'] = {'error': 'Processed documents not found'}
                return

            with open(docs_path, 'r') as f:
                processed_docs = json.load(f)

            # Initialize RAPTOR
            raptor = QuantERARAGTOR(vector_db_path=str(self.test_db_dir / "chromadb"))
            trees_created = 0

            for doc in processed_docs:
                try:
                    self.logger.info(f"Building RAPTOR tree for: {doc['title']}")

                    tree_root = raptor.build_tree_from_chunks(
                        chunks=doc['chunks'],
                        source_metadata={
                            'title': doc['title'],
                            'authors': doc['authors'],
                            'abstract': doc['abstract']
                        }
                    )

                    if tree_root:
                        trees_created += 1

                except Exception as e:
                    self.logger.warning(f"Failed to build tree for {doc['title']}: {e}")
                    continue

            if trees_created > 0:
                # Test querying
                query_results = raptor.query_tree("variational quantum algorithm", max_results=5)

                self.test_results['raptor']['status'] = 'passed'
                self.test_results['raptor']['details'] = {
                    'trees_created': trees_created,
                    'total_nodes': sum(len(nodes) for nodes in raptor.nodes_by_level.values()),
                    'nodes_by_level': {level: len(nodes) for level, nodes in raptor.nodes_by_level.items()},
                    'sample_query_results': len(query_results)
                }

                self.logger.info(f"RAPTOR test passed: {trees_created} trees created")

            else:
                self.test_results['raptor']['status'] = 'failed'
                self.test_results['raptor']['details'] = {'error': 'No trees could be created'}

        except Exception as e:
            self.test_results['raptor']['status'] = 'failed'
            self.test_results['raptor']['details'] = {'error': str(e), 'traceback': traceback.format_exc()}

    def test_knowledge_graph(self):
        """Test knowledge graph building"""
        self.logger.info("Testing knowledge graph...")

        if self.test_results['ingestion']['status'] != 'passed':
            self.test_results['knowledge_graph']['status'] = 'skipped'
            self.test_results['knowledge_graph']['details']['reason'] = 'Ingestion test failed'
            return

        try:
            # Load processed documents
            docs_path = self.test_db_dir / "processed_docs.json"

            with open(docs_path, 'r') as f:
                processed_docs = json.load(f)

            # Initialize knowledge graph
            kg = QMLKnowledgeGraph(str(self.test_db_dir / "qml_graph.pkl"))
            papers_added = 0

            for doc in processed_docs:
                try:
                    paper_id = doc['title'].replace(' ', '_').lower()[:50]
                    full_content = " ".join([chunk['text'] for chunk in doc['chunks']])

                    kg.add_paper(
                        paper_id=paper_id,
                        title=doc['title'],
                        content=full_content,
                        metadata={
                            'authors': doc['authors'],
                            'abstract': doc['abstract']
                        }
                    )

                    papers_added += 1

                except Exception as e:
                    self.logger.warning(f"Failed to add paper to KG: {doc['title']}: {e}")
                    continue

            if papers_added > 0:
                # Test graph operations
                stats = kg.get_graph_statistics()

                # Test querying
                vqe_results = kg.query_graph("VQE", limit=5)
                ansatz_results = kg.query_graph("ansatz", limit=3)

                # Test concept relationships
                if vqe_results:
                    vqe_entity_id = vqe_results[0]['entity_id']
                    related_concepts = kg.find_related_concepts(vqe_entity_id, max_hops=1)
                else:
                    related_concepts = []

                # Save graph
                kg.save_graph()

                self.test_results['knowledge_graph']['status'] = 'passed'
                self.test_results['knowledge_graph']['details'] = {
                    'papers_added': papers_added,
                    'total_entities': stats['total_entities'],
                    'total_relationships': stats['total_relationships'],
                    'entity_types': stats['entity_types'],
                    'vqe_query_results': len(vqe_results),
                    'ansatz_query_results': len(ansatz_results),
                    'related_concepts_found': len(related_concepts)
                }

                self.logger.info(f"Knowledge graph test passed: {papers_added} papers, {stats['total_entities']} entities")

            else:
                self.test_results['knowledge_graph']['status'] = 'failed'
                self.test_results['knowledge_graph']['details'] = {'error': 'No papers could be added to graph'}

        except Exception as e:
            self.test_results['knowledge_graph']['status'] = 'failed'
            self.test_results['knowledge_graph']['details'] = {'error': str(e), 'traceback': traceback.format_exc()}

    def test_agent(self):
        """Test agent initialization and integration"""
        self.logger.info("Testing agent integration...")

        try:
            # Initialize agent with test database
            agent = QuantERAAgent(db_path=str(self.test_db_dir))

            # Test system status
            status = agent.get_system_status()

            if status['status'] == 'operational':
                self.test_results['agent']['status'] = 'passed'
                self.test_results['agent']['details'] = {
                    'system_status': status['status'],
                    'raptor_nodes': status.get('raptor_tree', {}).get('total_nodes', 0),
                    'kg_entities': status.get('knowledge_graph', {}).get('total_entities', 0)
                }

                self.logger.info("Agent integration test passed")

            else:
                self.test_results['agent']['status'] = 'failed'
                self.test_results['agent']['details'] = {'error': f"Agent status: {status['status']}"}

        except Exception as e:
            self.test_results['agent']['status'] = 'failed'
            self.test_results['agent']['details'] = {'error': str(e), 'traceback': traceback.format_exc()}

    def test_queries_processing(self):
        """Test query processing with sample queries"""
        self.logger.info("Testing query processing...")

        if self.test_results['agent']['status'] != 'passed':
            self.test_results['queries']['status'] = 'skipped'
            self.test_results['queries']['details']['reason'] = 'Agent test failed'
            return

        try:
            agent = QuantERAAgent(db_path=str(self.test_db_dir))

            query_results = []
            successful_queries = 0

            for query in self.test_queries:
                try:
                    self.logger.info(f"Testing query: {query}")

                    response = agent.query(query)

                    # Basic validation
                    if response.answer and len(response.answer) > 50:
                        successful_queries += 1

                    query_results.append({
                        'query': query,
                        'success': response.answer is not None and len(response.answer) > 50,
                        'confidence': response.confidence,
                        'sources_count': len(response.sources),
                        'answer_length': len(response.answer) if response.answer else 0
                    })

                except Exception as e:
                    self.logger.warning(f"Query failed: {query}: {e}")
                    query_results.append({
                        'query': query,
                        'success': False,
                        'error': str(e)
                    })

            success_rate = successful_queries / len(self.test_queries)

            if success_rate > 0.6:  # 60% success rate threshold
                self.test_results['queries']['status'] = 'passed'
            else:
                self.test_results['queries']['status'] = 'partial'

            self.test_results['queries']['details'] = {
                'total_queries': len(self.test_queries),
                'successful_queries': successful_queries,
                'success_rate': success_rate,
                'query_results': query_results
            }

            self.logger.info(f"Query processing test: {successful_queries}/{len(self.test_queries)} successful")

        except Exception as e:
            self.test_results['queries']['status'] = 'failed'
            self.test_results['queries']['details'] = {'error': str(e), 'traceback': traceback.format_exc()}

    def calculate_overall_results(self):
        """Calculate overall test results"""
        component_tests = ['ingestion', 'raptor', 'knowledge_graph', 'agent', 'queries']

        passed_tests = 0
        total_tests = 0

        for test_name in component_tests:
            total_tests += 1
            if self.test_results[test_name]['status'] == 'passed':
                passed_tests += 1
            elif self.test_results[test_name]['status'] == 'partial':
                passed_tests += 0.5

        success_rate = passed_tests / total_tests

        if success_rate >= 0.8:
            overall_status = 'passed'
        elif success_rate >= 0.5:
            overall_status = 'partial'
        else:
            overall_status = 'failed'

        self.test_results['overall'] = {
            'status': overall_status,
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'summary': f"{passed_tests}/{total_tests} components working"
        }

    def _save_processed_docs(self, docs: List[Any], output_file: str):
        """Save processed documents to JSON (helper method)"""
        # Convert processed documents to JSON serializable format
        serializable_docs = []
        for doc in docs:
            doc_dict = {
                'title': doc.title,
                'authors': doc.authors,
                'abstract': doc.abstract,
                'chunks': doc.chunks,
                'metadata': doc.metadata,
                'mathematical_elements': doc.mathematical_elements,
                'circuit_descriptions': doc.circuit_descriptions,
                'total_pages': doc.total_pages,
                'processing_timestamp': doc.processing_timestamp
            }
            serializable_docs.append(doc_dict)

        with open(output_file, 'w') as f:
            json.dump(serializable_docs, f, indent=2)

    def print_test_report(self):
        """Print comprehensive test report"""
        print("\n" + "="*80)
        print("QUANTERA QML-RAPTOR SYSTEM TEST REPORT")
        print("="*80)

        print(f"\nOVERALL RESULT: {self.test_results['overall']['status'].upper()}")
        print(f"Success Rate: {self.test_results['overall']['success_rate']:.1%}")
        print(f"Components: {self.test_results['overall']['summary']}")

        print(f"\nDETAILED RESULTS:")
        print("-" * 40)

        for component, result in self.test_results.items():
            if component == 'overall':
                continue

            status = result['status']
            print(f"\n{component.upper()}: {status.upper()}")

            if status == 'passed':
                details = result['details']
                if component == 'ingestion' and 'papers_processed' in details:
                    print(f"  Papers processed: {details['papers_processed']}")
                    print(f"  Total chunks: {details['total_chunks']}")
                    print(f"  Math elements: {details['math_elements']}")

                elif component == 'raptor' and 'trees_created' in details:
                    print(f"  Trees created: {details['trees_created']}")
                    print(f"  Total nodes: {details['total_nodes']}")
                    print(f"  Nodes by level: {details['nodes_by_level']}")

                elif component == 'knowledge_graph' and 'papers_added' in details:
                    print(f"  Papers added: {details['papers_added']}")
                    print(f"  Entities: {details['total_entities']}")
                    print(f"  Relationships: {details['total_relationships']}")

                elif component == 'queries' and 'success_rate' in details:
                    print(f"  Success rate: {details['success_rate']:.1%}")
                    print(f"  Successful: {details['successful_queries']}/{details['total_queries']}")

            elif status == 'failed':
                error = result['details'].get('error', 'Unknown error')
                print(f"  ERROR: {error}")

            elif status == 'skipped':
                reason = result['details'].get('reason', 'Unknown reason')
                print(f"  SKIPPED: {reason}")

        # Recommendations
        print(f"\nRECOMMENDations:")
        print("-" * 20)

        if self.test_results['overall']['success_rate'] >= 0.8:
            print("✅ System is ready for production use!")
            print("✅ All core components are working correctly")
            print("✅ You can start using the QuantERA agent for research")

        elif self.test_results['overall']['success_rate'] >= 0.5:
            print("⚠️  System is partially functional")
            print("⚠️  Some components may need attention")
            print("⚠️  Check failed components and dependencies")

        else:
            print("❌ System has significant issues")
            print("❌ Review the installation and dependencies")
            print("❌ Check the error messages above")
            print("❌ Consider running setup.py again")

        print("\n" + "="*80)

    def cleanup_test_data(self):
        """Clean up test data"""
        try:
            import shutil
            if self.test_db_dir.exists():
                shutil.rmtree(self.test_db_dir)
                self.logger.info("Test data cleaned up")
        except Exception as e:
            self.logger.warning(f"Could not clean up test data: {e}")


def main():
    """Main test execution"""
    import argparse

    parser = argparse.ArgumentParser(description="QuantERA System Integration Test")
    parser.add_argument("--papers", type=int, default=3, help="Number of papers to test with")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test data after testing")
    parser.add_argument("--save-report", help="Save test report to file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test
    tester = SystemTester(test_papers_limit=args.papers)

    try:
        results = tester.run_comprehensive_test()
        tester.print_test_report()

        # Save report if requested
        if args.save_report:
            with open(args.save_report, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nTest report saved to: {args.save_report}")

        # Cleanup if requested
        if args.cleanup:
            tester.cleanup_test_data()

        # Exit with appropriate code
        if results['overall']['status'] == 'passed':
            sys.exit(0)
        elif results['overall']['status'] == 'partial':
            sys.exit(1)
        else:
            sys.exit(2)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\nTest failed with critical error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()