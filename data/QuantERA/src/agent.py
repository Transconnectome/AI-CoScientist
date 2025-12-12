"""
QuantERA QML-RAPTOR: Agentic Interface
Provides intelligent research assistance with autonomous reasoning
Integrates RAPTOR tree structure and knowledge graph for comprehensive QML research
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from collections import defaultdict

# Import our modules
from .ingest import QuantERAIngestor
from .raptor import QuantERARAGTOR
from .graph import QMLKnowledgeGraph


@dataclass
class QueryDecomposition:
    """Represents a decomposed research query"""
    main_query: str
    sub_queries: List[str]
    query_type: str  # concept, comparison, methodology, experimental
    entities_mentioned: List[str]
    complexity_score: float


@dataclass
class RetrievalResult:
    """Result from retrieval across multiple sources"""
    source_type: str  # raptor_l0, raptor_l1, raptor_l2, knowledge_graph
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    citations: List[str]


@dataclass
class ResearchResponse:
    """Comprehensive response to research query"""
    query: str
    answer: str
    confidence: float
    sources: List[RetrievalResult]
    reasoning_steps: List[str]
    limitations: List[str]
    follow_up_suggestions: List[str]
    timestamp: str


class QueryAnalyzer:
    """Analyzes and decomposes research queries"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Query patterns for different types
        self.query_patterns = {
            'concept': [
                r'what is\s+(.+?)\?',
                r'define\s+(.+)',
                r'explain\s+(.+)',
                r'describe\s+(.+)'
            ],
            'comparison': [
                r'(.+?)\s+(?:vs\.?|versus)\s+(.+?)\?',
                r'compare\s+(.+?)\s+(?:and|with)\s+(.+)',
                r'difference between\s+(.+?)\s+and\s+(.+)',
                r'which is better\s+(.+?)\s+or\s+(.+)'
            ],
            'methodology': [
                r'how to\s+(.+?)\?',
                r'how does\s+(.+?)\s+work',
                r'what (?:are the )?steps.+?to\s+(.+)',
                r'procedure for\s+(.+)'
            ],
            'experimental': [
                r'what (?:are the )?results.+?of\s+(.+)',
                r'performance of\s+(.+)',
                r'benchmark.+?of\s+(.+)',
                r'experimental.+?(?:results|evaluation).+?of\s+(.+)'
            ],
            'survey': [
                r'review of\s+(.+)',
                r'survey of\s+(.+)',
                r'state of art.+?in\s+(.+)',
                r'recent (?:advances|progress).+?in\s+(.+)'
            ]
        }

        # QML entity patterns for extraction
        self.qml_entities = [
            'vqe', 'qaoa', 'qnn', 'qgan', 'ansatz', 'barren plateau',
            'quantum advantage', 'nisq', 'variational', 'parameterized',
            'circuit', 'optimization', 'gradient', 'fidelity'
        ]

    def analyze_query(self, query: str) -> QueryDecomposition:
        """Analyze and decompose a research query"""
        query_lower = query.lower()

        # Determine query type
        query_type = 'concept'  # default
        for q_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    query_type = q_type
                    break
            if query_type != 'concept':
                break

        # Extract mentioned entities
        entities = []
        for entity in self.qml_entities:
            if entity in query_lower:
                entities.append(entity)

        # Generate sub-queries based on type
        sub_queries = self._generate_sub_queries(query, query_type, entities)

        # Calculate complexity score
        complexity_score = self._calculate_complexity(query, query_type, len(entities))

        return QueryDecomposition(
            main_query=query,
            sub_queries=sub_queries,
            query_type=query_type,
            entities_mentioned=entities,
            complexity_score=complexity_score
        )

    def _generate_sub_queries(self, query: str, query_type: str, entities: List[str]) -> List[str]:
        """Generate sub-queries based on main query"""
        sub_queries = []

        if query_type == 'comparison':
            # For comparisons, break into individual concept queries
            comp_match = re.search(r'(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)', query.lower())
            if comp_match:
                concept1, concept2 = comp_match.groups()
                sub_queries = [
                    f"What is {concept1.strip()}?",
                    f"What is {concept2.strip()}?",
                    f"Applications of {concept1.strip()}",
                    f"Applications of {concept2.strip()}",
                    f"Advantages and disadvantages of {concept1.strip()}",
                    f"Advantages and disadvantages of {concept2.strip()}"
                ]

        elif query_type == 'methodology':
            # For methodology, break into steps and requirements
            sub_queries = [
                f"Theory behind {' '.join(entities[:2])}",
                f"Implementation requirements for {' '.join(entities[:2])}",
                f"Common challenges in {' '.join(entities[:2])}",
                f"Best practices for {' '.join(entities[:2])}"
            ]

        elif query_type == 'experimental':
            # For experimental, focus on results and benchmarks
            sub_queries = [
                f"Performance metrics for {' '.join(entities[:2])}",
                f"Benchmark results of {' '.join(entities[:2])}",
                f"Experimental setups for {' '.join(entities[:2])}",
                f"Comparison with classical methods"
            ]

        elif query_type == 'survey':
            # For surveys, broad exploration
            sub_queries = [
                f"Recent developments in {' '.join(entities[:2])}",
                f"Key algorithms in {' '.join(entities[:2])}",
                f"Applications of {' '.join(entities[:2])}",
                f"Future directions in {' '.join(entities[:2])}"
            ]

        else:  # concept
            # For concept queries, explore related aspects
            if entities:
                main_entity = entities[0]
                sub_queries = [
                    f"Definition of {main_entity}",
                    f"Applications of {main_entity}",
                    f"Related concepts to {main_entity}",
                    f"Challenges with {main_entity}"
                ]

        # Filter out empty/invalid sub-queries
        sub_queries = [sq.strip() for sq in sub_queries if sq.strip()]
        return sub_queries[:5]  # Limit to 5 sub-queries

    def _calculate_complexity(self, query: str, query_type: str, num_entities: int) -> float:
        """Calculate query complexity score (0.0 to 1.0)"""
        base_complexity = {
            'concept': 0.2,
            'comparison': 0.6,
            'methodology': 0.7,
            'experimental': 0.8,
            'survey': 0.9
        }.get(query_type, 0.5)

        # Adjust based on number of entities and query length
        entity_factor = min(num_entities * 0.1, 0.3)
        length_factor = min(len(query.split()) * 0.02, 0.2)

        complexity = base_complexity + entity_factor + length_factor
        return min(complexity, 1.0)


class MultiSourceRetriever:
    """Retrieves information from multiple sources (RAPTOR + Knowledge Graph)"""

    def __init__(self, raptor: QuantERARAGTOR, knowledge_graph: QMLKnowledgeGraph):
        self.raptor = raptor
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(__name__)

    def retrieve(self, query: str, max_results: int = 15) -> List[RetrievalResult]:
        """Retrieve information from all sources"""
        results = []

        # 1. Query RAPTOR tree (all levels)
        raptor_results = self.raptor.query_tree(query, max_results=10)
        for result in raptor_results:
            retrieval_result = RetrievalResult(
                source_type=f"raptor_l{result['level']}",
                content=result['content'],
                metadata=result['metadata'],
                relevance_score=1.0 - (result['distance'] if result.get('distance') else 0.5),
                citations=[result['metadata'].get('source_file', 'Unknown')]
            )
            results.append(retrieval_result)

        # 2. Query Knowledge Graph
        kg_results = self.knowledge_graph.query_graph(query, limit=8)
        for result in kg_results:
            entity_stats = self.knowledge_graph.get_entity_statistics(result['entity_id'])

            # Get related concepts for richer context
            related = self.knowledge_graph.find_related_concepts(
                result['entity_id'], max_hops=1
            )[:3]

            content = f"Entity: {result['name']} ({result['type']})\n"
            if entity_stats:
                content += f"Frequency: {entity_stats['frequency']}, Papers: {entity_stats['paper_count']}\n"
                if related:
                    content += f"Related concepts: {', '.join([r['concept_name'] for r in related])}\n"

            retrieval_result = RetrievalResult(
                source_type="knowledge_graph",
                content=content,
                metadata={
                    'entity_type': result['type'],
                    'frequency': result['frequency'],
                    'related_concepts': [r['concept_name'] for r in related]
                },
                relevance_score=result['score'],
                citations=entity_stats.get('papers', []) if entity_stats else []
            )
            results.append(retrieval_result)

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]


class ResponseGenerator:
    """Generates comprehensive research responses"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_response(self, query_decomp: QueryDecomposition,
                         retrieval_results: List[RetrievalResult]) -> ResearchResponse:
        """Generate comprehensive research response"""

        # Organize results by source type for better synthesis
        results_by_source = defaultdict(list)
        for result in retrieval_results[:12]:  # Use top 12 results
            results_by_source[result.source_type].append(result)

        # Generate reasoning steps
        reasoning_steps = self._generate_reasoning_steps(query_decomp, results_by_source)

        # Synthesize answer
        answer = self._synthesize_answer(query_decomp, results_by_source)

        # Calculate confidence
        confidence = self._calculate_confidence(query_decomp, retrieval_results)

        # Identify limitations
        limitations = self._identify_limitations(query_decomp, retrieval_results)

        # Generate follow-up suggestions
        follow_ups = self._generate_follow_ups(query_decomp, retrieval_results)

        return ResearchResponse(
            query=query_decomp.main_query,
            answer=answer,
            confidence=confidence,
            sources=retrieval_results[:10],  # Include top 10 sources
            reasoning_steps=reasoning_steps,
            limitations=limitations,
            follow_up_suggestions=follow_ups,
            timestamp=datetime.now().isoformat()
        )

    def _generate_reasoning_steps(self, query_decomp: QueryDecomposition,
                                 results_by_source: Dict[str, List[RetrievalResult]]) -> List[str]:
        """Generate reasoning steps for the response"""
        steps = []

        steps.append(f"1. Query Analysis: Identified as {query_decomp.query_type} query with "
                    f"complexity score {query_decomp.complexity_score:.2f}")

        if query_decomp.entities_mentioned:
            steps.append(f"2. Entity Recognition: Found entities: {', '.join(query_decomp.entities_mentioned)}")

        raptor_count = len(results_by_source.get('raptor_l0', [])) + \
                      len(results_by_source.get('raptor_l1', [])) + \
                      len(results_by_source.get('raptor_l2', []))

        kg_count = len(results_by_source.get('knowledge_graph', []))

        if raptor_count > 0:
            steps.append(f"3. Document Analysis: Retrieved {raptor_count} relevant passages from paper collection")

        if kg_count > 0:
            steps.append(f"4. Concept Analysis: Found {kg_count} related concepts in knowledge graph")

        steps.append("5. Information Synthesis: Combined evidence from multiple sources to form coherent response")

        return steps

    def _synthesize_answer(self, query_decomp: QueryDecomposition,
                          results_by_source: Dict[str, List[RetrievalResult]]) -> str:
        """Synthesize answer from retrieval results"""
        answer_parts = []

        # Start with a direct answer based on query type
        if query_decomp.query_type == 'concept':
            answer_parts.append(self._answer_concept_query(query_decomp, results_by_source))

        elif query_decomp.query_type == 'comparison':
            answer_parts.append(self._answer_comparison_query(query_decomp, results_by_source))

        elif query_decomp.query_type == 'methodology':
            answer_parts.append(self._answer_methodology_query(query_decomp, results_by_source))

        elif query_decomp.query_type == 'experimental':
            answer_parts.append(self._answer_experimental_query(query_decomp, results_by_source))

        else:
            answer_parts.append(self._answer_general_query(query_decomp, results_by_source))

        # Add supporting details from knowledge graph
        kg_results = results_by_source.get('knowledge_graph', [])
        if kg_results:
            related_concepts = []
            for result in kg_results[:3]:
                related = result.metadata.get('related_concepts', [])
                related_concepts.extend(related[:2])

            if related_concepts:
                unique_related = list(set(related_concepts))[:5]
                answer_parts.append(f"\n\nRelated concepts include: {', '.join(unique_related)}")

        return "\n".join(answer_parts)

    def _answer_concept_query(self, query_decomp: QueryDecomposition,
                             results_by_source: Dict[str, List[RetrievalResult]]) -> str:
        """Answer concept-based queries"""
        # Look for high-level summaries first
        l2_results = results_by_source.get('raptor_l2', [])
        l1_results = results_by_source.get('raptor_l1', [])

        if l2_results:
            # Use global summary
            return f"Based on the research literature, {l2_results[0].content[:500]}..."
        elif l1_results:
            # Use thematic summary
            return f"According to the literature, {l1_results[0].content[:500]}..."
        else:
            # Fallback to knowledge graph
            kg_results = results_by_source.get('knowledge_graph', [])
            if kg_results:
                return f"From the concept analysis: {kg_results[0].content[:500]}..."
            else:
                return "Insufficient information available to provide a comprehensive answer."

    def _answer_comparison_query(self, query_decomp: QueryDecomposition,
                                results_by_source: Dict[str, List[RetrievalResult]]) -> str:
        """Answer comparison queries"""
        entities = query_decomp.entities_mentioned
        if len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]

            # Look for comparative information
            answer = f"Comparing {entity1} and {entity2}:\n\n"

            # Try to find information about each entity
            l1_results = results_by_source.get('raptor_l1', [])
            relevant_results = [r for r in l1_results if entity1 in r.content.lower() or entity2 in r.content.lower()]

            if relevant_results:
                answer += relevant_results[0].content[:400] + "..."
            else:
                answer += "Both approaches have distinct characteristics and applications in quantum machine learning."

            return answer

        return "Comparison requires identification of specific entities to compare."

    def _answer_methodology_query(self, query_decomp: QueryDecomposition,
                                 results_by_source: Dict[str, List[RetrievalResult]]) -> str:
        """Answer methodology queries"""
        # Focus on detailed implementation information
        l0_results = results_by_source.get('raptor_l0', [])
        l1_results = results_by_source.get('raptor_l1', [])

        methodology_results = []

        # Look for results with methodology keywords
        method_keywords = ['algorithm', 'method', 'approach', 'implementation', 'procedure', 'steps']

        for result_list in [l1_results, l0_results]:
            for result in result_list:
                if any(keyword in result.content.lower() for keyword in method_keywords):
                    methodology_results.append(result)

        if methodology_results:
            return f"Methodology overview:\n\n{methodology_results[0].content[:500]}..."
        else:
            return "Specific methodology details are not available in the current knowledge base."

    def _answer_experimental_query(self, query_decomp: QueryDecomposition,
                                  results_by_source: Dict[str, List[RetrievalResult]]) -> str:
        """Answer experimental/results queries"""
        # Look for results with experimental keywords
        exp_keywords = ['result', 'performance', 'benchmark', 'experiment', 'evaluation', 'comparison']

        all_results = []
        for source_results in results_by_source.values():
            all_results.extend(source_results)

        experimental_results = []
        for result in all_results:
            if any(keyword in result.content.lower() for keyword in exp_keywords):
                experimental_results.append(result)

        if experimental_results:
            return f"Experimental findings:\n\n{experimental_results[0].content[:500]}..."
        else:
            return "Specific experimental results are not available in the current knowledge base."

    def _answer_general_query(self, query_decomp: QueryDecomposition,
                             results_by_source: Dict[str, List[RetrievalResult]]) -> str:
        """Answer general queries"""
        # Use the best available result
        all_results = []
        for source_results in results_by_source.values():
            all_results.extend(source_results)

        if all_results:
            best_result = max(all_results, key=lambda x: x.relevance_score)
            return f"Based on available research:\n\n{best_result.content[:500]}..."
        else:
            return "I don't have sufficient information to answer this query comprehensively."

    def _calculate_confidence(self, query_decomp: QueryDecomposition,
                             retrieval_results: List[RetrievalResult]) -> float:
        """Calculate confidence score for the response"""
        if not retrieval_results:
            return 0.1

        # Base confidence from result scores
        avg_relevance = sum(r.relevance_score for r in retrieval_results[:5]) / min(5, len(retrieval_results))

        # Adjust for query complexity
        complexity_penalty = query_decomp.complexity_score * 0.3

        # Adjust for source diversity
        source_types = set(r.source_type for r in retrieval_results[:10])
        diversity_bonus = len(source_types) * 0.1

        confidence = avg_relevance - complexity_penalty + diversity_bonus
        return max(0.1, min(0.95, confidence))

    def _identify_limitations(self, query_decomp: QueryDecomposition,
                            retrieval_results: List[RetrievalResult]) -> List[str]:
        """Identify limitations in the response"""
        limitations = []

        if len(retrieval_results) < 5:
            limitations.append("Limited source material available for comprehensive analysis")

        if query_decomp.complexity_score > 0.7:
            limitations.append("High query complexity may require domain expert interpretation")

        source_types = set(r.source_type for r in retrieval_results)
        if 'knowledge_graph' not in source_types:
            limitations.append("Concept relationship analysis not available")

        if not any('raptor_l2' in r.source_type for r in retrieval_results):
            limitations.append("High-level document summaries not available")

        return limitations

    def _generate_follow_ups(self, query_decomp: QueryDecomposition,
                           retrieval_results: List[RetrievalResult]) -> List[str]:
        """Generate follow-up suggestions"""
        follow_ups = []

        # Based on query type
        if query_decomp.query_type == 'concept':
            follow_ups.extend([
                "What are the practical applications of this concept?",
                "What are the current limitations and challenges?",
                "How does this relate to other quantum ML approaches?"
            ])

        elif query_decomp.query_type == 'comparison':
            follow_ups.extend([
                "What are the specific use cases for each approach?",
                "Which approach performs better under different conditions?",
                "Are there hybrid approaches that combine both methods?"
            ])

        # Based on entities mentioned
        if query_decomp.entities_mentioned:
            main_entity = query_decomp.entities_mentioned[0]
            follow_ups.append(f"What are recent developments in {main_entity}?")
            follow_ups.append(f"How is {main_entity} implemented in practice?")

        return follow_ups[:5]  # Limit to 5 suggestions


class ResearchSession:
    """Manages multi-step research workflows"""

    def __init__(self, topic: str, agent: 'QuantERAAgent'):
        self.topic = topic
        self.agent = agent
        self.questions = []
        self.responses = []
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def add_question(self, question: str) -> ResearchResponse:
        """Add question to research session and get response"""
        response = self.agent.query(question)
        self.questions.append(question)
        self.responses.append(response)
        return response

    def synthesize_findings(self) -> str:
        """Synthesize findings from all questions"""
        if not self.responses:
            return "No questions have been processed yet."

        synthesis_parts = [f"Research Session Summary for: {self.topic}\n"]

        # Extract key findings from each response
        key_findings = []
        all_concepts = set()

        for i, response in enumerate(self.responses, 1):
            synthesis_parts.append(f"\n{i}. {self.questions[i-1]}")
            synthesis_parts.append(f"   Answer: {response.answer[:200]}...")

            # Extract key concepts from knowledge graph results
            for source in response.sources:
                if source.source_type == 'knowledge_graph':
                    concepts = source.metadata.get('related_concepts', [])
                    all_concepts.update(concepts[:2])

        # Add concept summary
        if all_concepts:
            synthesis_parts.append(f"\nKey concepts identified: {', '.join(list(all_concepts)[:10])}")

        # Add overall confidence
        avg_confidence = sum(r.confidence for r in self.responses) / len(self.responses)
        synthesis_parts.append(f"\nOverall confidence: {avg_confidence:.2f}")

        return "\n".join(synthesis_parts)


class QuantERAAgent:
    """Main intelligent research assistant for quantum ML"""

    def __init__(self,
                 db_path: str = "db",
                 config: Optional[Dict[str, Any]] = None):

        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.ingestor = QuantERAIngestor(config)
        self.raptor = QuantERARAGTOR(
            vector_db_path=f"{db_path}/chromadb",
            embedding_model_name=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.knowledge_graph = QMLKnowledgeGraph(f"{db_path}/qml_graph.pkl")

        self.query_analyzer = QueryAnalyzer()
        self.retriever = MultiSourceRetriever(self.raptor, self.knowledge_graph)
        self.response_generator = ResponseGenerator()

        self.logger.info("QuantERA Agent initialized successfully")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'embedding_model': 'all-MiniLM-L6-v2',
            'max_iterations': 3,
            'confidence_threshold': 0.7
        }

    def query(self, question: str) -> ResearchResponse:
        """Process research query and return comprehensive response"""
        self.logger.info(f"Processing query: {question}")

        # 1. Analyze and decompose query
        query_decomp = self.query_analyzer.analyze_query(question)

        # 2. Retrieve relevant information
        retrieval_results = self.retriever.retrieve(question)

        # 3. Self-correction: Check if we have sufficient information
        if len(retrieval_results) < 3:
            self.logger.warning("Insufficient results, trying sub-queries")

            # Try sub-queries for better coverage
            additional_results = []
            for sub_query in query_decomp.sub_queries[:2]:
                sub_results = self.retriever.retrieve(sub_query, max_results=5)
                additional_results.extend(sub_results)

            # Merge and deduplicate
            all_results = retrieval_results + additional_results
            seen_content = set()
            deduplicated_results = []
            for result in all_results:
                content_key = result.content[:100]  # Use first 100 chars as key
                if content_key not in seen_content:
                    deduplicated_results.append(result)
                    seen_content.add(content_key)

            retrieval_results = deduplicated_results

        # 4. Generate response
        response = self.response_generator.generate_response(query_decomp, retrieval_results)

        self.logger.info(f"Generated response with confidence: {response.confidence:.2f}")
        return response

    def start_research_session(self, topic: str) -> ResearchSession:
        """Start a multi-step research session"""
        return ResearchSession(topic, self)

    def add_paper_to_knowledge_base(self, pdf_path: str) -> bool:
        """Add a new paper to the knowledge base"""
        try:
            # Process paper with ingestor
            processed_doc = self.ingestor.process_paper(pdf_path)

            # Add to RAPTOR tree
            tree_root = self.raptor.build_tree_from_chunks(
                chunks=processed_doc.chunks,
                source_metadata={
                    'title': processed_doc.title,
                    'authors': processed_doc.authors,
                    'source_file': pdf_path
                }
            )

            # Add to knowledge graph
            paper_id = processed_doc.title.replace(' ', '_').lower()[:50]
            full_content = " ".join([chunk['text'] for chunk in processed_doc.chunks])

            self.knowledge_graph.add_paper(
                paper_id=paper_id,
                title=processed_doc.title,
                content=full_content,
                metadata={
                    'authors': processed_doc.authors,
                    'abstract': processed_doc.abstract
                }
            )

            # Save updated knowledge graph
            self.knowledge_graph.save_graph()

            self.logger.info(f"Successfully added paper: {processed_doc.title}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add paper {pdf_path}: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        try:
            # RAPTOR statistics
            raptor_stats = {
                'total_nodes': sum(len(nodes) for nodes in self.raptor.nodes_by_level.values()),
                'nodes_by_level': {level: len(nodes) for level, nodes in self.raptor.nodes_by_level.items()}
            }

            # Knowledge graph statistics
            kg_stats = self.knowledge_graph.get_graph_statistics()

            return {
                'status': 'operational',
                'raptor_tree': raptor_stats,
                'knowledge_graph': kg_stats,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'message': str(e)}


def main():
    """CLI interface for QuantERA Agent"""
    import argparse

    parser = argparse.ArgumentParser(description="QuantERA Research Agent")
    parser.add_argument("--query", help="Research query to process")
    parser.add_argument("--session", help="Start research session on topic")
    parser.add_argument("--add-paper", help="Add paper to knowledge base")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--db-path", default="db", help="Database path")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize agent
    agent = QuantERAAgent(db_path=args.db_path)

    if args.query:
        print("Processing query...")
        response = agent.query(args.query)

        print(f"\nQuery: {response.query}")
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence:.2f}")

        if response.sources:
            print(f"\nSources ({len(response.sources)}):")
            for i, source in enumerate(response.sources[:3], 1):
                print(f"{i}. {source.source_type}: {source.content[:100]}...")

        if response.follow_up_suggestions:
            print(f"\nSuggested follow-ups:")
            for suggestion in response.follow_up_suggestions[:3]:
                print(f"- {suggestion}")

    elif args.session:
        print(f"Starting research session on: {args.session}")
        session = agent.start_research_session(args.session)

        # Interactive session
        while True:
            question = input("\nEnter question (or 'quit' to exit, 'synthesis' for summary): ")
            if question.lower() == 'quit':
                break
            elif question.lower() == 'synthesis':
                synthesis = session.synthesize_findings()
                print(f"\n{synthesis}")
            else:
                response = session.add_question(question)
                print(f"\nAnswer: {response.answer}")
                print(f"Confidence: {response.confidence:.2f}")

    elif args.add_paper:
        print(f"Adding paper: {args.add_paper}")
        success = agent.add_paper_to_knowledge_base(args.add_paper)
        print("Success!" if success else "Failed!")

    elif args.status:
        status = agent.get_system_status()
        print(f"\nSystem Status: {status['status']}")
        if 'raptor_tree' in status:
            print(f"RAPTOR nodes: {status['raptor_tree']['total_nodes']}")
        if 'knowledge_graph' in status:
            print(f"Knowledge graph entities: {status['knowledge_graph']['total_entities']}")

    else:
        print("QuantERA Research Agent")
        print("Use --query 'your question' to ask a research question")
        print("Use --session 'topic' to start an interactive research session")


if __name__ == "__main__":
    main()