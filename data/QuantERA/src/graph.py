"""
QuantERA QML-RAPTOR: Knowledge Graph Module
Builds and manages concept relationships across papers
Implements quantum ML domain-specific entity extraction and relationship mapping
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
import pickle

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class QMLEntity:
    """Represents a Quantum ML entity/concept"""
    entity_id: str
    name: str
    entity_type: str  # concept, algorithm, hardware, metric, person, organization
    aliases: List[str]
    description: str
    frequency: int  # How often mentioned across papers
    papers: List[str]  # Paper IDs where this entity appears
    metadata: Dict[str, Any]


@dataclass
class QMLRelationship:
    """Represents a relationship between two entities"""
    relationship_id: str
    source_entity: str
    target_entity: str
    relationship_type: str  # uses, extends, mitigates, compares_to, etc.
    strength: float  # 0.0 to 1.0
    evidence: List[str]  # Text snippets supporting this relationship
    papers: List[str]  # Papers where this relationship is found
    confidence: float


class QMLEntityExtractor:
    """Extracts quantum ML specific entities from text"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Quantum ML taxonomy
        self.entity_patterns = {
            'algorithms': {
                'VQE': ['variational quantum eigensolver', 'vqe'],
                'QAOA': ['quantum approximate optimization algorithm', 'qaoa'],
                'QNN': ['quantum neural network', 'qnn', 'quantum neural networks'],
                'QGAN': ['quantum generative adversarial network', 'qgan'],
                'QML': ['quantum machine learning', 'qml'],
                'VQA': ['variational quantum algorithm', 'vqa', 'variational quantum algorithms'],
                'QSV': ['quantum support vector', 'qsv', 'quantum svm'],
                'QPR': ['quantum principal component', 'qpca', 'quantum pca'],
                'Shor': ['shor\'s algorithm', 'shor algorithm'],
                'Grover': ['grover\'s algorithm', 'grover algorithm', 'grover search'],
                'ADAPT-VQE': ['adapt-vqe', 'adaptive vqe'],
                'VQD': ['variational quantum deflation', 'vqd'],
                'VQT': ['variational quantum thermalizer', 'vqt'],
                'SPSA': ['simultaneous perturbation stochastic approximation', 'spsa'],
                'Adam': ['adam optimizer', 'adam'],
                'Natural Gradient': ['natural gradient', 'quantum natural gradient']
            },

            'concepts': {
                'Barren Plateau': ['barren plateau', 'barren plateaus', 'vanishing gradient'],
                'Ansatz': ['ansatz', 'variational ansatz', 'quantum ansatz'],
                'Parametrized Circuit': ['parametrized circuit', 'parameterized circuit', 'pqc'],
                'Quantum Advantage': ['quantum advantage', 'quantum supremacy'],
                'Expressibility': ['expressibility', 'circuit expressibility'],
                'Entangling Capability': ['entangling capability', 'entanglement generation'],
                'Gradient Vanishing': ['gradient vanishing', 'vanishing gradients'],
                'Local Minimum': ['local minimum', 'local minima', 'optimization landscape'],
                'Overparameterization': ['overparameterization', 'over-parameterization'],
                'Trainability': ['trainability', 'quantum trainability'],
                'Quantum Fisher Information': ['quantum fisher information', 'qfi'],
                'Shot Noise': ['shot noise', 'sampling noise', 'statistical noise'],
                'Coherent Error': ['coherent error', 'systematic error'],
                'Incoherent Error': ['incoherent error', 'stochastic error']
            },

            'hardware': {
                'Superconducting': ['superconducting', 'superconducting qubit', 'transmon'],
                'Ion Trap': ['ion trap', 'trapped ion', 'trapped ions'],
                'Photonic': ['photonic', 'optical', 'linear optical'],
                'Topological': ['topological', 'topological qubit', 'majorana'],
                'Neutral Atom': ['neutral atom', 'cold atom', 'rydberg'],
                'Silicon Quantum Dot': ['silicon quantum dot', 'quantum dot'],
                'NMR': ['nuclear magnetic resonance', 'nmr', 'liquid nmr'],
                'NISQ': ['nisq', 'noisy intermediate-scale quantum'],
                'IBM Quantum': ['ibm quantum', 'ibm q', 'qiskit'],
                'Google Quantum': ['google quantum', 'sycamore', 'cirq'],
                'Rigetti': ['rigetti', 'forest'],
                'IonQ': ['ionq', 'ion-q'],
                'Xanadu': ['xanadu', 'pennylane', 'strawberry fields']
            },

            'metrics': {
                'Fidelity': ['fidelity', 'state fidelity', 'process fidelity'],
                'Gate Fidelity': ['gate fidelity', 'single-qubit fidelity', 'two-qubit fidelity'],
                'Coherence Time': ['coherence time', 't1', 't2', 'decoherence time'],
                'Error Rate': ['error rate', 'gate error rate', 'readout error'],
                'Depth': ['circuit depth', 'gate depth', 'quantum depth'],
                'Width': ['circuit width', 'number of qubits', 'qubit count'],
                'Quantum Volume': ['quantum volume', 'qv'],
                'Cross Entropy Benchmarking': ['cross entropy benchmarking', 'xeb'],
                'Randomized Benchmarking': ['randomized benchmarking', 'rb'],
                'Energy Expectation': ['energy expectation', 'expectation value']
            },

            'techniques': {
                'Zero Noise Extrapolation': ['zero noise extrapolation', 'zne'],
                'Error Mitigation': ['error mitigation', 'quantum error mitigation'],
                'Readout Error Mitigation': ['readout error mitigation', 'measurement error mitigation'],
                'Symmetry Verification': ['symmetry verification', 'parity check'],
                'Clifford Data Regression': ['clifford data regression', 'cdr'],
                'Virtual Distillation': ['virtual distillation', 'virtual purification'],
                'Dynamical Decoupling': ['dynamical decoupling', 'dd'],
                'Composite Pulse': ['composite pulse', 'robust control'],
                'Optimal Control': ['optimal control', 'grape', 'krotov']
            }
        }

        # Relationship indicators
        self.relationship_patterns = {
            'uses': [
                r'(\w+)\s+uses?\s+(\w+)',
                r'(\w+)\s+employs?\s+(\w+)',
                r'(\w+)\s+utilizes?\s+(\w+)',
                r'(\w+)\s+applies?\s+(\w+)',
                r'(\w+)\s+implements?\s+(\w+)'
            ],
            'extends': [
                r'(\w+)\s+extends?\s+(\w+)',
                r'(\w+)\s+generalizes?\s+(\w+)',
                r'(\w+)\s+builds?\s+on\s+(\w+)',
                r'(\w+)\s+improves?\s+(\w+)',
                r'(\w+)\s+enhances?\s+(\w+)'
            ],
            'mitigates': [
                r'(\w+)\s+mitigates?\s+(\w+)',
                r'(\w+)\s+reduces?\s+(\w+)',
                r'(\w+)\s+suppresses?\s+(\w+)',
                r'(\w+)\s+addresses?\s+(\w+)',
                r'(\w+)\s+solves?\s+(\w+)'
            ],
            'compares_to': [
                r'(\w+)\s+(?:vs\.?|versus)\s+(\w+)',
                r'(\w+)\s+compared?\s+(?:to|with)\s+(\w+)',
                r'(\w+)\s+outperforms?\s+(\w+)',
                r'(\w+)\s+better\s+than\s+(\w+)',
                r'(\w+)\s+superior\s+to\s+(\w+)'
            ],
            'causes': [
                r'(\w+)\s+causes?\s+(\w+)',
                r'(\w+)\s+leads?\s+to\s+(\w+)',
                r'(\w+)\s+results?\s+in\s+(\w+)',
                r'(\w+)\s+induces?\s+(\w+)'
            ]
        }

    def extract_entities(self, text: str, paper_id: str = None) -> List[QMLEntity]:
        """Extract QML entities from text"""
        entities = []
        text_lower = text.lower()

        for category, entity_dict in self.entity_patterns.items():
            for canonical_name, patterns in entity_dict.items():
                found = False
                matched_aliases = []

                # Check for each pattern/alias
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        found = True
                        matched_aliases.append(pattern)

                if found:
                    # Count frequency
                    frequency = sum(text_lower.count(alias.lower()) for alias in patterns)

                    entity = QMLEntity(
                        entity_id=f"{category}_{canonical_name.lower().replace(' ', '_')}",
                        name=canonical_name,
                        entity_type=category,
                        aliases=matched_aliases,
                        description=f"{canonical_name} ({category})",
                        frequency=frequency,
                        papers=[paper_id] if paper_id else [],
                        metadata={'category': category, 'canonical_form': canonical_name}
                    )
                    entities.append(entity)

        return entities

    def extract_relationships(self, text: str, entities: List[QMLEntity]) -> List[QMLRelationship]:
        """Extract relationships between entities"""
        relationships = []

        # Create entity name lookup
        entity_names = {}
        for entity in entities:
            for alias in [entity.name] + entity.aliases:
                entity_names[alias.lower()] = entity

        # Look for relationship patterns
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_term = match.group(1).lower()
                    target_term = match.group(2).lower()

                    source_entity = entity_names.get(source_term)
                    target_entity = entity_names.get(target_term)

                    if source_entity and target_entity and source_entity != target_entity:
                        relationship = QMLRelationship(
                            relationship_id=f"{rel_type}_{source_entity.entity_id}_{target_entity.entity_id}",
                            source_entity=source_entity.entity_id,
                            target_entity=target_entity.entity_id,
                            relationship_type=rel_type,
                            strength=0.7,  # Default strength
                            evidence=[match.group(0)],
                            papers=[],
                            confidence=0.8
                        )
                        relationships.append(relationship)

        return relationships


class QMLKnowledgeGraph:
    """Main knowledge graph for Quantum ML concepts"""

    def __init__(self, graph_path: str = "db/qml_graph.pkl"):
        self.graph_path = Path(graph_path)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.extractor = QMLEntityExtractor()

        # Graph storage
        self.concept_graph = nx.DiGraph()
        self.entities = {}  # entity_id -> QMLEntity
        self.relationships = {}  # relationship_id -> QMLRelationship

        # Paper tracking
        self.papers = {}  # paper_id -> paper metadata
        self.paper_entities = defaultdict(list)  # paper_id -> [entity_ids]

        # Load existing graph if available
        self._load_graph()

    def _load_graph(self):
        """Load existing graph from file"""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, 'rb') as f:
                    data = pickle.load(f)

                self.concept_graph = data.get('graph', nx.DiGraph())
                self.entities = data.get('entities', {})
                self.relationships = data.get('relationships', {})
                self.papers = data.get('papers', {})
                self.paper_entities = data.get('paper_entities', defaultdict(list))

                self.logger.info(f"Loaded graph with {len(self.entities)} entities, "
                               f"{len(self.relationships)} relationships")

            except Exception as e:
                self.logger.warning(f"Could not load existing graph: {e}")

    def save_graph(self):
        """Save graph to file"""
        try:
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'graph': self.concept_graph,
                'entities': self.entities,
                'relationships': self.relationships,
                'papers': self.papers,
                'paper_entities': dict(self.paper_entities),
                'saved_at': datetime.now().isoformat()
            }

            with open(self.graph_path, 'wb') as f:
                pickle.dump(data, f)

            self.logger.info(f"Saved graph to {self.graph_path}")

        except Exception as e:
            self.logger.error(f"Failed to save graph: {e}")

    def add_paper(self, paper_id: str, title: str, content: str,
                  metadata: Dict[str, Any] = None):
        """Add a paper and extract its entities and relationships"""
        self.logger.info(f"Adding paper: {title}")

        # Store paper metadata
        self.papers[paper_id] = {
            'title': title,
            'content': content,
            'metadata': metadata or {},
            'added_at': datetime.now().isoformat()
        }

        # Extract entities
        extracted_entities = self.extractor.extract_entities(content, paper_id)

        # Add entities to graph
        for entity in extracted_entities:
            self._add_or_update_entity(entity, paper_id)

        # Extract relationships
        relationships = self.extractor.extract_relationships(content, extracted_entities)

        # Add relationships to graph
        for relationship in relationships:
            self._add_or_update_relationship(relationship, paper_id)

        self.logger.info(f"Added {len(extracted_entities)} entities, "
                        f"{len(relationships)} relationships for paper {paper_id}")

    def _add_or_update_entity(self, entity: QMLEntity, paper_id: str):
        """Add or update entity in the graph"""
        if entity.entity_id in self.entities:
            # Update existing entity
            existing = self.entities[entity.entity_id]
            existing.frequency += entity.frequency
            if paper_id not in existing.papers:
                existing.papers.append(paper_id)

            # Merge aliases
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)

        else:
            # Add new entity
            self.entities[entity.entity_id] = entity
            self.concept_graph.add_node(entity.entity_id, **asdict(entity))

        # Track paper-entity relationship
        if entity.entity_id not in self.paper_entities[paper_id]:
            self.paper_entities[paper_id].append(entity.entity_id)

    def _add_or_update_relationship(self, relationship: QMLRelationship, paper_id: str):
        """Add or update relationship in the graph"""
        if relationship.relationship_id in self.relationships:
            # Update existing relationship
            existing = self.relationships[relationship.relationship_id]
            existing.strength = max(existing.strength, relationship.strength)
            existing.evidence.extend(relationship.evidence)
            if paper_id not in existing.papers:
                existing.papers.append(paper_id)

        else:
            # Add new relationship
            if paper_id:
                relationship.papers = [paper_id]
            self.relationships[relationship.relationship_id] = relationship

            # Add edge to graph
            self.concept_graph.add_edge(
                relationship.source_entity,
                relationship.target_entity,
                **asdict(relationship)
            )

    def find_related_concepts(self, concept_id: str, max_hops: int = 2,
                             min_strength: float = 0.5) -> List[Dict[str, Any]]:
        """Find concepts related to given concept"""
        if concept_id not in self.concept_graph:
            return []

        related = []

        # BFS to find related concepts
        visited = set()
        queue = [(concept_id, 0, [])]  # (node, distance, path)

        while queue:
            current, distance, path = queue.pop(0)

            if current in visited or distance > max_hops:
                continue

            visited.add(current)

            # Get neighbors
            for neighbor in self.concept_graph.neighbors(current):
                if neighbor not in visited:
                    edge_data = self.concept_graph[current][neighbor]
                    strength = edge_data.get('strength', 0.5)

                    if strength >= min_strength:
                        rel_info = {
                            'concept_id': neighbor,
                            'concept_name': self.entities[neighbor].name if neighbor in self.entities else neighbor,
                            'relationship_type': edge_data.get('relationship_type', 'related'),
                            'strength': strength,
                            'distance': distance + 1,
                            'path': path + [current],
                            'evidence': edge_data.get('evidence', [])
                        }
                        related.append(rel_info)

                        if distance + 1 < max_hops:
                            queue.append((neighbor, distance + 1, path + [current]))

        # Sort by strength and distance
        related.sort(key=lambda x: (-x['strength'], x['distance']))
        return related

    def get_entity_statistics(self, entity_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an entity"""
        if entity_id not in self.entities:
            return {}

        entity = self.entities[entity_id]

        # Count relationships
        incoming_rels = len(list(self.concept_graph.predecessors(entity_id)))
        outgoing_rels = len(list(self.concept_graph.successors(entity_id)))

        # Find most related concepts
        related = self.find_related_concepts(entity_id, max_hops=1)

        stats = {
            'entity_id': entity_id,
            'name': entity.name,
            'type': entity.entity_type,
            'frequency': entity.frequency,
            'paper_count': len(entity.papers),
            'incoming_relationships': incoming_rels,
            'outgoing_relationships': outgoing_rels,
            'most_related': related[:5],
            'papers': entity.papers
        }

        return stats

    def get_concept_cooccurrence(self, concept1_id: str, concept2_id: str) -> Dict[str, Any]:
        """Analyze co-occurrence of two concepts"""
        if concept1_id not in self.entities or concept2_id not in self.entities:
            return {}

        entity1 = self.entities[concept1_id]
        entity2 = self.entities[concept2_id]

        # Find papers where both concepts appear
        common_papers = set(entity1.papers) & set(entity2.papers)

        # Calculate co-occurrence strength
        total_papers = len(self.papers)
        p1_papers = len(entity1.papers)
        p2_papers = len(entity2.papers)
        common_count = len(common_papers)

        # PMI-like score
        if p1_papers > 0 and p2_papers > 0:
            expected = (p1_papers * p2_papers) / total_papers
            pmi_score = np.log(common_count / expected) if expected > 0 else 0
        else:
            pmi_score = 0

        return {
            'concept1': entity1.name,
            'concept2': entity2.name,
            'common_papers': list(common_papers),
            'cooccurrence_count': common_count,
            'pmi_score': pmi_score,
            'jaccard_similarity': common_count / (p1_papers + p2_papers - common_count) if p1_papers + p2_papers > 0 else 0
        }

    def export_subgraph(self, entity_ids: List[str], include_neighbors: bool = True) -> Dict[str, Any]:
        """Export a subgraph for visualization"""
        if include_neighbors:
            # Add immediate neighbors
            extended_ids = set(entity_ids)
            for entity_id in entity_ids:
                if entity_id in self.concept_graph:
                    extended_ids.update(self.concept_graph.neighbors(entity_id))
                    extended_ids.update(self.concept_graph.predecessors(entity_id))
            entity_ids = list(extended_ids)

        # Extract subgraph
        subgraph = self.concept_graph.subgraph(entity_ids)

        # Format for visualization
        nodes = []
        edges = []

        for node in subgraph.nodes():
            if node in self.entities:
                entity = self.entities[node]
                nodes.append({
                    'id': node,
                    'name': entity.name,
                    'type': entity.entity_type,
                    'frequency': entity.frequency,
                    'papers': len(entity.papers)
                })

        for edge in subgraph.edges(data=True):
            source, target, data = edge
            edges.append({
                'source': source,
                'target': target,
                'type': data.get('relationship_type', 'related'),
                'strength': data.get('strength', 0.5),
                'evidence': data.get('evidence', [])[:3]  # Limit evidence
            })

        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'node_count': len(nodes),
                'edge_count': len(edges),
                'exported_at': datetime.now().isoformat()
            }
        }

    def query_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query the graph for relevant concepts"""
        query_lower = query.lower()
        results = []

        # Search entities by name and aliases
        for entity_id, entity in self.entities.items():
            score = 0

            # Exact name match
            if query_lower == entity.name.lower():
                score = 1.0
            elif query_lower in entity.name.lower():
                score = 0.8

            # Alias matches
            for alias in entity.aliases:
                if query_lower == alias.lower():
                    score = max(score, 0.9)
                elif query_lower in alias.lower():
                    score = max(score, 0.7)

            # Partial matches with frequency weighting
            if query_lower in entity.name.lower() or any(query_lower in alias.lower() for alias in entity.aliases):
                score = max(score, 0.5 * np.log(1 + entity.frequency) / 10)

            if score > 0:
                results.append({
                    'entity_id': entity_id,
                    'name': entity.name,
                    'type': entity.entity_type,
                    'score': score,
                    'frequency': entity.frequency,
                    'paper_count': len(entity.papers)
                })

        # Sort by score and frequency
        results.sort(key=lambda x: (-x['score'], -x['frequency']))
        return results[:limit]

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        entity_types = defaultdict(int)
        relationship_types = defaultdict(int)

        for entity in self.entities.values():
            entity_types[entity.entity_type] += 1

        for relationship in self.relationships.values():
            relationship_types[relationship.relationship_type] += 1

        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'total_papers': len(self.papers),
            'entity_types': dict(entity_types),
            'relationship_types': dict(relationship_types),
            'graph_density': nx.density(self.concept_graph),
            'connected_components': nx.number_weakly_connected_components(self.concept_graph)
        }


def main():
    """CLI interface for knowledge graph building"""
    import argparse

    parser = argparse.ArgumentParser(description="QuantERA Knowledge Graph Builder")
    parser.add_argument("--input", required=True, help="Input JSON file from ingestor")
    parser.add_argument("--graph-path", default="db/qml_graph.pkl", help="Graph storage path")
    parser.add_argument("--export", help="Export subgraph to JSON file")
    parser.add_argument("--query", help="Query the graph")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize knowledge graph
    kg = QMLKnowledgeGraph(graph_path=args.graph_path)

    if args.input:
        # Load and process documents
        with open(args.input, 'r') as f:
            processed_docs = json.load(f)

        for doc in processed_docs:
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

        # Save the updated graph
        kg.save_graph()

    if args.query:
        results = kg.query_graph(args.query)
        print(f"\nQuery results for '{args.query}':")
        for result in results:
            print(f"- {result['name']} ({result['type']}) - Score: {result['score']:.2f}")

    if args.stats:
        stats = kg.get_graph_statistics()
        print("\nGraph Statistics:")
        print(f"Total entities: {stats['total_entities']}")
        print(f"Total relationships: {stats['total_relationships']}")
        print(f"Total papers: {stats['total_papers']}")
        print(f"Entity types: {stats['entity_types']}")
        print(f"Relationship types: {stats['relationship_types']}")

    if args.export:
        # Export all entities for visualization
        all_entities = list(kg.entities.keys())[:50]  # Limit for visualization
        subgraph_data = kg.export_subgraph(all_entities)

        with open(args.export, 'w') as f:
            json.dump(subgraph_data, f, indent=2)

        print(f"Exported subgraph to {args.export}")


if __name__ == "__main__":
    main()