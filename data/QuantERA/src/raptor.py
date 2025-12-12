"""
QuantERA QML-RAPTOR: RAPTOR Structure Implementation
Implements recursive hierarchical summarization (L0 -> L1 -> L2)
Based on RAPTOR paper: https://arxiv.org/abs/2401.18059
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import uuid

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Vector database
import chromadb
from chromadb.config import Settings


@dataclass
class RAPTORNode:
    """Represents a node in the RAPTOR tree structure"""
    node_id: str
    level: int  # 0=atomic, 1=thematic, 2=global
    content: str
    summary: str
    children: List[str]  # Child node IDs
    parent: Optional[str]  # Parent node ID
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    created_at: str


@dataclass
class ClusterInfo:
    """Information about a cluster at a specific level"""
    cluster_id: str
    level: int
    node_ids: List[str]
    centroid: List[float]
    coherence_score: float
    topic_keywords: List[str]


class QMLSummarizer:
    """Handles summarization specifically for Quantum ML content"""

    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def summarize_atomic_chunks(self, chunks: List[str]) -> str:
        """Summarize L0 chunks to create L1 thematic summary"""
        # For now, implement a simple extractive summarization
        # TODO: Replace with actual LLM calls when API keys available

        combined_text = " ".join(chunks)

        # Extract key sentences (simple heuristic)
        sentences = combined_text.split('.')
        key_sentences = []

        # Look for sentences with quantum ML keywords
        qml_keywords = [
            'variational', 'quantum', 'ansatz', 'barren plateau', 'VQE', 'QAOA',
            'parameterized', 'NISQ', 'optimization', 'gradient', 'convergence'
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length
                keyword_count = sum(1 for kw in qml_keywords if kw.lower() in sentence.lower())
                if keyword_count >= 2:  # Contains multiple QML keywords
                    key_sentences.append(sentence)

        # Take top sentences and create summary
        summary = ". ".join(key_sentences[:3]) + "."

        if not summary.strip() or summary == ".":
            # Fallback: take first substantive sentences
            substantive_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            summary = ". ".join(substantive_sentences[:2]) + "."

        return summary

    def summarize_thematic_clusters(self, summaries: List[str]) -> str:
        """Summarize L1 summaries to create L2 global summary"""
        combined_summaries = " ".join(summaries)

        # Extract main themes and findings
        sentences = combined_summaries.split('.')

        # Prioritize methodology and results sentences
        priority_patterns = [
            'propose', 'demonstrate', 'show', 'achieve', 'improve', 'reduce',
            'algorithm', 'method', 'approach', 'technique', 'framework'
        ]

        prioritized_sentences = []
        other_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                has_priority = any(pattern in sentence.lower() for pattern in priority_patterns)
                if has_priority:
                    prioritized_sentences.append(sentence)
                else:
                    other_sentences.append(sentence)

        # Combine prioritized and other sentences
        selected_sentences = prioritized_sentences[:2] + other_sentences[:1]
        global_summary = ". ".join(selected_sentences) + "."

        return global_summary

    def extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction based on frequency and QML relevance
        words = text.lower().split()

        # QML-specific important terms
        qml_terms = {
            'variational', 'quantum', 'ansatz', 'barren', 'plateau', 'vqe', 'qaoa',
            'parameterized', 'circuit', 'gate', 'qubit', 'entanglement', 'superposition',
            'interference', 'decoherence', 'nisq', 'optimization', 'gradient', 'adam',
            'convergence', 'fidelity', 'hamiltonian', 'eigenvalue', 'unitary'
        }

        # Count occurrences
        term_counts = {}
        for word in words:
            cleaned_word = word.strip('.,;:!?()[]')
            if cleaned_word in qml_terms:
                term_counts[cleaned_word] = term_counts.get(cleaned_word, 0) + 1

        # Sort by frequency and return top terms
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return [term for term, count in sorted_terms[:10]]


class RAPTORClustering:
    """Handles clustering for RAPTOR tree construction"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

    def cluster_embeddings(self, embeddings: np.ndarray, min_cluster_size: int = 2) -> List[List[int]]:
        """Cluster embeddings using adaptive clustering"""
        if len(embeddings) < min_cluster_size:
            return [list(range(len(embeddings)))]

        # Try different number of clusters
        best_clusters = None
        best_score = -1

        max_k = min(10, len(embeddings) // min_cluster_size)

        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)

                # Calculate silhouette-like score
                score = self._evaluate_clustering(embeddings, cluster_labels)

                if score > best_score:
                    best_score = score
                    best_clusters = cluster_labels

            except Exception as e:
                self.logger.warning(f"Clustering with k={k} failed: {e}")
                continue

        if best_clusters is None:
            # Fallback: create single cluster
            return [list(range(len(embeddings)))]

        # Convert to list of lists
        clusters = {}
        for idx, label in enumerate(best_clusters):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Filter out clusters that are too small
        valid_clusters = [cluster for cluster in clusters.values()
                         if len(cluster) >= min_cluster_size]

        if not valid_clusters:
            return [list(range(len(embeddings)))]

        return valid_clusters

    def _evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate clustering quality"""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1:
                return 0.0

            total_score = 0.0
            total_points = 0

            for label in unique_labels:
                cluster_points = embeddings[labels == label]
                if len(cluster_points) <= 1:
                    continue

                # Intra-cluster similarity (higher is better)
                intra_sim = np.mean(cosine_similarity(cluster_points))

                # Inter-cluster dissimilarity
                other_points = embeddings[labels != label]
                if len(other_points) > 0:
                    inter_sim = np.mean(cosine_similarity(cluster_points, other_points))
                    score = intra_sim - inter_sim
                else:
                    score = intra_sim

                total_score += score * len(cluster_points)
                total_points += len(cluster_points)

            return total_score / total_points if total_points > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Error evaluating clustering: {e}")
            return 0.0


class QuantERARAGTOR:
    """Main RAPTOR implementation for QuantERA QML papers"""

    def __init__(self,
                 vector_db_path: str = "db/chromadb",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):

        self.vector_db_path = Path(vector_db_path)
        self.embedding_model_name = embedding_model_name
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.summarizer = QMLSummarizer()
        self.clusterer = RAPTORClustering()

        # Initialize vector database
        self._initialize_vector_db()

        # Load embedding model
        self._load_embedding_model()

        # Tree structure storage
        self.tree = nx.DiGraph()
        self.nodes_by_level = {0: [], 1: [], 2: []}

    def _initialize_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.vector_db_path.mkdir(parents=True, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collections for each level
            self.collections = {}
            for level in [0, 1, 2]:
                collection_name = f"quantera_level_{level}"
                try:
                    self.collections[level] = self.chroma_client.get_collection(collection_name)
                except ValueError:
                    # Collection doesn't exist, create it
                    self.collections[level] = self.chroma_client.create_collection(collection_name)

            self.logger.info("Vector database initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            self.chroma_client = None
            self.collections = {}

    def _load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            self.logger.warning(f"Could not load embedding model: {e}")
            self.embedding_model = None

    def _generate_node_id(self, level: int, content: str) -> str:
        """Generate unique node ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"L{level}_{content_hash}_{uuid.uuid4().hex[:8]}"

    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for text"""
        if self.embedding_model is None:
            return None

        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.warning(f"Failed to create embedding: {e}")
            return None

    def build_tree_from_chunks(self, chunks: List[Dict[str, Any]],
                              source_metadata: Dict[str, Any]) -> RAPTORNode:
        """Build complete RAPTOR tree from document chunks"""
        self.logger.info(f"Building RAPTOR tree from {len(chunks)} chunks")

        # Level 0: Create atomic nodes from chunks
        l0_nodes = self._create_level_0_nodes(chunks, source_metadata)

        # Level 1: Create thematic clusters and summaries
        l1_nodes = self._create_level_1_nodes(l0_nodes)

        # Level 2: Create global summary
        l2_node = self._create_level_2_node(l1_nodes, source_metadata)

        # Store in vector database
        self._store_nodes_in_db(l0_nodes + l1_nodes + [l2_node])

        self.logger.info(f"RAPTOR tree created: {len(l0_nodes)} L0, {len(l1_nodes)} L1, 1 L2")
        return l2_node

    def _create_level_0_nodes(self, chunks: List[Dict[str, Any]],
                             source_metadata: Dict[str, Any]) -> List[RAPTORNode]:
        """Create Level 0 (atomic) nodes from chunks"""
        l0_nodes = []

        for i, chunk in enumerate(chunks):
            content = chunk['text']
            embedding = self._create_embedding(content)

            metadata = {
                'source_file': source_metadata.get('source_file', ''),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'entities': chunk.get('entities', {}),
                'has_math': chunk.get('metadata', {}).get('has_math', False),
                'has_circuits': chunk.get('metadata', {}).get('has_circuits', False)
            }

            node = RAPTORNode(
                node_id=self._generate_node_id(0, content),
                level=0,
                content=content,
                summary=content[:500] + "..." if len(content) > 500 else content,
                children=[],
                parent=None,
                embedding=embedding,
                metadata=metadata,
                created_at=datetime.now().isoformat()
            )

            l0_nodes.append(node)
            self.tree.add_node(node.node_id, **asdict(node))
            self.nodes_by_level[0].append(node.node_id)

        return l0_nodes

    def _create_level_1_nodes(self, l0_nodes: List[RAPTORNode]) -> List[RAPTORNode]:
        """Create Level 1 (thematic) nodes by clustering L0 nodes"""
        if not l0_nodes:
            return []

        # Extract embeddings
        embeddings = []
        valid_nodes = []

        for node in l0_nodes:
            if node.embedding:
                embeddings.append(node.embedding)
                valid_nodes.append(node)

        if not embeddings:
            self.logger.warning("No embeddings available for L1 clustering")
            return []

        embeddings_array = np.array(embeddings)

        # Cluster the embeddings
        clusters = self.clusterer.cluster_embeddings(embeddings_array)

        l1_nodes = []
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < 2:  # Skip single-node clusters
                continue

            # Get nodes in this cluster
            cluster_nodes = [valid_nodes[i] for i in cluster]
            cluster_contents = [node.content for node in cluster_nodes]

            # Create summary
            summary = self.summarizer.summarize_atomic_chunks(cluster_contents)

            # Extract keywords
            combined_content = " ".join(cluster_contents)
            keywords = self.summarizer.extract_keywords(combined_content)

            # Create embedding for summary
            embedding = self._create_embedding(summary)

            # Metadata for L1 node
            metadata = {
                'cluster_id': cluster_idx,
                'cluster_size': len(cluster_nodes),
                'topic_keywords': keywords,
                'source_files': list(set(node.metadata.get('source_file', '')
                                        for node in cluster_nodes)),
                'has_math_chunks': any(node.metadata.get('has_math', False)
                                      for node in cluster_nodes),
                'has_circuit_chunks': any(node.metadata.get('has_circuits', False)
                                         for node in cluster_nodes)
            }

            l1_node = RAPTORNode(
                node_id=self._generate_node_id(1, summary),
                level=1,
                content=summary,
                summary=summary[:300] + "..." if len(summary) > 300 else summary,
                children=[node.node_id for node in cluster_nodes],
                parent=None,
                embedding=embedding,
                metadata=metadata,
                created_at=datetime.now().isoformat()
            )

            # Update parent references for children
            for child_node in cluster_nodes:
                child_node.parent = l1_node.node_id
                self.tree.add_edge(l1_node.node_id, child_node.node_id)

            l1_nodes.append(l1_node)
            self.tree.add_node(l1_node.node_id, **asdict(l1_node))
            self.nodes_by_level[1].append(l1_node.node_id)

        return l1_nodes

    def _create_level_2_node(self, l1_nodes: List[RAPTORNode],
                            source_metadata: Dict[str, Any]) -> RAPTORNode:
        """Create Level 2 (global) node from L1 nodes"""
        if not l1_nodes:
            # If no L1 nodes, create L2 from source metadata
            content = f"Document: {source_metadata.get('title', 'Unknown')}"
            summary = source_metadata.get('abstract', content)
        else:
            # Create global summary from L1 summaries
            l1_summaries = [node.summary for node in l1_nodes]
            summary = self.summarizer.summarize_thematic_clusters(l1_summaries)
            content = summary

        # Extract global keywords
        all_keywords = []
        for node in l1_nodes:
            all_keywords.extend(node.metadata.get('topic_keywords', []))

        global_keywords = list(set(all_keywords))[:15]  # Top 15 unique keywords

        embedding = self._create_embedding(content)

        metadata = {
            'document_title': source_metadata.get('title', ''),
            'authors': source_metadata.get('authors', []),
            'total_l1_nodes': len(l1_nodes),
            'global_keywords': global_keywords,
            'processing_complete': True,
            'paper_type': 'quantum_ml_research'
        }

        l2_node = RAPTORNode(
            node_id=self._generate_node_id(2, content),
            level=2,
            content=content,
            summary=summary,
            children=[node.node_id for node in l1_nodes],
            parent=None,
            embedding=embedding,
            metadata=metadata,
            created_at=datetime.now().isoformat()
        )

        # Update parent references for L1 children
        for l1_node in l1_nodes:
            l1_node.parent = l2_node.node_id
            self.tree.add_edge(l2_node.node_id, l1_node.node_id)

        self.tree.add_node(l2_node.node_id, **asdict(l2_node))
        self.nodes_by_level[2].append(l2_node.node_id)

        return l2_node

    def _store_nodes_in_db(self, nodes: List[RAPTORNode]):
        """Store nodes in vector database"""
        if not self.collections:
            self.logger.warning("No vector database available for storage")
            return

        nodes_by_level = {0: [], 1: [], 2: []}
        for node in nodes:
            nodes_by_level[node.level].append(node)

        for level, level_nodes in nodes_by_level.items():
            if not level_nodes:
                continue

            try:
                collection = self.collections[level]

                # Prepare data for ChromaDB
                ids = [node.node_id for node in level_nodes]
                documents = [node.content for node in level_nodes]
                metadatas = [node.metadata for node in level_nodes]

                # Use embeddings if available
                embeddings = None
                if all(node.embedding for node in level_nodes):
                    embeddings = [node.embedding for node in level_nodes]

                if embeddings:
                    collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                else:
                    collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )

                self.logger.info(f"Stored {len(level_nodes)} nodes at level {level}")

            except Exception as e:
                self.logger.error(f"Failed to store level {level} nodes: {e}")

    def query_tree(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query the RAPTOR tree structure"""
        results = []

        for level in [2, 1, 0]:  # Search top-down
            if not self.collections.get(level):
                continue

            try:
                collection = self.collections[level]
                level_results = collection.query(
                    query_texts=[query],
                    n_results=min(max_results, 5)
                )

                for i, doc_id in enumerate(level_results['ids'][0]):
                    result = {
                        'node_id': doc_id,
                        'level': level,
                        'content': level_results['documents'][0][i],
                        'metadata': level_results['metadatas'][0][i],
                        'distance': level_results['distances'][0][i] if 'distances' in level_results else None
                    }
                    results.append(result)

            except Exception as e:
                self.logger.error(f"Error querying level {level}: {e}")

        return results[:max_results]

    def save_tree(self, output_path: str):
        """Save the tree structure to file"""
        tree_data = {
            'nodes_by_level': self.nodes_by_level,
            'tree_edges': list(self.tree.edges()),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'embedding_model': self.embedding_model_name,
                'total_nodes': sum(len(nodes) for nodes in self.nodes_by_level.values())
            }
        }

        with open(output_path, 'w') as f:
            json.dump(tree_data, f, indent=2)

        self.logger.info(f"Tree structure saved to {output_path}")


def main():
    """CLI interface for RAPTOR tree building"""
    import argparse

    parser = argparse.ArgumentParser(description="QuantERA RAPTOR Tree Builder")
    parser.add_argument("--input", required=True, help="Input JSON file from ingestor")
    parser.add_argument("--output", help="Output file for tree structure")
    parser.add_argument("--db-path", default="db/chromadb", help="Vector database path")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize RAPTOR system
    raptor = QuantERARAGTOR(vector_db_path=args.db_path)

    # Load processed documents
    with open(args.input, 'r') as f:
        processed_docs = json.load(f)

    # Build trees for each document
    for doc in processed_docs:
        print(f"Processing: {doc['title']}")

        tree_root = raptor.build_tree_from_chunks(
            chunks=doc['chunks'],
            source_metadata={
                'title': doc['title'],
                'authors': doc['authors'],
                'abstract': doc['abstract'],
                'source_file': doc.get('metadata', {}).get('source_file', '')
            }
        )

        print(f"Tree created with root: {tree_root.node_id}")

    # Save tree structure if requested
    if args.output:
        raptor.save_tree(args.output)


if __name__ == "__main__":
    main()