"""GraphRAG module for entity and relationship extraction.

Implements Phase 3: Knowledge Graph construction from scientific papers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re


@dataclass
class Entity:
    """Knowledge graph entity."""
    entity_id: str
    name: str
    type: str  # "author", "concept", "method", "institution"
    metadata: Dict = field(default_factory=dict)


@dataclass
class Relationship:
    """Knowledge graph relationship."""
    source: str  # Entity ID or paper ID
    target: str  # Entity ID
    type: str  # "cites", "uses_method", "authored_by", "extends"
    metadata: Dict = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure."""
    nodes: Dict[str, Entity] = field(default_factory=dict)
    edges: List[Relationship] = field(default_factory=list)
    
    def add_node(self, entity: Entity):
        """Add entity node."""
        self.nodes[entity.entity_id] = entity
    
    def add_edge(self, relationship: Relationship):
        """Add relationship edge."""
        self.edges.append(relationship)


class GraphRAGExtractor:
    """Extract entities and relationships for GraphRAG."""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
    
    async def extract_entities(
        self, 
        text: str, 
        entity_type: str = "all"
    ) -> List[Dict]:
        """Extract entities from text.
        
        Args:
            text: Input text
            entity_type: "author", "concept", "method", or "all"
            
        Returns:
            List of entity dicts
        """
        entities = []
        
        if entity_type in ["author", "all"]:
            entities.extend(self._extract_authors(text))
        
        if entity_type in ["concept", "all"]:
            entities.extend(self._extract_concepts(text))
        
        if entity_type in ["method", "all"]:
            entities.extend(self._extract_methods(text))
        
        return entities
    
    def _extract_authors(self, text: str) -> List[Dict]:
        """Extract author names using patterns."""
        authors = []
        
        # Pattern: "by [Name], [Name], and [Name]"
        pattern1 = r'by ([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)'
        matches = re.findall(pattern1, text)
        
        for match in matches:
            names = re.split(r',\s*(?:and\s+)?', match)
            for name in names:
                name = name.strip()
                if name:
                    authors.append({
                        "name": name,
                        "type": "author",
                        "entity_id": f"author_{name.replace(' ', '_').lower()}"
                    })
        
        # Pattern: "[Name] et al."
        pattern2 = r'([A-Z][a-z]+)\s+et\s+al\.'
        matches = re.findall(pattern2, text)
        for name in matches:
            authors.append({
                "name": name,
                "type": "author",
                "entity_id": f"author_{name.lower()}"
            })
        
        return authors
    
    def _extract_concepts(self, text: str) -> List[Dict]:
        """Extract key concepts using keyword matching."""
        concepts = []
        
        # Scientific concepts (simple keyword extraction)
        concept_keywords = [
            "protein structure", "neural network", "deep learning", 
            "machine learning", "gene editing", "genome sequencing",
            "attention mechanism", "multiple sequence alignment",
            "protein folding", "structural biology"
        ]
        
        text_lower = text.lower()
        for concept in concept_keywords:
            if concept in text_lower:
                concepts.append({
                    "name": concept.title(),
                    "type": "concept",
                    "entity_id": f"concept_{concept.replace(' ', '_')}"
                })
        
        return concepts
    
    def _extract_methods(self, text: str) -> List[Dict]:
        """Extract methodology mentions."""
        methods = []
        
        # Method keywords
        method_keywords = [
            "CRISPR-Cas9", "CRISPR", "whole-genome sequencing",
            "power analysis", "statistical analysis", "deep learning",
            "convolutional neural network", "gradient descent"
        ]
        
        text_lower = text.lower()
        for method in method_keywords:
            if method.lower() in text_lower:
                methods.append({
                    "name": method,
                    "type": "method",
                    "entity_id": f"method_{method.replace(' ', '_').replace('-', '_').lower()}"
                })
        
        return methods
    
    async def extract_relationships(
        self, 
        papers: List
    ) -> List[Dict]:
        """Extract relationships between papers and entities.
        
        Args:
            papers: List of GoldenReferencePaper objects
            
        Returns:
            List of relationship dicts
        """
        relationships = []
        
        for paper in papers:
            # Extract citation relationships
            if paper.introduction:
                cited_authors = self._extract_authors(paper.introduction)
                for author in cited_authors:
                    relationships.append({
                        "source": paper.paper_id,
                        "target": author["entity_id"],
                        "type": "cites",
                        "metadata": {"section": "introduction"}
                    })
            
            # Extract method usage
            if paper.methods:
                methods = self._extract_methods(paper.methods)
                for method in methods:
                    relationships.append({
                        "source": paper.paper_id,
                        "target": method["entity_id"],
                        "type": "uses_method",
                        "metadata": {"section": "methods"}
                    })
        
        return relationships


class KnowledgeGraphBuilder:
    """Build knowledge graph from papers."""
    
    def __init__(self, llm_service=None):
        self.extractor = GraphRAGExtractor(llm_service)
    
    async def build_graph(self, papers: List) -> KnowledgeGraph:
        """Build complete knowledge graph.
        
        Args:
            papers: List of GoldenReferencePaper objects
            
        Returns:
            KnowledgeGraph object
        """
        graph = KnowledgeGraph()
        
        # Add paper nodes
        for paper in papers:
            entity = Entity(
                entity_id=paper.paper_id,
                name=paper.title,
                type="paper",
                metadata={
                    "journal": paper.journal,
                    "year": paper.year,
                    "citation_count": getattr(paper, 'citation_count', 0)
                }
            )
            graph.add_node(entity)
        
        # Extract and add entity nodes
        all_entities = []
        for paper in papers:
            text = f"{paper.abstract} {paper.introduction or ''} {paper.methods or ''}"
            entities = await self.extractor.extract_entities(text, entity_type="all")
            all_entities.extend(entities)
        
        # Deduplicate entities
        unique_entities = {e["entity_id"]: e for e in all_entities}
        for entity_dict in unique_entities.values():
            entity = Entity(
                entity_id=entity_dict["entity_id"],
                name=entity_dict["name"],
                type=entity_dict["type"]
            )
            graph.add_node(entity)
        
        # Extract and add relationships
        relationships = await self.extractor.extract_relationships(papers)
        for rel_dict in relationships:
            rel = Relationship(
                source=rel_dict["source"],
                target=rel_dict["target"],
                type=rel_dict["type"],
                metadata=rel_dict.get("metadata", {})
            )
            graph.add_edge(rel)
        
        return graph
    
    def detect_communities(self, graph: KnowledgeGraph) -> List[Set[str]]:
        """Detect communities in the graph using simple clustering.
        
        Args:
            graph: KnowledgeGraph object
            
        Returns:
            List of communities (sets of node IDs)
        """
        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in graph.edges:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        
        # Simple connected components
        visited = set()
        communities = []
        
        for node_id in graph.nodes:
            if node_id not in visited:
                community = self._dfs(node_id, adjacency, visited)
                if community:
                    communities.append(community)
        
        return communities
    
    def _dfs(
        self, 
        node: str, 
        adjacency: Dict[str, Set[str]], 
        visited: Set[str]
    ) -> Set[str]:
        """Depth-first search for connected components."""
        if node in visited:
            return set()
        
        visited.add(node)
        community = {node}
        
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                community.update(self._dfs(neighbor, adjacency, visited))
        
        return community
