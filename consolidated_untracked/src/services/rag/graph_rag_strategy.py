"""
GraphRAG Strategy for Knowledge Graph-Enhanced Retrieval

Implementation for: GraphRAG integration with knowledge graphs
Created: 2025-12-05

Acceptance Criteria:
- Entity-centric retrieval using knowledge graphs
- Relationship-aware context expansion
- Multi-hop reasoning through graph traversal
- Integration with unified RAG orchestrator

This module provides GraphRAG capabilities that leverage knowledge graphs
to enhance retrieval through entity relationships and graph traversal.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from abc import ABC, abstractmethod

# External dependencies with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core dependencies
from datetime import datetime
from ..rag.unified_rag_orchestrator import (
    RAGStrategy, QueryContext, RAGResponse, PerformanceMetrics
)
from ..rag.knowledge_graph_builder import (
    KnowledgeGraphBuilder, KnowledgeGraph, Entity, Relationship,
    EntityType, RelationType
)
from ..knowledge_base.vector_store import VectorStore

logger = logging.getLogger(__name__)

class GraphTraversalMode(Enum):
    """Graph traversal modes"""
    ENTITY_CENTRIC = "entity_centric"
    RELATIONSHIP_AWARE = "relationship_aware"
    MULTI_HOP = "multi_hop"
    SEMANTIC_WALK = "semantic_walk"

class ContextExpansionStrategy(Enum):
    """Context expansion strategies"""
    IMMEDIATE_NEIGHBORS = "immediate_neighbors"
    TYPED_RELATIONSHIPS = "typed_relationships"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HIERARCHICAL = "hierarchical"

@dataclass
class GraphContext:
    """Graph-based context"""
    entities: List[Entity]
    relationships: List[Relationship]
    subgraph: KnowledgeGraph
    traversal_path: List[str]
    expansion_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphRetrievalResult:
    """Result of graph-based retrieval"""
    graph_context: GraphContext
    text_context: List[str]
    confidence: float
    reasoning_path: List[str]
    performance_metrics: Dict[str, float]

class EntityMatcher:
    """Match query entities with knowledge graph entities"""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        semantic_model: Optional[SentenceTransformer] = None
    ):
        self.knowledge_graph = knowledge_graph
        self.semantic_model = semantic_model
        self._initialize_semantic_model()

        # Create entity index for fast lookup
        self.entity_index = self._build_entity_index()

    def _initialize_semantic_model(self):
        """Initialize semantic similarity model"""
        if not self.semantic_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized semantic model for entity matching")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic model: {e}")

    def _build_entity_index(self) -> Dict[str, List[Entity]]:
        """Build index for fast entity lookup"""
        index = {}

        for entity in self.knowledge_graph.entities.values():
            # Index by text (normalized)
            key = entity.text.lower().strip()
            if key not in index:
                index[key] = []
            index[key].append(entity)

            # Index by aliases
            for alias in entity.aliases:
                alias_key = alias.lower().strip()
                if alias_key not in index:
                    index[alias_key] = []
                index[alias_key].append(entity)

        return index

    async def find_matching_entities(
        self,
        query: str,
        query_context: QueryContext,
        max_matches: int = 10
    ) -> List[Tuple[Entity, float]]:
        """Find entities that match the query"""
        matching_entities = []

        try:
            # Exact text matching
            exact_matches = self._find_exact_matches(query)
            matching_entities.extend([(entity, 1.0) for entity in exact_matches])

            # Partial text matching
            partial_matches = self._find_partial_matches(query)
            matching_entities.extend([(entity, 0.8) for entity in partial_matches])

            # Semantic similarity matching
            if self.semantic_model:
                semantic_matches = await self._find_semantic_matches(query, query_context)
                matching_entities.extend(semantic_matches)

            # Remove duplicates and sort by confidence
            seen_entities = set()
            unique_matches = []
            for entity, confidence in matching_entities:
                if entity.id not in seen_entities:
                    seen_entities.add(entity.id)
                    unique_matches.append((entity, confidence))

            unique_matches.sort(key=lambda x: x[1], reverse=True)
            return unique_matches[:max_matches]

        except Exception as e:
            logger.error(f"Error finding matching entities: {e}")
            return []

    def _find_exact_matches(self, query: str) -> List[Entity]:
        """Find entities with exact text matches"""
        query_words = query.lower().split()
        matches = []

        for word in query_words:
            if word in self.entity_index:
                matches.extend(self.entity_index[word])

        return list(set(matches))  # Remove duplicates

    def _find_partial_matches(self, query: str) -> List[Entity]:
        """Find entities with partial text matches"""
        query_lower = query.lower()
        matches = []

        for entity_text, entities in self.entity_index.items():
            # Check if entity text is in query or vice versa
            if (entity_text in query_lower or
                any(word in entity_text for word in query_lower.split()) or
                len(entity_text) > 3 and entity_text in query_lower):
                matches.extend(entities)

        return list(set(matches))

    async def _find_semantic_matches(
        self,
        query: str,
        query_context: QueryContext,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Entity, float]]:
        """Find entities using semantic similarity"""
        if not self.semantic_model:
            return []

        matches = []

        try:
            # Get query embedding
            query_embedding = self.semantic_model.encode([query])[0]

            # Calculate similarity with all entities
            for entity in self.knowledge_graph.entities.values():
                # Get entity embedding (cache if needed)
                if entity.embeddings:
                    entity_embedding = entity.embeddings
                else:
                    entity_embedding = self.semantic_model.encode([entity.text])[0]
                    entity.embeddings = entity_embedding.tolist()

                # Calculate similarity
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity([query_embedding], [entity_embedding])[0][0]
                else:
                    # Simple dot product similarity
                    similarity = sum(a * b for a, b in zip(query_embedding, entity_embedding))

                if similarity >= similarity_threshold:
                    matches.append((entity, float(similarity)))

        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")

        return matches

class GraphTraverser:
    """Traverse knowledge graph to expand context"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph

        # Build NetworkX graph for efficient traversal
        self.nx_graph = self._build_networkx_graph() if NETWORKX_AVAILABLE else None

    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from knowledge graph"""
        G = nx.DiGraph()

        # Add nodes (entities)
        for entity in self.knowledge_graph.entities.values():
            G.add_node(entity.id, entity=entity)

        # Add edges (relationships)
        for rel in self.knowledge_graph.relationships.values():
            G.add_edge(
                rel.source_entity.id,
                rel.target_entity.id,
                relationship=rel,
                weight=rel.confidence
            )

        return G

    async def expand_context(
        self,
        seed_entities: List[Entity],
        mode: GraphTraversalMode,
        strategy: ContextExpansionStrategy,
        max_depth: int = 2,
        max_entities: int = 50
    ) -> GraphContext:
        """Expand context through graph traversal"""
        try:
            if mode == GraphTraversalMode.ENTITY_CENTRIC:
                return await self._entity_centric_expansion(
                    seed_entities, strategy, max_depth, max_entities
                )
            elif mode == GraphTraversalMode.RELATIONSHIP_AWARE:
                return await self._relationship_aware_expansion(
                    seed_entities, strategy, max_depth, max_entities
                )
            elif mode == GraphTraversalMode.MULTI_HOP:
                return await self._multi_hop_expansion(
                    seed_entities, strategy, max_depth, max_entities
                )
            elif mode == GraphTraversalMode.SEMANTIC_WALK:
                return await self._semantic_walk_expansion(
                    seed_entities, strategy, max_depth, max_entities
                )
            else:
                # Default to entity-centric
                return await self._entity_centric_expansion(
                    seed_entities, strategy, max_depth, max_entities
                )

        except Exception as e:
            logger.error(f"Error expanding context: {e}")
            return GraphContext(
                entities=seed_entities,
                relationships=[],
                subgraph=KnowledgeGraph(),
                traversal_path=[]
            )

    async def _entity_centric_expansion(
        self,
        seed_entities: List[Entity],
        strategy: ContextExpansionStrategy,
        max_depth: int,
        max_entities: int
    ) -> GraphContext:
        """Entity-centric context expansion"""
        expanded_entities = set(seed_entities)
        expanded_relationships = []
        traversal_path = [f"seed:{','.join([e.id for e in seed_entities])}"]

        current_entities = seed_entities

        for depth in range(max_depth):
            if len(expanded_entities) >= max_entities:
                break

            next_entities = []

            for entity in current_entities:
                # Get relationships for this entity
                entity_rels = self.knowledge_graph.get_entity_relationships(entity.id)

                for rel in entity_rels:
                    # Add relationship
                    if rel not in expanded_relationships:
                        expanded_relationships.append(rel)

                    # Add connected entity
                    connected_entity = (rel.target_entity if rel.source_entity.id == entity.id
                                      else rel.source_entity)

                    if connected_entity not in expanded_entities:
                        if strategy == ContextExpansionStrategy.IMMEDIATE_NEIGHBORS:
                            # Add all immediate neighbors
                            expanded_entities.add(connected_entity)
                            next_entities.append(connected_entity)

                        elif strategy == ContextExpansionStrategy.TYPED_RELATIONSHIPS:
                            # Filter by relationship type
                            important_types = [
                                RelationType.CAUSES, RelationType.TREATS,
                                RelationType.MEASURES, RelationType.USED_FOR
                            ]
                            if rel.type in important_types:
                                expanded_entities.add(connected_entity)
                                next_entities.append(connected_entity)

                        elif strategy == ContextExpansionStrategy.SEMANTIC_SIMILARITY:
                            # Add based on entity type similarity
                            if self._entities_are_similar(entity, connected_entity):
                                expanded_entities.add(connected_entity)
                                next_entities.append(connected_entity)

            current_entities = next_entities
            traversal_path.append(f"depth_{depth}:{len(next_entities)}_entities")

            if not next_entities:
                break

        # Create subgraph
        subgraph = KnowledgeGraph()
        for entity in expanded_entities:
            subgraph.add_entity(entity)
        for rel in expanded_relationships:
            subgraph.add_relationship(rel)

        return GraphContext(
            entities=list(expanded_entities),
            relationships=expanded_relationships,
            subgraph=subgraph,
            traversal_path=traversal_path,
            expansion_metadata={
                "strategy": strategy.value,
                "max_depth": max_depth,
                "final_depth": depth + 1,
                "total_entities": len(expanded_entities),
                "total_relationships": len(expanded_relationships)
            }
        )

    async def _relationship_aware_expansion(
        self,
        seed_entities: List[Entity],
        strategy: ContextExpansionStrategy,
        max_depth: int,
        max_entities: int
    ) -> GraphContext:
        """Relationship-aware context expansion"""
        # Prioritize expansion based on relationship types and strengths
        expanded_entities = set(seed_entities)
        expanded_relationships = []
        traversal_path = []

        # Relationship type priorities
        relationship_priorities = {
            RelationType.CAUSES: 1.0,
            RelationType.TREATS: 0.9,
            RelationType.MEASURES: 0.8,
            RelationType.USED_FOR: 0.7,
            RelationType.ASSOCIATED_WITH: 0.5,
            RelationType.SIMILAR_TO: 0.4
        }

        current_entities = seed_entities

        for depth in range(max_depth):
            candidate_expansions = []

            for entity in current_entities:
                entity_rels = self.knowledge_graph.get_entity_relationships(entity.id)

                for rel in entity_rels:
                    if rel in expanded_relationships:
                        continue

                    # Calculate expansion score
                    rel_priority = relationship_priorities.get(rel.type, 0.3)
                    expansion_score = rel.confidence * rel_priority

                    connected_entity = (rel.target_entity if rel.source_entity.id == entity.id
                                      else rel.source_entity)

                    if connected_entity not in expanded_entities:
                        candidate_expansions.append((expansion_score, rel, connected_entity))

            # Sort by expansion score and take top candidates
            candidate_expansions.sort(key=lambda x: x[0], reverse=True)

            added_this_depth = 0
            max_per_depth = min(20, max_entities - len(expanded_entities))

            for score, rel, entity in candidate_expansions:
                if added_this_depth >= max_per_depth:
                    break

                expanded_relationships.append(rel)
                expanded_entities.add(entity)
                added_this_depth += 1

            traversal_path.append(f"depth_{depth}:score_based:{added_this_depth}_added")
            current_entities = [entity for _, _, entity in candidate_expansions[:added_this_depth]]

            if not current_entities or len(expanded_entities) >= max_entities:
                break

        # Create subgraph
        subgraph = KnowledgeGraph()
        for entity in expanded_entities:
            subgraph.add_entity(entity)
        for rel in expanded_relationships:
            subgraph.add_relationship(rel)

        return GraphContext(
            entities=list(expanded_entities),
            relationships=expanded_relationships,
            subgraph=subgraph,
            traversal_path=traversal_path,
            expansion_metadata={
                "strategy": "relationship_aware",
                "relationship_priorities": relationship_priorities
            }
        )

    async def _multi_hop_expansion(
        self,
        seed_entities: List[Entity],
        strategy: ContextExpansionStrategy,
        max_depth: int,
        max_entities: int
    ) -> GraphContext:
        """Multi-hop reasoning expansion"""
        # Use NetworkX for efficient multi-hop traversal
        if not self.nx_graph:
            return await self._entity_centric_expansion(seed_entities, strategy, max_depth, max_entities)

        expanded_entities = set(seed_entities)
        expanded_relationships = []
        traversal_path = []

        # Find all paths between seed entities (if multiple)
        if len(seed_entities) > 1:
            for i, source_entity in enumerate(seed_entities):
                for target_entity in seed_entities[i+1:]:
                    try:
                        # Find shortest path
                        if nx.has_path(self.nx_graph, source_entity.id, target_entity.id):
                            path = nx.shortest_path(self.nx_graph, source_entity.id, target_entity.id)

                            # Add entities and relationships along path
                            for j in range(len(path) - 1):
                                source_id = path[j]
                                target_id = path[j + 1]

                                # Add entities
                                if source_id in self.knowledge_graph.entities:
                                    expanded_entities.add(self.knowledge_graph.entities[source_id])
                                if target_id in self.knowledge_graph.entities:
                                    expanded_entities.add(self.knowledge_graph.entities[target_id])

                                # Find relationship
                                edge_data = self.nx_graph.get_edge_data(source_id, target_id)
                                if edge_data and 'relationship' in edge_data:
                                    expanded_relationships.append(edge_data['relationship'])

                            traversal_path.append(f"path:{source_entity.id}->{target_entity.id}:{len(path)}_hops")

                    except nx.NetworkXNoPath:
                        traversal_path.append(f"no_path:{source_entity.id}->{target_entity.id}")

        # Expand around each seed entity
        for entity in seed_entities:
            # Get entities within max_depth hops
            if entity.id in self.nx_graph:
                neighbors = nx.single_source_shortest_path(
                    self.nx_graph, entity.id, cutoff=max_depth
                )

                for target_id, path in neighbors.items():
                    if len(expanded_entities) >= max_entities:
                        break

                    if target_id in self.knowledge_graph.entities:
                        expanded_entities.add(self.knowledge_graph.entities[target_id])

                        # Add relationships along path
                        for j in range(len(path) - 1):
                            edge_data = self.nx_graph.get_edge_data(path[j], path[j + 1])
                            if edge_data and 'relationship' in edge_data:
                                rel = edge_data['relationship']
                                if rel not in expanded_relationships:
                                    expanded_relationships.append(rel)

        # Create subgraph
        subgraph = KnowledgeGraph()
        for entity in expanded_entities:
            subgraph.add_entity(entity)
        for rel in expanded_relationships:
            subgraph.add_relationship(rel)

        return GraphContext(
            entities=list(expanded_entities),
            relationships=expanded_relationships,
            subgraph=subgraph,
            traversal_path=traversal_path,
            expansion_metadata={
                "strategy": "multi_hop",
                "max_depth": max_depth,
                "paths_found": len([p for p in traversal_path if "path:" in p])
            }
        )

    async def _semantic_walk_expansion(
        self,
        seed_entities: List[Entity],
        strategy: ContextExpansionStrategy,
        max_depth: int,
        max_entities: int
    ) -> GraphContext:
        """Semantic random walk expansion"""
        # Implement semantic random walk for context expansion
        expanded_entities = set(seed_entities)
        expanded_relationships = []
        traversal_path = []

        current_entities = seed_entities

        for depth in range(max_depth):
            next_entities = []

            for entity in current_entities:
                if len(expanded_entities) >= max_entities:
                    break

                # Get neighboring entities
                entity_rels = self.knowledge_graph.get_entity_relationships(entity.id)

                if entity_rels:
                    # Calculate semantic similarity scores for neighbors
                    neighbor_scores = []

                    for rel in entity_rels:
                        connected_entity = (rel.target_entity if rel.source_entity.id == entity.id
                                          else rel.source_entity)

                        if connected_entity not in expanded_entities:
                            # Calculate semantic score
                            semantic_score = self._calculate_semantic_relevance(
                                entity, connected_entity, rel
                            )
                            neighbor_scores.append((semantic_score, rel, connected_entity))

                    # Select top semantic neighbors
                    neighbor_scores.sort(key=lambda x: x[0], reverse=True)

                    for score, rel, neighbor in neighbor_scores[:3]:  # Top 3 per entity
                        if neighbor not in expanded_entities:
                            expanded_entities.add(neighbor)
                            expanded_relationships.append(rel)
                            next_entities.append(neighbor)

            traversal_path.append(f"semantic_depth_{depth}:{len(next_entities)}_selected")
            current_entities = next_entities

            if not next_entities:
                break

        # Create subgraph
        subgraph = KnowledgeGraph()
        for entity in expanded_entities:
            subgraph.add_entity(entity)
        for rel in expanded_relationships:
            subgraph.add_relationship(rel)

        return GraphContext(
            entities=list(expanded_entities),
            relationships=expanded_relationships,
            subgraph=subgraph,
            traversal_path=traversal_path,
            expansion_metadata={"strategy": "semantic_walk"}
        )

    def _entities_are_similar(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities are semantically similar"""
        # Simple similarity based on type and text
        if entity1.type == entity2.type:
            return True

        # Related entity types
        related_types = {
            EntityType.DISEASE: [EntityType.CHEMICAL, EntityType.PROTEIN, EntityType.GENE],
            EntityType.TECHNIQUE: [EntityType.MEASUREMENT, EntityType.METHOD],
            EntityType.CHEMICAL: [EntityType.PROTEIN, EntityType.GENE],
        }

        if entity1.type in related_types:
            return entity2.type in related_types[entity1.type]

        return False

    def _calculate_semantic_relevance(
        self,
        source_entity: Entity,
        target_entity: Entity,
        relationship: Relationship
    ) -> float:
        """Calculate semantic relevance score for expansion"""
        score = relationship.confidence

        # Boost score for important entity types
        important_types = [EntityType.DISEASE, EntityType.TECHNIQUE, EntityType.METHOD]
        if target_entity.type in important_types:
            score *= 1.2

        # Boost score for important relationships
        important_rels = [RelationType.CAUSES, RelationType.TREATS, RelationType.MEASURES]
        if relationship.type in important_rels:
            score *= 1.1

        return score

class GraphRAGStrategy:
    """GraphRAG strategy for knowledge graph-enhanced retrieval"""

    def __init__(
        self,
        knowledge_graph_builder: KnowledgeGraphBuilder,
        vector_store: Optional[VectorStore] = None,
        default_traversal_mode: GraphTraversalMode = GraphTraversalMode.RELATIONSHIP_AWARE,
        default_expansion_strategy: ContextExpansionStrategy = ContextExpansionStrategy.TYPED_RELATIONSHIPS
    ):
        self.knowledge_graph_builder = knowledge_graph_builder
        self.vector_store = vector_store
        self.default_traversal_mode = default_traversal_mode
        self.default_expansion_strategy = default_expansion_strategy

        # Initialize components (lazy loading)
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.entity_matcher: Optional[EntityMatcher] = None
        self.graph_traverser: Optional[GraphTraverser] = None

        # Performance tracking
        self.retrieval_times: List[float] = []
        self.expansion_sizes: List[int] = []

    async def initialize(self, documents: List[Tuple[str, str]]):
        """Initialize GraphRAG with documents"""
        logger.info("Initializing GraphRAG strategy")

        try:
            # Build knowledge graph
            self.knowledge_graph = await self.knowledge_graph_builder.build_graph_from_documents(documents)

            # Initialize components
            self.entity_matcher = EntityMatcher(self.knowledge_graph)
            self.graph_traverser = GraphTraverser(self.knowledge_graph)

            logger.info(f"GraphRAG initialized with {len(self.knowledge_graph.entities)} entities")

        except Exception as e:
            logger.error(f"Error initializing GraphRAG: {e}")
            raise

    async def search(
        self,
        query_context: QueryContext,
        max_entities: int = 20,
        max_depth: int = 2,
        traversal_mode: Optional[GraphTraversalMode] = None,
        expansion_strategy: Optional[ContextExpansionStrategy] = None
    ) -> RAGResponse:
        """Execute GraphRAG search"""
        start_time = time.time()

        if not self.knowledge_graph or not self.entity_matcher or not self.graph_traverser:
            raise ValueError("GraphRAG not initialized. Call initialize() first.")

        try:
            # Find matching entities
            matching_entities = await self.entity_matcher.find_matching_entities(
                query_context.query, query_context, max_matches=10
            )

            if not matching_entities:
                # Fallback to empty response
                return RAGResponse(
                    answer="No relevant entities found in knowledge graph.",
                    sources=[],
                    confidence=0.1,
                    strategy_used=RAGStrategy.GRAPH_RAG,
                    performance_metrics=PerformanceMetrics(
                        strategy="graph_rag",
                        latency=time.time() - start_time,
                        quality_score=0.1,
                        context_size=0,
                        tokens_used=0
                    )
                )

            # Extract top entities
            seed_entities = [entity for entity, _ in matching_entities[:5]]

            # Expand context through graph traversal
            graph_context = await self.graph_traverser.expand_context(
                seed_entities,
                traversal_mode or self.default_traversal_mode,
                expansion_strategy or self.default_expansion_strategy,
                max_depth=max_depth,
                max_entities=max_entities
            )

            # Retrieve text context using vector store
            text_context = []
            if self.vector_store:
                text_context = await self._retrieve_text_context(graph_context, query_context)

            # Generate answer from graph and text context
            answer = await self._generate_answer(graph_context, text_context, query_context)

            # Calculate confidence based on entity matches and graph connectivity
            confidence = self._calculate_confidence(matching_entities, graph_context)

            # Create sources from entities and relationships
            sources = self._create_sources(graph_context)

            # Track performance
            retrieval_time = time.time() - start_time
            self.retrieval_times.append(retrieval_time)
            self.expansion_sizes.append(len(graph_context.entities))

            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                strategy_used=RAGStrategy.GRAPH_RAG,
                performance_metrics=PerformanceMetrics(
                    strategy="graph_rag",
                    latency=retrieval_time,
                    quality_score=confidence,
                    context_size=len(text_context),
                    tokens_used=len(answer.split()) if answer else 0
                ),
                metadata={
                    "matched_entities": len(matching_entities),
                    "expanded_entities": len(graph_context.entities),
                    "relationships": len(graph_context.relationships),
                    "traversal_path": graph_context.traversal_path,
                    "expansion_metadata": graph_context.expansion_metadata
                }
            )

        except Exception as e:
            logger.error(f"Error in GraphRAG search: {e}")
            return RAGResponse(
                answer=f"Error in GraphRAG search: {str(e)}",
                sources=[],
                confidence=0.0,
                strategy_used=RAGStrategy.GRAPH_RAG,
                performance_metrics=PerformanceMetrics(
                    strategy="graph_rag",
                    latency=time.time() - start_time,
                    quality_score=0.0,
                    context_size=0,
                    tokens_used=0
                )
            )

    async def _retrieve_text_context(
        self,
        graph_context: GraphContext,
        query_context: QueryContext
    ) -> List[str]:
        """Retrieve text context using vector store"""
        if not self.vector_store:
            return []

        text_contexts = []

        try:
            # Get text for each entity's source documents
            for entity in graph_context.entities:
                for doc_id in entity.source_docs[:2]:  # Limit to 2 docs per entity
                    # Retrieve document content
                    # This would depend on the vector store implementation
                    pass

            # Also search for query-relevant content
            query_results = await self.vector_store.similarity_search(
                query_context.query,
                k=5,
                collection_name="research_documents"
            )

            for result in query_results:
                text_contexts.append(result.get('content', ''))

        except Exception as e:
            logger.error(f"Error retrieving text context: {e}")

        return text_contexts[:10]  # Limit total context

    async def _generate_answer(
        self,
        graph_context: GraphContext,
        text_context: List[str],
        query_context: QueryContext
    ) -> str:
        """Generate answer from graph and text context"""
        try:
            # Create structured answer from graph context
            answer_parts = []

            # Include key entities
            if graph_context.entities:
                entity_names = [entity.text for entity in graph_context.entities[:5]]
                answer_parts.append(f"Key entities: {', '.join(entity_names)}")

            # Include key relationships
            if graph_context.relationships:
                rel_descriptions = []
                for rel in graph_context.relationships[:3]:
                    rel_desc = f"{rel.source_entity.text} {rel.type.value} {rel.target_entity.text}"
                    rel_descriptions.append(rel_desc)

                if rel_descriptions:
                    answer_parts.append(f"Key relationships: {'; '.join(rel_descriptions)}")

            # Include text context summary
            if text_context:
                # Simple summarization - take first sentences
                context_summary = []
                for context in text_context[:3]:
                    sentences = context.split('.')[:2]  # First 2 sentences
                    context_summary.extend(sentences)

                if context_summary:
                    summary_text = '. '.join(context_summary)[:500]  # Limit length
                    answer_parts.append(f"Context: {summary_text}")

            if answer_parts:
                return ' | '.join(answer_parts)
            else:
                return "Based on the knowledge graph analysis, relevant information was found but no specific answer could be generated."

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer from graph context."

    def _calculate_confidence(
        self,
        matching_entities: List[Tuple[Entity, float]],
        graph_context: GraphContext
    ) -> float:
        """Calculate confidence score"""
        if not matching_entities:
            return 0.0

        # Base confidence from entity matches
        entity_confidences = [confidence for _, confidence in matching_entities]
        base_confidence = sum(entity_confidences) / len(entity_confidences)

        # Boost confidence based on graph connectivity
        connectivity_boost = min(0.2, len(graph_context.relationships) / 20)

        # Boost confidence based on entity diversity
        entity_types = set(entity.type for entity in graph_context.entities)
        diversity_boost = min(0.1, len(entity_types) / 10)

        final_confidence = min(1.0, base_confidence + connectivity_boost + diversity_boost)
        return final_confidence

    def _create_sources(self, graph_context: GraphContext) -> List[Dict[str, Any]]:
        """Create sources from graph context"""
        sources = []

        # Add entities as sources
        for entity in graph_context.entities[:10]:  # Limit to top 10
            sources.append({
                "type": "entity",
                "id": entity.id,
                "text": entity.text,
                "entity_type": entity.type.value,
                "confidence": entity.confidence,
                "source_docs": entity.source_docs
            })

        # Add key relationships as sources
        for rel in graph_context.relationships[:5]:  # Limit to top 5
            sources.append({
                "type": "relationship",
                "id": rel.id,
                "source": rel.source_entity.text,
                "target": rel.target_entity.text,
                "relationship_type": rel.type.value,
                "confidence": rel.confidence,
                "evidence": rel.evidence[:200]  # Limit evidence length
            })

        return sources

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.retrieval_times:
            return {"error": "No retrieval data available"}

        return {
            "avg_retrieval_time": sum(self.retrieval_times) / len(self.retrieval_times),
            "avg_expansion_size": sum(self.expansion_sizes) / len(self.expansion_sizes),
            "total_retrievals": len(self.retrieval_times),
            "graph_stats": {
                "total_entities": len(self.knowledge_graph.entities) if self.knowledge_graph else 0,
                "total_relationships": len(self.knowledge_graph.relationships) if self.knowledge_graph else 0
            }
        }

def create_graph_rag_strategy(
    neo4j_uri: Optional[str] = None,
    neo4j_username: str = "neo4j",
    neo4j_password: str = "password",
    vector_store: Optional[VectorStore] = None
) -> GraphRAGStrategy:
    """Factory function to create GraphRAG strategy"""
    from .knowledge_graph_builder import create_knowledge_graph_builder

    kg_builder = create_knowledge_graph_builder(neo4j_uri, neo4j_username, neo4j_password)

    return GraphRAGStrategy(
        knowledge_graph_builder=kg_builder,
        vector_store=vector_store
    )

# Example usage
if __name__ == "__main__":
    async def test_graph_rag_strategy():
        """Test GraphRAG strategy"""
        strategy = create_graph_rag_strategy()

        # Test documents
        documents = [
            ("doc1", "fMRI measures brain activity using magnetic resonance imaging. It is widely used in neuroscience research."),
            ("doc2", "Machine learning algorithms can analyze fMRI data to detect patterns in brain activity related to cognitive functions."),
            ("doc3", "Autism spectrum disorder affects social communication and may show distinct patterns in fMRI studies."),
        ]

        try:
            # Initialize with documents
            await strategy.initialize(documents)

            # Test query
            from ..rag.unified_rag_orchestrator import QueryContext, QueryComplexity, QueryDomain

            query_context = QueryContext(
                query="How does fMRI help study autism?",
                complexity=QueryComplexity.MEDIUM,
                domain=QueryDomain.NEUROSCIENCE,
                intent="causal",
                confidence=0.9,
                metadata={}
            )

            # Execute search
            response = await strategy.search(query_context)

            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Sources: {len(response.sources)}")
            print(f"Performance: {response.performance_metrics}")

        except Exception as e:
            print(f"Test failed: {e}")

    # Run test
    asyncio.run(test_graph_rag_strategy())