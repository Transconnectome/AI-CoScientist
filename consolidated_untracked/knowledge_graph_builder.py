"""
Knowledge Graph Builder for GraphRAG Integration

Implementation for: Knowledge graph construction from documents
Created: 2025-12-05

Acceptance Criteria:
- Entity extraction with scientific domain awareness
- Relationship detection and graph construction
- Neo4j integration with cypher query generation
- SciBERT optimization for scientific text processing

This module provides intelligent knowledge graph construction from scientific
documents with domain-specific entity recognition and relationship extraction.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib

# External dependencies with fallbacks
try:
    import neo4j
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import spacy
    from spacy import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Core dependencies
from datetime import datetime

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Scientific entity types"""
    CONCEPT = "concept"
    METHOD = "method"
    MEASUREMENT = "measurement"
    ORGANISM = "organism"
    DISEASE = "disease"
    CHEMICAL = "chemical"
    GENE = "gene"
    PROTEIN = "protein"
    TECHNIQUE = "technique"
    LOCATION = "location"
    PERSON = "person"
    ORGANIZATION = "organization"
    PUBLICATION = "publication"
    DATASET = "dataset"

class RelationType(Enum):
    """Relationship types"""
    CAUSES = "causes"
    TREATS = "treats"
    ASSOCIATED_WITH = "associated_with"
    PART_OF = "part_of"
    USED_FOR = "used_for"
    MEASURES = "measures"
    LOCATED_IN = "located_in"
    DEVELOPED_BY = "developed_by"
    BASED_ON = "based_on"
    SIMILAR_TO = "similar_to"
    AFFECTS = "affects"
    REQUIRES = "requires"
    PRODUCES = "produces"
    DERIVED_FROM = "derived_from"

@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    text: str
    type: EntityType
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    source_docs: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.id == other.id

@dataclass
class Relationship:
    """Knowledge graph relationship"""
    id: str
    source_entity: Entity
    target_entity: Entity
    type: RelationType
    confidence: float
    evidence: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_docs: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

@dataclass
class KnowledgeGraph:
    """Knowledge graph representation"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: Entity):
        """Add entity to graph"""
        self.entities[entity.id] = entity

    def add_relationship(self, relationship: Relationship):
        """Add relationship to graph"""
        self.relationships[relationship.id] = relationship

    def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity"""
        return [
            rel for rel in self.relationships.values()
            if rel.source_entity.id == entity_id or rel.target_entity.id == entity_id
        ]

    def get_subgraph(self, entity_ids: List[str], depth: int = 1) -> 'KnowledgeGraph':
        """Extract subgraph around specific entities"""
        subgraph = KnowledgeGraph()

        # Add initial entities
        for entity_id in entity_ids:
            if entity_id in self.entities:
                subgraph.add_entity(self.entities[entity_id])

        # Expand to specified depth
        current_entities = set(entity_ids)
        for _ in range(depth):
            next_entities = set()

            for entity_id in current_entities:
                for rel in self.get_entity_relationships(entity_id):
                    # Add relationship
                    subgraph.add_relationship(rel)

                    # Add connected entities
                    if rel.source_entity.id not in subgraph.entities:
                        subgraph.add_entity(rel.source_entity)
                        next_entities.add(rel.source_entity.id)

                    if rel.target_entity.id not in subgraph.entities:
                        subgraph.add_entity(rel.target_entity)
                        next_entities.add(rel.target_entity.id)

            current_entities = next_entities

        return subgraph

class EntityExtractor(ABC):
    """Abstract entity extractor"""

    @abstractmethod
    async def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities from text"""
        pass

class SciBERTEntityExtractor(EntityExtractor):
    """SciBERT-based entity extractor for scientific text"""

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self._initialize_model()

        # Scientific domain patterns
        self.domain_patterns = {
            EntityType.MEASUREMENT: [
                r'\b\d+\.?\d*\s*(mg|kg|ml|cm|mm|μm|nm|Hz|kHz|MHz|GHz|°C|°F|%|ppm|mmHg|Torr)\b',
                r'\bp\s*<?\s*0\.\d+\b',  # p-values
                r'\b(mean|average|median)\s*[±=]\s*\d+\.?\d*\b'
            ],
            EntityType.TECHNIQUE: [
                r'\b(fMRI|MRI|EEG|PET|CT|MEG|TMS|DTI)\b',
                r'\b(PCR|qPCR|RT-PCR|ELISA|Western blot|immunofluorescence)\b',
                r'\b(machine learning|deep learning|neural network|CNN|RNN|LSTM)\b'
            ],
            EntityType.CHEMICAL: [
                r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b',  # Chemical formulas
                r'\bdopamine|serotonin|acetylcholine|GABA|glutamate\b'
            ],
            EntityType.DISEASE: [
                r'\bautism spectrum disorder|ASD|ADHD|depression|anxiety|schizophrenia\b',
                r'\bAlzheimer\'s|Parkinson\'s|multiple sclerosis|epilepsy\b'
            ]
        }

    def _initialize_model(self):
        """Initialize SciBERT model and pipeline"""
        try:
            if TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

                # Create NER pipeline if model supports it
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple"
                )
                logger.info(f"Initialized SciBERT entity extractor: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize SciBERT model: {e}")

    async def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using SciBERT and domain patterns"""
        entities = []

        try:
            # SciBERT NER extraction
            if self.ner_pipeline:
                scibert_entities = await self._extract_scibert_entities(text, doc_id)
                entities.extend(scibert_entities)

            # Domain pattern extraction
            pattern_entities = await self._extract_pattern_entities(text, doc_id)
            entities.extend(pattern_entities)

            # Merge similar entities
            entities = self._merge_similar_entities(entities)

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")

        return entities

    async def _extract_scibert_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using SciBERT NER"""
        entities = []

        try:
            # Split text into chunks for processing
            chunks = self._split_text(text, max_length=512)

            for chunk in chunks:
                ner_results = self.ner_pipeline(chunk)

                for result in ner_results:
                    entity_text = result['word']
                    entity_type = self._map_scibert_label(result['entity_group'])
                    confidence = result['score']

                    if confidence > 0.5 and len(entity_text) > 2:
                        entity_id = self._generate_entity_id(entity_text, entity_type)

                        entity = Entity(
                            id=entity_id,
                            text=entity_text,
                            type=entity_type,
                            confidence=confidence,
                            source_docs=[doc_id],
                            properties={
                                'start': result.get('start', 0),
                                'end': result.get('end', 0),
                                'original_label': result['entity_group']
                            }
                        )

                        entities.append(entity)

        except Exception as e:
            logger.error(f"SciBERT entity extraction error: {e}")

        return entities

    async def _extract_pattern_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using domain-specific patterns"""
        entities = []

        for entity_type, patterns in self.domain_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    entity_text = match.group().strip()
                    entity_id = self._generate_entity_id(entity_text, entity_type)

                    entity = Entity(
                        id=entity_id,
                        text=entity_text,
                        type=entity_type,
                        confidence=0.8,  # High confidence for pattern matches
                        source_docs=[doc_id],
                        properties={
                            'start': match.start(),
                            'end': match.end(),
                            'pattern_matched': pattern
                        }
                    )

                    entities.append(entity)

        return entities

    def _split_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into manageable chunks"""
        # Simple sentence-based splitting
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _map_scibert_label(self, label: str) -> EntityType:
        """Map SciBERT labels to our entity types"""
        label_mapping = {
            'CHEMICAL': EntityType.CHEMICAL,
            'DISEASE': EntityType.DISEASE,
            'GENE': EntityType.GENE,
            'SPECIES': EntityType.ORGANISM,
            'MUTATION': EntityType.GENE,
            'CELL_LINE': EntityType.ORGANISM,
            'CELL_TYPE': EntityType.ORGANISM,
            'PROTEIN': EntityType.PROTEIN,
            'DNA': EntityType.GENE,
            'RNA': EntityType.GENE,
        }

        return label_mapping.get(label.upper(), EntityType.CONCEPT)

    def _generate_entity_id(self, text: str, entity_type: EntityType) -> str:
        """Generate unique entity ID"""
        normalized_text = text.lower().strip()
        content = f"{entity_type.value}:{normalized_text}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _merge_similar_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge similar entities to avoid duplicates"""
        merged_entities = {}

        for entity in entities:
            key = (entity.text.lower(), entity.type)

            if key in merged_entities:
                # Merge with existing entity
                existing = merged_entities[key]
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.source_docs.extend(entity.source_docs)
                existing.source_docs = list(set(existing.source_docs))  # Remove duplicates
            else:
                merged_entities[key] = entity

        return list(merged_entities.values())

class SpacyEntityExtractor(EntityExtractor):
    """spaCy-based entity extractor for fallback"""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize spaCy model"""
        try:
            if SPACY_AVAILABLE:
                self.nlp = spacy.load(self.model_name)
                logger.info(f"Initialized spaCy entity extractor: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize spaCy model: {e}")

    async def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        entities = []

        if not self.nlp:
            return entities

        try:
            doc = self.nlp(text)

            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                entity_id = self._generate_entity_id(ent.text, entity_type)

                entity = Entity(
                    id=entity_id,
                    text=ent.text,
                    type=entity_type,
                    confidence=0.7,  # Default confidence for spaCy
                    source_docs=[doc_id],
                    properties={
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'spacy_label': ent.label_
                    }
                )

                entities.append(entity)

        except Exception as e:
            logger.error(f"spaCy entity extraction error: {e}")

        return entities

    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy labels to our entity types"""
        label_mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'DISEASE': EntityType.DISEASE,
            'CHEMICAL': EntityType.CHEMICAL,
            'GENE': EntityType.GENE,
        }

        return label_mapping.get(label, EntityType.CONCEPT)

    def _generate_entity_id(self, text: str, entity_type: EntityType) -> str:
        """Generate unique entity ID"""
        normalized_text = text.lower().strip()
        content = f"{entity_type.value}:{normalized_text}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

class RelationshipExtractor:
    """Extract relationships between entities"""

    def __init__(self):
        # Relationship patterns
        self.relationship_patterns = {
            RelationType.CAUSES: [
                r'\b(\w+)\s+(causes?|leads? to|results? in|induces?)\s+(\w+)',
                r'\b(\w+)\s+is\s+caused\s+by\s+(\w+)',
            ],
            RelationType.TREATS: [
                r'\b(\w+)\s+(treats?|cures?|alleviates?)\s+(\w+)',
                r'\b(\w+)\s+treatment\s+for\s+(\w+)',
            ],
            RelationType.ASSOCIATED_WITH: [
                r'\b(\w+)\s+(associated\s+with|correlated\s+with|linked\s+to)\s+(\w+)',
                r'\b(\w+)\s+and\s+(\w+)\s+are\s+related',
            ],
            RelationType.MEASURES: [
                r'\b(\w+)\s+(measures?|quantifies?|assesses?)\s+(\w+)',
                r'\bmeasurement\s+of\s+(\w+)\s+using\s+(\w+)',
            ],
            RelationType.USED_FOR: [
                r'\b(\w+)\s+(used\s+for|applied\s+to|utilized\s+in)\s+(\w+)',
                r'\b(\w+)\s+is\s+a\s+method\s+for\s+(\w+)',
            ]
        }

    async def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        doc_id: str
    ) -> List[Relationship]:
        """Extract relationships from text given entities"""
        relationships = []

        try:
            # Pattern-based relationship extraction
            pattern_rels = await self._extract_pattern_relationships(text, entities, doc_id)
            relationships.extend(pattern_rels)

            # Co-occurrence based relationships
            cooccurrence_rels = await self._extract_cooccurrence_relationships(text, entities, doc_id)
            relationships.extend(cooccurrence_rels)

        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")

        return relationships

    async def _extract_pattern_relationships(
        self,
        text: str,
        entities: List[Entity],
        doc_id: str
    ) -> List[Relationship]:
        """Extract relationships using patterns"""
        relationships = []

        # Create entity lookup
        entity_lookup = {entity.text.lower(): entity for entity in entities}

        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_text = groups[0].lower()
                        target_text = groups[-1].lower()

                        # Find matching entities
                        source_entity = None
                        target_entity = None

                        for entity_text, entity in entity_lookup.items():
                            if entity_text in source_text or source_text in entity_text:
                                source_entity = entity
                            if entity_text in target_text or target_text in entity_text:
                                target_entity = entity

                        if source_entity and target_entity and source_entity != target_entity:
                            rel_id = self._generate_relationship_id(
                                source_entity, target_entity, rel_type
                            )

                            relationship = Relationship(
                                id=rel_id,
                                source_entity=source_entity,
                                target_entity=target_entity,
                                type=rel_type,
                                confidence=0.8,
                                evidence=match.group(),
                                source_docs=[doc_id],
                                properties={
                                    'pattern_matched': pattern,
                                    'start': match.start(),
                                    'end': match.end()
                                }
                            )

                            relationships.append(relationship)

        return relationships

    async def _extract_cooccurrence_relationships(
        self,
        text: str,
        entities: List[Entity],
        doc_id: str,
        window_size: int = 50
    ) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence"""
        relationships = []

        try:
            # Find entity positions in text
            entity_positions = []
            for entity in entities:
                for match in re.finditer(re.escape(entity.text), text, re.IGNORECASE):
                    entity_positions.append((entity, match.start(), match.end()))

            # Sort by position
            entity_positions.sort(key=lambda x: x[1])

            # Find co-occurring entities within window
            for i, (entity1, start1, end1) in enumerate(entity_positions):
                for j, (entity2, start2, end2) in enumerate(entity_positions[i+1:], i+1):
                    # Check if within window
                    distance = start2 - end1
                    if distance <= window_size and entity1 != entity2:
                        # Create generic association relationship
                        rel_id = self._generate_relationship_id(
                            entity1, entity2, RelationType.ASSOCIATED_WITH
                        )

                        # Calculate confidence based on distance (closer = higher confidence)
                        confidence = max(0.3, 1.0 - (distance / window_size))

                        relationship = Relationship(
                            id=rel_id,
                            source_entity=entity1,
                            target_entity=entity2,
                            type=RelationType.ASSOCIATED_WITH,
                            confidence=confidence,
                            evidence=text[start1:end2],
                            source_docs=[doc_id],
                            properties={
                                'extraction_method': 'co-occurrence',
                                'distance': distance,
                                'window_size': window_size
                            }
                        )

                        relationships.append(relationship)
                    elif distance > window_size:
                        # Entities are too far apart, stop checking
                        break

        except Exception as e:
            logger.error(f"Co-occurrence relationship extraction error: {e}")

        return relationships

    def _generate_relationship_id(
        self,
        source_entity: Entity,
        target_entity: Entity,
        rel_type: RelationType
    ) -> str:
        """Generate unique relationship ID"""
        content = f"{source_entity.id}:{rel_type.value}:{target_entity.id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

class Neo4jGraphStore:
    """Neo4j graph database storage"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password"
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver: Optional[Driver] = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize Neo4j connection"""
        try:
            if NEO4J_AVAILABLE:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")

    async def store_graph(self, graph: KnowledgeGraph) -> bool:
        """Store knowledge graph in Neo4j"""
        if not self.driver:
            logger.warning("Neo4j driver not available")
            return False

        try:
            with self.driver.session() as session:
                # Create entities
                for entity in graph.entities.values():
                    await self._create_entity(session, entity)

                # Create relationships
                for relationship in graph.relationships.values():
                    await self._create_relationship(session, relationship)

            logger.info(f"Stored graph with {len(graph.entities)} entities and {len(graph.relationships)} relationships")
            return True

        except Exception as e:
            logger.error(f"Error storing graph in Neo4j: {e}")
            return False

    async def _create_entity(self, session: Session, entity: Entity):
        """Create entity in Neo4j"""
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.text = $text,
            e.type = $type,
            e.confidence = $confidence,
            e.properties = $properties,
            e.source_docs = $source_docs
        """

        session.run(query, {
            'entity_id': entity.id,
            'text': entity.text,
            'type': entity.type.value,
            'confidence': entity.confidence,
            'properties': json.dumps(entity.properties),
            'source_docs': entity.source_docs
        })

    async def _create_relationship(self, session: Session, relationship: Relationship):
        """Create relationship in Neo4j"""
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATED {type: $rel_type}]->(target)
        SET r.confidence = $confidence,
            r.evidence = $evidence,
            r.properties = $properties,
            r.source_docs = $source_docs
        """

        session.run(query, {
            'source_id': relationship.source_entity.id,
            'target_id': relationship.target_entity.id,
            'rel_type': relationship.type.value,
            'confidence': relationship.confidence,
            'evidence': relationship.evidence,
            'properties': json.dumps(relationship.properties),
            'source_docs': relationship.source_docs
        })

    async def query_graph(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute cypher query on graph"""
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]

        except Exception as e:
            logger.error(f"Error executing cypher query: {e}")
            return []

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

class KnowledgeGraphBuilder:
    """Main knowledge graph builder"""

    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        relationship_extractor: Optional[RelationshipExtractor] = None,
        graph_store: Optional[Neo4jGraphStore] = None
    ):
        self.entity_extractor = entity_extractor or self._create_default_extractor()
        self.relationship_extractor = relationship_extractor or RelationshipExtractor()
        self.graph_store = graph_store

        # Build cache
        self.entity_cache: Dict[str, List[Entity]] = {}
        self.relationship_cache: Dict[str, List[Relationship]] = {}

    def _create_default_extractor(self) -> EntityExtractor:
        """Create default entity extractor"""
        if TRANSFORMERS_AVAILABLE:
            return SciBERTEntityExtractor()
        elif SPACY_AVAILABLE:
            return SpacyEntityExtractor()
        else:
            logger.warning("No entity extraction models available")
            return None

    async def build_graph_from_documents(
        self,
        documents: List[Tuple[str, str]],  # (doc_id, content)
        merge_similar: bool = True
    ) -> KnowledgeGraph:
        """Build knowledge graph from documents"""
        logger.info(f"Building knowledge graph from {len(documents)} documents")

        graph = KnowledgeGraph()
        all_entities = []
        all_relationships = []

        try:
            # Extract entities and relationships from each document
            for doc_id, content in documents:
                # Extract entities
                doc_entities = await self.entity_extractor.extract_entities(content, doc_id)
                all_entities.extend(doc_entities)

                # Extract relationships
                doc_relationships = await self.relationship_extractor.extract_relationships(
                    content, doc_entities, doc_id
                )
                all_relationships.extend(doc_relationships)

            # Merge similar entities if requested
            if merge_similar:
                all_entities = self._merge_entities_globally(all_entities)
                all_relationships = self._update_relationships_after_merge(
                    all_relationships, all_entities
                )

            # Add to graph
            for entity in all_entities:
                graph.add_entity(entity)

            for relationship in all_relationships:
                graph.add_relationship(relationship)

            # Store in graph database if available
            if self.graph_store:
                await self.graph_store.store_graph(graph)

            logger.info(f"Built graph with {len(graph.entities)} entities and {len(graph.relationships)} relationships")

        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")

        return graph

    def _merge_entities_globally(self, entities: List[Entity]) -> List[Entity]:
        """Merge similar entities across all documents"""
        # Group entities by normalized text and type
        entity_groups = {}

        for entity in entities:
            key = (entity.text.lower().strip(), entity.type)

            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # Merge entities in each group
        merged_entities = []
        for group in entity_groups.values():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # Merge entities
                primary_entity = group[0]

                # Combine confidence (average)
                primary_entity.confidence = sum(e.confidence for e in group) / len(group)

                # Combine source docs
                all_source_docs = []
                for entity in group:
                    all_source_docs.extend(entity.source_docs)
                primary_entity.source_docs = list(set(all_source_docs))

                # Combine aliases
                all_aliases = []
                for entity in group:
                    all_aliases.extend(entity.aliases)
                    if entity.text not in all_aliases:
                        all_aliases.append(entity.text)
                primary_entity.aliases = list(set(all_aliases))

                merged_entities.append(primary_entity)

        return merged_entities

    def _update_relationships_after_merge(
        self,
        relationships: List[Relationship],
        merged_entities: List[Entity]
    ) -> List[Relationship]:
        """Update relationships after entity merging"""
        # Create entity lookup by text and type
        entity_lookup = {}
        for entity in merged_entities:
            for alias in [entity.text] + entity.aliases:
                key = (alias.lower().strip(), entity.type)
                entity_lookup[key] = entity

        # Update relationships
        updated_relationships = []
        for rel in relationships:
            # Find updated source entity
            source_key = (rel.source_entity.text.lower().strip(), rel.source_entity.type)
            target_key = (rel.target_entity.text.lower().strip(), rel.target_entity.type)

            if source_key in entity_lookup and target_key in entity_lookup:
                rel.source_entity = entity_lookup[source_key]
                rel.target_entity = entity_lookup[target_key]
                updated_relationships.append(rel)

        return updated_relationships

    async def query_graph(self, query: str, entities: List[str]) -> KnowledgeGraph:
        """Query graph for relevant subgraph"""
        if not self.graph_store:
            return KnowledgeGraph()

        # Build cypher query
        cypher_query = """
        MATCH (e:Entity)
        WHERE e.text IN $entities
        OPTIONAL MATCH (e)-[r:RELATED]-(connected)
        RETURN e, r, connected
        """

        try:
            results = await self.graph_store.query_graph(cypher_query, {'entities': entities})

            # Build subgraph from results
            subgraph = KnowledgeGraph()

            for result in results:
                # Add entities and relationships from query results
                # Implementation would depend on Neo4j result structure
                pass

            return subgraph

        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return KnowledgeGraph()

def create_knowledge_graph_builder(
    neo4j_uri: Optional[str] = None,
    neo4j_username: str = "neo4j",
    neo4j_password: str = "password"
) -> KnowledgeGraphBuilder:
    """Factory function to create knowledge graph builder"""
    graph_store = None
    if neo4j_uri and NEO4J_AVAILABLE:
        graph_store = Neo4jGraphStore(neo4j_uri, neo4j_username, neo4j_password)

    return KnowledgeGraphBuilder(graph_store=graph_store)

# Example usage
if __name__ == "__main__":
    async def test_knowledge_graph_builder():
        """Test knowledge graph builder"""
        builder = create_knowledge_graph_builder()

        # Test documents
        documents = [
            ("doc1", "fMRI measures brain activity using magnetic resonance. It is used for neuroscience research."),
            ("doc2", "Machine learning algorithms can analyze fMRI data to detect patterns in brain activity."),
        ]

        # Build graph
        graph = await builder.build_graph_from_documents(documents)

        print(f"Built graph with {len(graph.entities)} entities and {len(graph.relationships)} relationships")

        # Print entities
        for entity in graph.entities.values():
            print(f"Entity: {entity.text} ({entity.type.value}) - confidence: {entity.confidence:.2f}")

        # Print relationships
        for rel in graph.relationships.values():
            print(f"Relationship: {rel.source_entity.text} --{rel.type.value}--> {rel.target_entity.text}")

    # Run test
    asyncio.run(test_knowledge_graph_builder())