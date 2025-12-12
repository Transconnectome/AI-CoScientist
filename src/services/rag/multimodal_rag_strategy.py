"""
Multimodal RAG Strategy

Implementation for: Cross-modal retrieval and reasoning
Created: 2025-12-05

Acceptance Criteria:
- Cross-modal similarity matching (text-image-table)
- Multimodal embeddings integration
- Vision-language model reasoning
- Unified multimodal response generation

This module provides multimodal RAG capabilities with cross-modal retrieval,
vision-language reasoning, and unified response generation across modalities.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import base64
import io

# External dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        CLIPProcessor, CLIPModel,
        AutoTokenizer, AutoModel
    )
    VISION_MODELS_AVAILABLE = True
except ImportError:
    VISION_MODELS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core dependencies
from datetime import datetime
from ..rag.unified_rag_orchestrator import (
    RAGStrategy, QueryContext, RAGResponse, PerformanceMetrics
)
from ..rag.multimodal_document_processor import (
    MultimodalDocumentProcessor, ProcessedDocument, ContentBlock,
    ModalityType, ProcessingQuality, create_multimodal_processor
)
from ..knowledge_base.vector_store import VectorStore

logger = logging.getLogger(__name__)

class CrossModalSimilarity(Enum):
    """Cross-modal similarity types"""
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_TABLE = "text_to_table"
    IMAGE_TO_TEXT = "image_to_text"
    IMAGE_TO_TABLE = "image_to_table"
    TABLE_TO_TEXT = "table_to_text"
    TABLE_TO_IMAGE = "table_to_image"

class ReasoningMode(Enum):
    """Multimodal reasoning modes"""
    TEXT_ONLY = "text_only"
    VISUAL_REASONING = "visual_reasoning"
    CROSS_MODAL = "cross_modal"
    MULTIMODAL_FUSION = "multimodal_fusion"

@dataclass
class MultimodalMatch:
    """Match between query and multimodal content"""
    content_block: ContentBlock
    similarity_score: float
    cross_modal_type: Optional[CrossModalSimilarity]
    reasoning_evidence: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultimodalContext:
    """Multimodal context for response generation"""
    text_blocks: List[ContentBlock]
    image_blocks: List[ContentBlock]
    table_blocks: List[ContentBlock]
    cross_modal_matches: List[MultimodalMatch]
    reasoning_chain: List[str]
    confidence_scores: Dict[str, float]

class MultimodalEmbedder:
    """Multimodal embedding generation and similarity computation"""

    def __init__(self):
        self.text_model = None
        self.vision_model = None
        self.clip_processor = None
        self.clip_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize multimodal embedding models"""
        try:
            # Text embeddings
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized text embedding model")

            # Vision-language embeddings (CLIP)
            if VISION_MODELS_AVAILABLE:
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("Initialized CLIP vision-language model")

        except Exception as e:
            logger.warning(f"Failed to initialize multimodal models: {e}")

    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate text embeddings"""
        try:
            if self.text_model:
                embedding = self.text_model.encode([text])
                return embedding[0]
        except Exception as e:
            logger.error(f"Text embedding error: {e}")
        return None

    async def embed_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Generate image embeddings"""
        try:
            if self.clip_processor and self.clip_model and PIL_AVAILABLE:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_data))

                # Process with CLIP
                inputs = self.clip_processor(images=image, return_tensors="pt")
                image_features = self.clip_model.get_image_features(**inputs)

                return image_features.detach().numpy()[0]
        except Exception as e:
            logger.error(f"Image embedding error: {e}")
        return None

    async def embed_table(self, table_content: str) -> Optional[np.ndarray]:
        """Generate table embeddings (using text model on table content)"""
        # Tables are embedded as structured text
        return await self.embed_text(table_content)

    async def compute_cross_modal_similarity(
        self,
        text: str,
        image_data: Optional[bytes] = None,
        table_content: Optional[str] = None
    ) -> Dict[CrossModalSimilarity, float]:
        """Compute cross-modal similarity scores"""
        similarities = {}

        try:
            # Get text embedding
            text_embedding = await self.embed_text(text)

            if text_embedding is not None:
                # Text-to-image similarity
                if image_data:
                    image_embedding = await self.embed_image(image_data)
                    if image_embedding is not None:
                        if SKLEARN_AVAILABLE:
                            sim = cosine_similarity([text_embedding], [image_embedding])[0][0]
                            similarities[CrossModalSimilarity.TEXT_TO_IMAGE] = float(sim)

                # Text-to-table similarity
                if table_content:
                    table_embedding = await self.embed_table(table_content)
                    if table_embedding is not None and SKLEARN_AVAILABLE:
                        sim = cosine_similarity([text_embedding], [table_embedding])[0][0]
                        similarities[CrossModalSimilarity.TEXT_TO_TABLE] = float(sim)

        except Exception as e:
            logger.error(f"Cross-modal similarity computation error: {e}")

        return similarities

class VisualReasoner:
    """Vision-language reasoning for multimodal content"""

    def __init__(self):
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize vision-language models"""
        try:
            if VISION_MODELS_AVAILABLE:
                # BLIP for image captioning and VQA
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

                # CLIP for text-image matching
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

                logger.info("Initialized vision-language reasoning models")

        except Exception as e:
            logger.warning(f"Failed to initialize vision models: {e}")

    async def caption_image(self, image_data: bytes) -> str:
        """Generate image caption"""
        try:
            if self.blip_processor and self.blip_model and PIL_AVAILABLE:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_data))

                # Generate caption
                inputs = self.blip_processor(image, return_tensors="pt")
                out = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

                return caption
            else:
                return "Image caption (vision model not available)"
        except Exception as e:
            logger.error(f"Image captioning error: {e}")
            return "Image caption generation failed"

    async def answer_visual_question(self, image_data: bytes, question: str) -> str:
        """Answer question about image"""
        try:
            if self.blip_processor and self.blip_model and PIL_AVAILABLE:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_data))

                # Process image and question for VQA
                inputs = self.blip_processor(image, question, return_tensors="pt")
                out = self.blip_model.generate(**inputs, max_length=30)
                answer = self.blip_processor.decode(out[0], skip_special_tokens=True)

                return answer
            else:
                return "Visual question answering not available"
        except Exception as e:
            logger.error(f"Visual QA error: {e}")
            return "Visual question answering failed"

    async def analyze_image_content(self, image_data: bytes) -> Dict[str, Any]:
        """Comprehensive image content analysis"""
        analysis = {}

        try:
            # Generate caption
            caption = await self.caption_image(image_data)
            analysis["caption"] = caption

            # Analyze image type
            analysis["content_type"] = await self._classify_image_type(image_data)

            # Extract key visual elements
            analysis["visual_elements"] = await self._extract_visual_elements(image_data)

        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            analysis["error"] = str(e)

        return analysis

    async def _classify_image_type(self, image_data: bytes) -> str:
        """Classify type of scientific image"""
        try:
            # Use CLIP to classify image type
            if self.clip_processor and self.clip_model and PIL_AVAILABLE:
                image = Image.open(io.BytesIO(image_data))

                # Define image type candidates
                type_candidates = [
                    "scientific chart", "graph", "diagram", "brain scan",
                    "microscopy image", "photograph", "illustration",
                    "table", "flowchart", "brain imaging"
                ]

                # Compute similarity with each candidate
                inputs = self.clip_processor(
                    text=type_candidates,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )

                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                # Get most likely type
                best_idx = probs.argmax().item()
                confidence = probs[0][best_idx].item()

                if confidence > 0.3:
                    return type_candidates[best_idx]
                else:
                    return "unclassified image"
            else:
                return "scientific image"
        except Exception as e:
            logger.error(f"Image classification error: {e}")
            return "unknown image type"

    async def _extract_visual_elements(self, image_data: bytes) -> List[str]:
        """Extract key visual elements from image"""
        elements = []

        try:
            # Use predefined visual questions to extract elements
            questions = [
                "What type of data is shown?",
                "Are there any graphs or charts?",
                "Is there text visible in the image?",
                "What colors are prominent?",
                "Are there any anatomical structures?"
            ]

            for question in questions:
                answer = await self.answer_visual_question(image_data, question)
                if answer and answer != "Visual question answering failed":
                    elements.append(f"{question} -> {answer}")

        except Exception as e:
            logger.error(f"Visual elements extraction error: {e}")

        return elements

class MultimodalRetriever:
    """Multimodal content retrieval and matching"""

    def __init__(
        self,
        embedder: MultimodalEmbedder,
        visual_reasoner: VisualReasoner,
        vector_store: Optional[VectorStore] = None
    ):
        self.embedder = embedder
        self.visual_reasoner = visual_reasoner
        self.vector_store = vector_store

        # Content cache
        self.processed_documents: Dict[str, ProcessedDocument] = {}
        self.content_embeddings: Dict[str, np.ndarray] = {}

    async def index_document(self, processed_doc: ProcessedDocument):
        """Index processed document for retrieval"""
        try:
            # Store document
            self.processed_documents[processed_doc.document_id] = processed_doc

            # Generate embeddings for content blocks
            for block in processed_doc.content_blocks:
                embedding_key = f"{processed_doc.document_id}_{block.id}"

                if block.modality == ModalityType.TEXT:
                    embedding = await self.embedder.embed_text(block.content)
                elif block.modality == ModalityType.IMAGE and block.raw_data:
                    embedding = await self.embedder.embed_image(block.raw_data)
                elif block.modality == ModalityType.TABLE:
                    embedding = await self.embedder.embed_table(block.content)
                else:
                    continue

                if embedding is not None:
                    self.content_embeddings[embedding_key] = embedding

            logger.info(f"Indexed document {processed_doc.document_id} with {len(processed_doc.content_blocks)} blocks")

        except Exception as e:
            logger.error(f"Document indexing error: {e}")

    async def retrieve_multimodal_content(
        self,
        query: str,
        query_context: QueryContext,
        max_results: int = 20,
        cross_modal_threshold: float = 0.6
    ) -> MultimodalContext:
        """Retrieve relevant multimodal content"""
        try:
            # Get query embedding
            query_embedding = await self.embedder.embed_text(query)
            if query_embedding is None:
                return MultimodalContext([], [], [], [], [], {})

            # Find similar content blocks
            similar_blocks = await self._find_similar_blocks(
                query_embedding, query, max_results
            )

            # Perform cross-modal matching
            cross_modal_matches = await self._find_cross_modal_matches(
                query, similar_blocks, cross_modal_threshold
            )

            # Organize by modality
            text_blocks = [block for block in similar_blocks if block.modality == ModalityType.TEXT]
            image_blocks = [block for block in similar_blocks if block.modality == ModalityType.IMAGE]
            table_blocks = [block for block in similar_blocks if block.modality == ModalityType.TABLE]

            # Generate reasoning chain
            reasoning_chain = await self._generate_reasoning_chain(
                query, text_blocks, image_blocks, table_blocks, cross_modal_matches
            )

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                similar_blocks, cross_modal_matches
            )

            return MultimodalContext(
                text_blocks=text_blocks[:10],  # Limit results
                image_blocks=image_blocks[:5],
                table_blocks=table_blocks[:5],
                cross_modal_matches=cross_modal_matches,
                reasoning_chain=reasoning_chain,
                confidence_scores=confidence_scores
            )

        except Exception as e:
            logger.error(f"Multimodal retrieval error: {e}")
            return MultimodalContext([], [], [], [], [], {})

    async def _find_similar_blocks(
        self,
        query_embedding: np.ndarray,
        query: str,
        max_results: int
    ) -> List[ContentBlock]:
        """Find content blocks similar to query"""
        similarities = []

        try:
            for embedding_key, content_embedding in self.content_embeddings.items():
                # Calculate similarity
                if SKLEARN_AVAILABLE:
                    sim = cosine_similarity([query_embedding], [content_embedding])[0][0]
                else:
                    # Simple dot product similarity
                    sim = np.dot(query_embedding, content_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                    )

                similarities.append((embedding_key, float(sim)))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get corresponding content blocks
            similar_blocks = []
            for embedding_key, sim_score in similarities[:max_results * 2]:  # Get extra for filtering
                doc_id, block_id = embedding_key.split('_', 1)
                if doc_id in self.processed_documents:
                    doc = self.processed_documents[doc_id]
                    block = next((b for b in doc.content_blocks if b.id == block_id), None)
                    if block:
                        # Add similarity score to metadata
                        block.metadata = block.metadata or {}
                        block.metadata['query_similarity'] = sim_score
                        similar_blocks.append(block)

            return similar_blocks[:max_results]

        except Exception as e:
            logger.error(f"Similar blocks search error: {e}")
            return []

    async def _find_cross_modal_matches(
        self,
        query: str,
        content_blocks: List[ContentBlock],
        threshold: float
    ) -> List[MultimodalMatch]:
        """Find cross-modal matches for query"""
        matches = []

        try:
            for block in content_blocks:
                # Skip text-to-text matches (already covered)
                if block.modality == ModalityType.TEXT:
                    continue

                # Cross-modal similarity
                cross_modal_type = None
                similarity_score = 0.0
                reasoning_evidence = ""

                if block.modality == ModalityType.IMAGE and block.raw_data:
                    # Text-to-image matching
                    similarities = await self.embedder.compute_cross_modal_similarity(
                        query, image_data=block.raw_data
                    )

                    if CrossModalSimilarity.TEXT_TO_IMAGE in similarities:
                        similarity_score = similarities[CrossModalSimilarity.TEXT_TO_IMAGE]
                        cross_modal_type = CrossModalSimilarity.TEXT_TO_IMAGE

                        # Generate reasoning evidence
                        image_analysis = await self.visual_reasoner.analyze_image_content(block.raw_data)
                        reasoning_evidence = f"Image shows: {image_analysis.get('caption', 'visual content')}"

                elif block.modality == ModalityType.TABLE:
                    # Text-to-table matching
                    similarities = await self.embedder.compute_cross_modal_similarity(
                        query, table_content=block.content
                    )

                    if CrossModalSimilarity.TEXT_TO_TABLE in similarities:
                        similarity_score = similarities[CrossModalSimilarity.TEXT_TO_TABLE]
                        cross_modal_type = CrossModalSimilarity.TEXT_TO_TABLE
                        reasoning_evidence = f"Table contains relevant data: {block.content[:100]}..."

                # Create match if above threshold
                if similarity_score >= threshold and cross_modal_type:
                    match = MultimodalMatch(
                        content_block=block,
                        similarity_score=similarity_score,
                        cross_modal_type=cross_modal_type,
                        reasoning_evidence=reasoning_evidence,
                        confidence=similarity_score * block.confidence,
                        metadata={
                            "query": query,
                            "cross_modal_type": cross_modal_type.value
                        }
                    )
                    matches.append(match)

            # Sort by confidence
            matches.sort(key=lambda x: x.confidence, reverse=True)

        except Exception as e:
            logger.error(f"Cross-modal matching error: {e}")

        return matches[:10]  # Limit to top 10 matches

    async def _generate_reasoning_chain(
        self,
        query: str,
        text_blocks: List[ContentBlock],
        image_blocks: List[ContentBlock],
        table_blocks: List[ContentBlock],
        cross_modal_matches: List[MultimodalMatch]
    ) -> List[str]:
        """Generate reasoning chain for multimodal answer"""
        reasoning_steps = []

        try:
            # Step 1: Query analysis
            reasoning_steps.append(f"Analyzing query: '{query}'")

            # Step 2: Text evidence
            if text_blocks:
                reasoning_steps.append(f"Found {len(text_blocks)} relevant text passages")

            # Step 3: Visual evidence
            if image_blocks:
                reasoning_steps.append(f"Found {len(image_blocks)} relevant images")
                for img_block in image_blocks[:2]:  # Analyze top 2 images
                    if img_block.raw_data:
                        caption = await self.visual_reasoner.caption_image(img_block.raw_data)
                        reasoning_steps.append(f"Image analysis: {caption}")

            # Step 4: Tabular evidence
            if table_blocks:
                reasoning_steps.append(f"Found {len(table_blocks)} relevant tables")

            # Step 5: Cross-modal connections
            if cross_modal_matches:
                reasoning_steps.append(f"Cross-modal analysis:")
                for match in cross_modal_matches[:3]:  # Top 3 cross-modal matches
                    reasoning_steps.append(f"  - {match.reasoning_evidence}")

            # Step 6: Synthesis
            reasoning_steps.append("Synthesizing multimodal evidence for comprehensive answer")

        except Exception as e:
            logger.error(f"Reasoning chain generation error: {e}")
            reasoning_steps.append("Reasoning chain generation encountered an error")

        return reasoning_steps

    def _calculate_confidence_scores(
        self,
        content_blocks: List[ContentBlock],
        cross_modal_matches: List[MultimodalMatch]
    ) -> Dict[str, float]:
        """Calculate confidence scores for different modalities"""
        scores = {}

        try:
            # Text confidence
            text_blocks = [b for b in content_blocks if b.modality == ModalityType.TEXT]
            if text_blocks:
                text_similarities = [b.metadata.get('query_similarity', 0.0) for b in text_blocks]
                scores['text'] = sum(text_similarities) / len(text_similarities)

            # Image confidence
            image_blocks = [b for b in content_blocks if b.modality == ModalityType.IMAGE]
            if image_blocks:
                image_similarities = [b.metadata.get('query_similarity', 0.0) for b in image_blocks]
                scores['image'] = sum(image_similarities) / len(image_similarities)

            # Table confidence
            table_blocks = [b for b in content_blocks if b.modality == ModalityType.TABLE]
            if table_blocks:
                table_similarities = [b.metadata.get('query_similarity', 0.0) for b in table_blocks]
                scores['table'] = sum(table_similarities) / len(table_similarities)

            # Cross-modal confidence
            if cross_modal_matches:
                cross_modal_confidences = [m.confidence for m in cross_modal_matches]
                scores['cross_modal'] = sum(cross_modal_confidences) / len(cross_modal_confidences)

            # Overall confidence
            all_scores = list(scores.values())
            if all_scores:
                scores['overall'] = sum(all_scores) / len(all_scores)

        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            scores['overall'] = 0.5

        return scores

class MultimodalRAGStrategy:
    """Main multimodal RAG strategy"""

    def __init__(
        self,
        document_processor: Optional[MultimodalDocumentProcessor] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.document_processor = document_processor or create_multimodal_processor()
        self.vector_store = vector_store

        # Initialize components
        self.embedder = MultimodalEmbedder()
        self.visual_reasoner = VisualReasoner()
        self.retriever = MultimodalRetriever(self.embedder, self.visual_reasoner, vector_store)

        # Performance tracking
        self.query_times: List[float] = []
        self.modality_usage: Dict[str, int] = {
            'text': 0, 'image': 0, 'table': 0, 'cross_modal': 0
        }

    async def initialize(self, file_paths: List[str], processing_quality: ProcessingQuality = ProcessingQuality.HIGH):
        """Initialize with multimodal documents"""
        logger.info(f"Initializing MultimodalRAG with {len(file_paths)} documents")

        try:
            # Process documents
            processed_docs = await self.document_processor.process_batch(
                file_paths, processing_quality, max_concurrent=3
            )

            # Index documents for retrieval
            for doc in processed_docs:
                await self.retriever.index_document(doc)

            logger.info(f"MultimodalRAG initialized with {len(processed_docs)} processed documents")

        except Exception as e:
            logger.error(f"MultimodalRAG initialization error: {e}")
            raise

    async def search(
        self,
        query_context: QueryContext,
        reasoning_mode: ReasoningMode = ReasoningMode.MULTIMODAL_FUSION,
        max_results: int = 15
    ) -> RAGResponse:
        """Execute multimodal RAG search"""
        start_time = time.time()

        try:
            # Retrieve multimodal content
            multimodal_context = await self.retriever.retrieve_multimodal_content(
                query_context.query, query_context, max_results
            )

            # Generate multimodal response
            answer = await self._generate_multimodal_answer(
                query_context, multimodal_context, reasoning_mode
            )

            # Create sources from multimodal content
            sources = self._create_multimodal_sources(multimodal_context)

            # Calculate overall confidence
            confidence = multimodal_context.confidence_scores.get('overall', 0.5)

            # Update usage statistics
            self._update_usage_stats(multimodal_context)

            # Track performance
            query_time = time.time() - start_time
            self.query_times.append(query_time)

            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                strategy_used=RAGStrategy.MULTIMODAL_RAG,
                performance_metrics=PerformanceMetrics(
                    strategy="multimodal_rag",
                    latency=query_time,
                    quality_score=confidence,
                    context_size=len(multimodal_context.text_blocks) + len(multimodal_context.image_blocks) + len(multimodal_context.table_blocks),
                    tokens_used=len(answer.split()) if answer else 0
                ),
                metadata={
                    "reasoning_mode": reasoning_mode.value,
                    "text_blocks": len(multimodal_context.text_blocks),
                    "image_blocks": len(multimodal_context.image_blocks),
                    "table_blocks": len(multimodal_context.table_blocks),
                    "cross_modal_matches": len(multimodal_context.cross_modal_matches),
                    "reasoning_chain": multimodal_context.reasoning_chain,
                    "confidence_scores": multimodal_context.confidence_scores
                }
            )

        except Exception as e:
            logger.error(f"MultimodalRAG search error: {e}")
            return RAGResponse(
                answer=f"Multimodal search error: {str(e)}",
                sources=[],
                confidence=0.0,
                strategy_used=RAGStrategy.MULTIMODAL_RAG,
                performance_metrics=PerformanceMetrics(
                    strategy="multimodal_rag",
                    latency=time.time() - start_time,
                    quality_score=0.0,
                    context_size=0,
                    tokens_used=0
                )
            )

    async def _generate_multimodal_answer(
        self,
        query_context: QueryContext,
        multimodal_context: MultimodalContext,
        reasoning_mode: ReasoningMode
    ) -> str:
        """Generate answer from multimodal context"""
        try:
            answer_parts = []

            # Add reasoning chain if available
            if multimodal_context.reasoning_chain:
                answer_parts.append("Analysis Process:")
                answer_parts.extend([f"  - {step}" for step in multimodal_context.reasoning_chain[:3]])
                answer_parts.append("")

            # Text-based answer
            if multimodal_context.text_blocks:
                text_content = " ".join([block.content for block in multimodal_context.text_blocks[:3]])
                answer_parts.append("Text Evidence:")
                answer_parts.append(text_content[:500] + "..." if len(text_content) > 500 else text_content)
                answer_parts.append("")

            # Visual analysis
            if multimodal_context.image_blocks and reasoning_mode in [ReasoningMode.VISUAL_REASONING, ReasoningMode.MULTIMODAL_FUSION]:
                answer_parts.append("Visual Analysis:")
                for img_block in multimodal_context.image_blocks[:2]:
                    if img_block.raw_data:
                        caption = await self.visual_reasoner.caption_image(img_block.raw_data)
                        answer_parts.append(f"  - Image: {caption}")
                answer_parts.append("")

            # Table analysis
            if multimodal_context.table_blocks:
                answer_parts.append("Data Analysis:")
                for table_block in multimodal_context.table_blocks[:2]:
                    table_summary = table_block.content[:200] + "..." if len(table_block.content) > 200 else table_block.content
                    answer_parts.append(f"  - Table: {table_summary}")
                answer_parts.append("")

            # Cross-modal insights
            if multimodal_context.cross_modal_matches and reasoning_mode == ReasoningMode.MULTIMODAL_FUSION:
                answer_parts.append("Cross-Modal Insights:")
                for match in multimodal_context.cross_modal_matches[:2]:
                    answer_parts.append(f"  - {match.reasoning_evidence}")
                answer_parts.append("")

            # Synthesis
            if answer_parts:
                answer_parts.append("Summary:")
                answer_parts.append(await self._synthesize_multimodal_insights(query_context, multimodal_context))

            return "\n".join(answer_parts) if answer_parts else "No relevant multimodal content found."

        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "Error generating multimodal answer."

    async def _synthesize_multimodal_insights(
        self,
        query_context: QueryContext,
        multimodal_context: MultimodalContext
    ) -> str:
        """Synthesize insights from multimodal evidence"""
        try:
            # Count available evidence types
            evidence_types = []
            if multimodal_context.text_blocks:
                evidence_types.append(f"{len(multimodal_context.text_blocks)} text source(s)")
            if multimodal_context.image_blocks:
                evidence_types.append(f"{len(multimodal_context.image_blocks)} image(s)")
            if multimodal_context.table_blocks:
                evidence_types.append(f"{len(multimodal_context.table_blocks)} table(s)")

            synthesis = f"Based on {', '.join(evidence_types)}, "

            # Domain-specific synthesis
            if query_context.domain and hasattr(query_context.domain, 'value'):
                domain = query_context.domain.value
                if domain == 'neuroscience':
                    synthesis += "the neuroimaging and textual evidence suggests"
                elif domain == 'quantum_ml':
                    synthesis += "the quantum computing literature and visual diagrams indicate"
                else:
                    synthesis += "the multimodal evidence indicates"
            else:
                synthesis += "the combined evidence suggests"

            # Add confidence qualifier
            overall_confidence = multimodal_context.confidence_scores.get('overall', 0.5)
            if overall_confidence > 0.8:
                synthesis += " strong support for"
            elif overall_confidence > 0.6:
                synthesis += " moderate support for"
            else:
                synthesis += " limited evidence regarding"

            # Add query-specific conclusion
            synthesis += f" the query about {query_context.query.lower()}."

            return synthesis

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return "Multimodal analysis provides relevant insights for the query."

    def _create_multimodal_sources(self, multimodal_context: MultimodalContext) -> List[Dict[str, Any]]:
        """Create sources from multimodal context"""
        sources = []

        try:
            # Text sources
            for block in multimodal_context.text_blocks[:5]:
                sources.append({
                    "type": "text",
                    "content": block.content[:200] + "..." if len(block.content) > 200 else block.content,
                    "confidence": block.confidence,
                    "modality": block.modality.value,
                    "page": block.page_number
                })

            # Image sources
            for block in multimodal_context.image_blocks[:3]:
                sources.append({
                    "type": "image",
                    "content": block.content,
                    "confidence": block.confidence,
                    "modality": block.modality.value,
                    "page": block.page_number,
                    "has_visual_data": block.raw_data is not None
                })

            # Table sources
            for block in multimodal_context.table_blocks[:3]:
                sources.append({
                    "type": "table",
                    "content": block.content[:200] + "..." if len(block.content) > 200 else block.content,
                    "confidence": block.confidence,
                    "modality": block.modality.value,
                    "page": block.page_number
                })

            # Cross-modal matches
            for match in multimodal_context.cross_modal_matches[:2]:
                sources.append({
                    "type": "cross_modal",
                    "content": match.reasoning_evidence,
                    "confidence": match.confidence,
                    "cross_modal_type": match.cross_modal_type.value if match.cross_modal_type else None,
                    "similarity_score": match.similarity_score
                })

        except Exception as e:
            logger.error(f"Sources creation error: {e}")

        return sources

    def _update_usage_stats(self, multimodal_context: MultimodalContext):
        """Update modality usage statistics"""
        try:
            if multimodal_context.text_blocks:
                self.modality_usage['text'] += len(multimodal_context.text_blocks)
            if multimodal_context.image_blocks:
                self.modality_usage['image'] += len(multimodal_context.image_blocks)
            if multimodal_context.table_blocks:
                self.modality_usage['table'] += len(multimodal_context.table_blocks)
            if multimodal_context.cross_modal_matches:
                self.modality_usage['cross_modal'] += len(multimodal_context.cross_modal_matches)
        except Exception as e:
            logger.error(f"Usage stats update error: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.query_times:
            return {"error": "No query data available"}

        return {
            "avg_query_time": sum(self.query_times) / len(self.query_times),
            "total_queries": len(self.query_times),
            "modality_usage": self.modality_usage.copy(),
            "indexed_documents": len(self.retriever.processed_documents),
            "total_embeddings": len(self.retriever.content_embeddings)
        }

def create_multimodal_rag_strategy(
    vector_store: Optional[VectorStore] = None
) -> MultimodalRAGStrategy:
    """Factory function to create multimodal RAG strategy"""
    return MultimodalRAGStrategy(vector_store=vector_store)

# Example usage
if __name__ == "__main__":
    async def test_multimodal_rag():
        """Test multimodal RAG strategy"""
        strategy = create_multimodal_rag_strategy()

        # Test files (would need actual files)
        test_files = ["sample.pdf", "image.jpg", "data.csv"]

        try:
            # Initialize with test files
            await strategy.initialize(test_files, ProcessingQuality.HIGH)

            # Test query
            from ..rag.unified_rag_orchestrator import QueryContext, QueryComplexity, QueryDomain

            query_context = QueryContext(
                query="How does fMRI show brain activity patterns?",
                complexity=QueryComplexity.MEDIUM,
                domain=QueryDomain.NEUROSCIENCE,
                intent="procedural",
                confidence=0.9,
                metadata={}
            )

            # Execute search
            response = await strategy.search(query_context, ReasoningMode.MULTIMODAL_FUSION)

            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Sources: {len(response.sources)}")
            print(f"Metadata: {response.metadata}")

        except Exception as e:
            print(f"Test failed: {e}")

    # Run test
    asyncio.run(test_multimodal_rag())