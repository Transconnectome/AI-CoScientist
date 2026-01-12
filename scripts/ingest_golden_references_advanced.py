#!/usr/bin/env python3
"""
Advanced Golden Reference Ingestion with RAPTOR and Multi-Provider LLM Support.

Implements state-of-the-art RAG techniques:
- RAPTOR: Hierarchical 3-level indexing
- Section-aware chunking with overlap
- Multi-provider LLM support (OpenAI GPT-4o, Claude 4.5, Gemini 2.5, DeepSeek R1)
- Rich metadata extraction
- SciBERT embeddings
- ChromaDB storage

Usage:
    poetry run python scripts/ingest_golden_references_advanced.py --test  # 5 papers
    poetry run python scripts/ingest_golden_references_advanced.py --all   # all 53 papers
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import PyPDF2
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from tqdm import tqdm
import argparse

# Import LLM providers
import anthropic
import openai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class LLMProvider(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


# Latest best models from each provider (Nov 2025)
PROVIDER_MODELS = {
    LLMProvider.OPENAI: "gpt-3.5-turbo",  # Safest bet for accessibility
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.GEMINI: "gemini-3-pro-preview",  # User requested upgrade from Flash
    LLMProvider.DEEPSEEK: "deepseek-chat"
}

# Provider priority order (fallback chain)
PROVIDER_PRIORITY = [
    LLMProvider.GEMINI,     # User requested priority
    LLMProvider.ANTHROPIC,  # Best for reasoning and code
    LLMProvider.OPENAI,     # Excellent all-around
    LLMProvider.DEEPSEEK    # Fast and cost-effective
]


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Section:
    """Paper section with content and metadata."""
    name: str
    content: str
    order: int
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class Chunk:
    """Text chunk with metadata."""
    chunk_id: str
    content: str
    section: str
    chunk_index: int
    total_chunks: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAPTORNode:
    """Node in RAPTOR hierarchical tree."""
    node_id: str
    content: str
    level: int  # 0=chunk, 1=section_summary, 2=paper_summary
    embedding: Optional[np.ndarray] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class GoldenReferencePaper:
    """Complete paper data for ingestion."""
    paper_id: str
    filename: str
    title: str
    journal: str
    year: int

    # Extracted sections
    sections: List[Section] = field(default_factory=list)

    # RAPTOR hierarchy
    level0_chunks: List[Chunk] = field(default_factory=list)  # Original chunks
    level1_summaries: List[RAPTORNode] = field(default_factory=list)  # Section summaries
    level2_summary: Optional[RAPTORNode] = None  # Paper summary

    # Metadata
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    total_words: int = 0
    full_text: str = ""


# ============================================================================
# PDF Extraction
# ============================================================================

class PDFExtractor:
    """Extract text from PDF files."""

    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extract text from PDF."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

                full_text = '\n\n'.join(text_parts)

                # Clean up
                full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Multiple newlines
                full_text = re.sub(r' {2,}', ' ', full_text)  # Multiple spaces

                return full_text

        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return ""

    @staticmethod
    def estimate_metadata(filename: str, text: str) -> Dict:
        """Estimate metadata from filename and text."""
        # Extract year from filename or text
        year_match = re.search(r'20[12][0-9]', filename)
        year = int(year_match.group()) if year_match else 2024

        # Detect journal from filename
        filename_lower = filename.lower()
        if 'nature_medicine' in filename_lower or 's41591' in filename_lower:
            journal = "Nature Medicine"
        elif 'nature_biomedical' in filename_lower or 's41551' in filename_lower:
            journal = "Nature Biomedical Engineering"
        elif 'nature_human' in filename_lower or 's41562' in filename_lower:
            journal = "Nature Human Behaviour"
        elif 'science' in filename_lower and 'nature' not in filename_lower:
            journal = "Science"
        else:
            journal = "Nature"

        # Extract title (first non-empty line)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        title = lines[0][:100] if lines else filename[:60]

        return {
            'title': title,
            'journal': journal,
            'year': year
        }


# ============================================================================
# Multi-Provider LLM Interface
# ============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.1) -> str:
        """Generate text from prompt."""
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model_name: str = PROVIDER_MODELS[LLMProvider.ANTHROPIC]):
        super().__init__(api_key, model_name)
        self.client = anthropic.Anthropic(api_key=api_key)

    async def generate(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.1) -> str:
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str, model_name: str = PROVIDER_MODELS[LLMProvider.OPENAI]):
        super().__init__(api_key, model_name)
        openai.api_key = api_key

    async def generate(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.1) -> str:
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model_name: str = PROVIDER_MODELS[LLMProvider.GEMINI]):
        super().__init__(api_key, model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Configure safety settings to be permissive for scientific content
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    async def generate(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.1) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature
                },
                safety_settings=self.safety_settings
            )
            
            # Check if response has content
            if not response.parts:
                if response.prompt_feedback:
                    raise RuntimeError(f"Gemini blocked prompt: {response.prompt_feedback}")
                raise RuntimeError(f"Gemini returned empty response. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider (OpenAI-compatible API)."""

    def __init__(self, api_key: str, model_name: str = PROVIDER_MODELS[LLMProvider.DEEPSEEK]):
        super().__init__(api_key, model_name)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    async def generate(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.1) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"DeepSeek API error: {e}")


class MultiProviderLLM:
    """Multi-provider LLM with automatic fallback."""

    def __init__(self):
        """Initialize all available providers."""
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}

        # Load API keys from environment
        api_keys = {
            LLMProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
            LLMProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
            LLMProvider.GEMINI: os.getenv("GOOGLE_API_KEY"),
            LLMProvider.DEEPSEEK: os.getenv("DEEPSEEK_API_KEY")
        }

        # Initialize available providers
        provider_classes = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.ANTHROPIC: AnthropicProvider,
            LLMProvider.GEMINI: GeminiProvider,
            LLMProvider.DEEPSEEK: DeepSeekProvider
        }

        for provider_type, api_key in api_keys.items():
            if api_key:
                try:
                    provider_class = provider_classes[provider_type]
                    self.providers[provider_type] = provider_class(api_key)
                    print(f"  ✓ Initialized {provider_type.value}: {PROVIDER_MODELS[provider_type]}")
                except Exception as e:
                    print(f"  ⚠️  Failed to initialize {provider_type.value}: {e}")

        if not self.providers:
            raise RuntimeError("No LLM providers available! Please set API keys in .env")

        print(f"\n  Total providers available: {len(self.providers)}/{len(PROVIDER_PRIORITY)}")

    async def generate(self, prompt: str, max_tokens: int = 8000, temperature: float = 0.1) -> Tuple[str, LLMProvider]:
        """Generate text with automatic provider fallback and retries."""
        errors = []
        
        import time

        for provider_type in PROVIDER_PRIORITY:
            if provider_type not in self.providers:
                continue

            # Retry logic
            max_retries = 3  # Increased for better reliability
            base_delay = 2
            
            for attempt in range(max_retries):
                try:
                    result = await self.providers[provider_type].generate(prompt, max_tokens, temperature)
                    return result, provider_type
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ⚠️  DEBUG: Full error from {provider_type.value}: {e}")
                    
                    # Check for rate limits or temporary errors
                    is_rate_limit = "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower()
                    is_server_error = "500" in error_msg or "503" in error_msg
                    
                    if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  ⚠️  {provider_type.value} error (attempt {attempt+1}/{max_retries}): {error_msg[:100]}... Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    
                    # If not retryable or max retries reached, move to next provider
                    full_error = f"{provider_type.value}: {error_msg}"
                    errors.append(full_error)
                    print(f"  ⚠️  {full_error[:100]}")
                    break

        # All providers failed
        raise RuntimeError(f"All LLM providers failed:\n" + "\n".join(errors))


# ============================================================================
# LLM-based Section Parser (Multi-Provider)
# ============================================================================

class SectionParser:
    """Parse papers into sections using multi-provider LLM."""

    def __init__(self, llm: MultiProviderLLM):
        self.llm = llm

    async def parse_sections(self, text: str, title: str) -> List[Section]:
        """Parse paper into sections using LLM to find headers, then split text."""

        # Truncate if too long for context (Gemini 2.5 has 1M+ context, so this is generous)
        max_chars = 500000 
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"

        prompt = f"""Analyze this academic paper and identify the exact section headers used in the text.
        
Paper title: {title}

Return a JSON list of strings, where each string is an EXACT match for a section header found in the text.
Include standard sections like Abstract, Introduction, Methods, Results, Discussion, Conclusion, References.
Do NOT include subsections.
Do NOT include the content, ONLY the headers.

Example format:
["Abstract", "1. Introduction", "2. Materials and Methods", "3. Results", "Discussion", "References"]

Paper text (first 20000 chars):
{text[:20000]}...

Return ONLY the JSON list."""

        try:
            response_text, provider = await self.llm.generate(prompt, max_tokens=1000, temperature=0.1)

            # Extract JSON
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            headers = json.loads(response_text)
            
            # Split text by headers
            sections = []
            
            # Find header positions
            header_positions = []
            lower_text = text.lower()
            
            for header in headers:
                # Simple find (could be improved with regex)
                # Try exact match first
                pos = text.find(header)
                if pos == -1:
                    # Try case-insensitive
                    pos = lower_text.find(header.lower())
                
                if pos != -1:
                    header_positions.append((pos, header))
            
            # Sort by position
            header_positions.sort()
            
            # Create sections
            for i, (pos, header) in enumerate(header_positions):
                start = pos
                end = header_positions[i+1][0] if i < len(header_positions) - 1 else len(text)
                
                # Content includes header, so we might want to strip it or keep it
                # Let's keep it but maybe clean up
                content = text[start:end].strip()
                
                # Map to standard names
                name = header.lower()
                order = i
                
                # Simple normalization
                if 'abstract' in name: normalized_name = 'abstract'
                elif 'intro' in name: normalized_name = 'introduction'
                elif 'method' in name: normalized_name = 'methods'
                elif 'result' in name: normalized_name = 'results'
                elif 'discuss' in name: normalized_name = 'discussion'
                elif 'conclus' in name: normalized_name = 'conclusion'
                elif 'referen' in name: normalized_name = 'references'
                else: normalized_name = name[:50] # Custom section
                
                if len(content) > 50:
                    sections.append(Section(
                        name=normalized_name,
                        content=content,
                        order=order
                    ))
            
            if not sections:
                # Fallback if splitting failed
                return [Section(name="full_text", content=text, order=0)]
                
            return sections

        except Exception as e:
            print(f"  ⚠️  Section parsing failed: {e}")
            # Fallback: create single section
            return [Section(name="full_text", content=text, order=0)]

    async def generate_section_summary(self, section: Section, paper_title: str) -> str:
        """Generate summary of a section."""

        prompt = f"""Summarize this section from the paper "{paper_title}".

Section: {section.name}

Create a concise 2-3 sentence summary that captures the key points.

Section content:
{section.content[:3000]}

Return only the summary, no additional text."""

        try:
            summary, provider = await self.llm.generate(prompt, max_tokens=500, temperature=0.3)
            return summary.strip()

        except Exception as e:
            print(f"  ⚠️  Summary generation failed: {e}")
            # Fallback: first 200 chars
            return section.content[:200] + "..."

    async def generate_paper_summary(self, sections: List[Section], title: str) -> str:
        """Generate overall paper summary."""

        # Combine section summaries
        section_texts = []
        for section in sections[:5]:  # Top 5 sections
            section_texts.append(f"{section.name.title()}: {section.content[:500]}")

        combined = "\n\n".join(section_texts)

        prompt = f"""Create a comprehensive summary of this paper: "{title}"

Key sections:
{combined}

Generate a 3-4 sentence summary that captures:
1. Main research question or problem
2. Key methodology or approach
3. Primary findings or contributions
4. Significance or implications

Return only the summary, no additional text."""

        try:
            summary, provider = await self.llm.generate(prompt, max_tokens=800, temperature=0.3)
            return summary.strip()

        except Exception as e:
            print(f"  ⚠️  Paper summary failed: {e}")
            # Fallback: first section beginning
            if sections:
                return sections[0].content[:300] + "..."
            return f"Paper: {title}"


# ============================================================================
# Section-Aware Chunker
# ============================================================================

class SectionAwareChunker:
    """Section-aware chunking with overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_section(self, section: Section, paper_id: str) -> List[Chunk]:
        """Chunk a section with overlap."""

        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', section.content)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Estimate tokens (rough: 4 chars per token)
            sentence_tokens = len(sentence) // 4

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{paper_id}_{section.name}_{chunk_index}"

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    section=section.name,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will update later
                    metadata={
                        'section': section.name,
                        'section_order': section.order,
                        'paper_id': paper_id
                    }
                ))

                chunk_index += 1

                # Keep last 2 sentences as overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(s) // 4 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{paper_id}_{section.name}_{chunk_index}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=chunk_text,
                section=section.name,
                chunk_index=chunk_index,
                total_chunks=0,
                metadata={
                    'section': section.name,
                    'section_order': section.order,
                    'paper_id': paper_id
                }
            ))

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks


# ============================================================================
# RAPTOR Hierarchy Builder
# ============================================================================

class RAPTORBuilder:
    """Build RAPTOR hierarchical tree structure."""

    def __init__(self, section_parser: SectionParser):
        self.section_parser = section_parser

    async def build_hierarchy(
        self,
        chunks: List[Chunk],
        sections: List[Section],
        paper_title: str,
        paper_id: str
    ) -> Tuple[List[RAPTORNode], Optional[RAPTORNode]]:
        """
        Build 3-level RAPTOR hierarchy.

        Returns:
            level1_summaries: Section-level summaries
            level2_summary: Paper-level summary
        """

        print("  Building RAPTOR hierarchy...")

        # Level 1: Section summaries
        level1_summaries = []
        for section in sections:
            try:
                summary_text = await self.section_parser.generate_section_summary(section, paper_title)

                node = RAPTORNode(
                    node_id=f"{paper_id}_L1_{section.name}",
                    content=summary_text,
                    level=1,
                    parent_id=None,  # Will be set when L2 is created
                    children_ids=[
                        chunk.chunk_id for chunk in chunks
                        if chunk.section == section.name
                    ],
                    metadata={
                        'section': section.name,
                        'section_order': section.order,
                        'paper_id': paper_id,
                        'paper_title': paper_title
                    }
                )

                level1_summaries.append(node)
                print(f"    ✓ L1: {section.name} summary")

            except Exception as e:
                print(f"    ⚠️  Failed L1 for {section.name}: {e}")

        # Level 2: Paper summary
        level2_summary = None
        try:
            paper_summary_text = await self.section_parser.generate_paper_summary(sections, paper_title)

            level2_summary = RAPTORNode(
                node_id=f"{paper_id}_L2_paper",
                content=paper_summary_text,
                level=2,
                parent_id=None,
                children_ids=[node.node_id for node in level1_summaries],
                metadata={
                    'paper_id': paper_id,
                    'paper_title': paper_title,
                    'total_sections': len(sections),
                    'total_chunks': len(chunks)
                }
            )

            # Update L1 parent references
            for node in level1_summaries:
                node.parent_id = level2_summary.node_id

            print(f"    ✓ L2: Paper summary")

        except Exception as e:
            print(f"    ⚠️  Failed L2: {e}")

        return level1_summaries, level2_summary


# ============================================================================
# Advanced Golden Reference Ingestor
# ============================================================================

class AdvancedGoldenReferenceIngestor:
    """Advanced ingestion pipeline with RAPTOR and multi-provider LLM."""

    def __init__(self, chromadb_path: str = "chromadb_data"):
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.llm = MultiProviderLLM()
        self.section_parser = SectionParser(self.llm)
        self.chunker = SectionAwareChunker(chunk_size=512, overlap=50)
        self.raptor_builder = RAPTORBuilder(self.section_parser)

        # Initialize embedding model
        print("Loading SciBERT embedding model...")
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("✓ SciBERT loaded")

        # Initialize ChromaDB
        print(f"Initializing ChromaDB at {chromadb_path}...")
        self.chroma_client = chromadb.PersistentClient(path=chromadb_path)

        # Create collections for each RAPTOR level
        self.collection_l0 = self.chroma_client.get_or_create_collection(
            name="golden_references_advanced_L0",
            metadata={"description": "Level 0: Original chunks"}
        )
        self.collection_l1 = self.chroma_client.get_or_create_collection(
            name="golden_references_advanced_L1",
            metadata={"description": "Level 1: Section summaries"}
        )
        self.collection_l2 = self.chroma_client.get_or_create_collection(
            name="golden_references_advanced_L2",
            metadata={"description": "Level 2: Paper summaries"}
        )
        print("✓ ChromaDB initialized\n")

    async def process_paper(self, pdf_path: Path) -> Optional[GoldenReferencePaper]:
        """Process a single paper through the complete pipeline."""

        print("=" * 70)
        print(f"Processing: {pdf_path.name}")
        print("=" * 70)

        # 1. Extract PDF text
        print("1. Extracting PDF text...")
        full_text = self.pdf_extractor.extract_text(pdf_path)

        if not full_text or len(full_text) < 500:
            print(f"  ✗ No text extracted from {pdf_path.name}")
            return None

        print(f"  ✓ Extracted {len(full_text):,} characters")

        # Get metadata
        metadata = self.pdf_extractor.estimate_metadata(pdf_path.name, full_text)
        print(f"  Title: {metadata['title']}")
        print(f"  Journal: {metadata['journal']} ({metadata['year']})")

        paper_id = pdf_path.stem

        # 2. Parse sections with LLM
        print("2. Parsing sections with LLM...")
        sections = await self.section_parser.parse_sections(full_text, metadata['title'])
        print(f"  ✓ Found {len(sections)} sections:")
        for section in sections:
            print(f"    - {section.name}: {section.word_count} words")

        # 3. Chunk sections
        print("3. Chunking sections...")
        all_chunks = []
        for section in sections:
            chunks = self.chunker.chunk_section(section, paper_id)
            all_chunks.extend(chunks)
            print(f"  - {section.name}: {len(chunks)} chunks")

        print(f"  ✓ Total chunks: {len(all_chunks)}")

        # 4. Build RAPTOR hierarchy
        print("4. Building RAPTOR hierarchy...")
        level1_summaries, level2_summary = await self.raptor_builder.build_hierarchy(
            all_chunks, sections, metadata['title'], paper_id
        )

        # 5. Generate embeddings
        print("5. Generating embeddings...")

        # L0 embeddings
        l0_texts = [chunk.content for chunk in all_chunks]
        l0_embeddings = self.embedding_model.encode(l0_texts, show_progress_bar=False)
        for chunk, embedding in zip(all_chunks, l0_embeddings):
            chunk.embedding = embedding

        # L1 embeddings
        l1_texts = [node.content for node in level1_summaries]
        if l1_texts:
            l1_embeddings = self.embedding_model.encode(l1_texts, show_progress_bar=False)
            for node, embedding in zip(level1_summaries, l1_embeddings):
                node.embedding = embedding

        # L2 embedding
        if level2_summary:
            l2_embedding = self.embedding_model.encode([level2_summary.content], show_progress_bar=False)[0]
            level2_summary.embedding = l2_embedding

        print("  ✓ Generated embeddings for all levels")

        # 6. Store in ChromaDB
        print("6. Storing in ChromaDB...")

        # Store L0
        if all_chunks:
            self.collection_l0.upsert(
                ids=[chunk.chunk_id for chunk in all_chunks],
                embeddings=[chunk.embedding.tolist() for chunk in all_chunks],
                documents=[chunk.content for chunk in all_chunks],
                metadatas=[chunk.metadata for chunk in all_chunks]
            )
        
        # Store L1
        if level1_summaries:
            self.collection_l1.upsert(
                ids=[node.node_id for node in level1_summaries],
                embeddings=[node.embedding.tolist() for node in level1_summaries],
                documents=[node.content for node in level1_summaries],
                metadatas=[node.metadata for node in level1_summaries]
            )
        
        # Store L2
        if level2_summary:
            self.collection_l2.upsert(
                ids=[level2_summary.node_id],
                embeddings=[level2_summary.embedding.tolist()],
                documents=[level2_summary.content],
                metadatas=[level2_summary.metadata]
            )

        print("  ✓ Stored in ChromaDB\n")

        # Create GoldenReferencePaper object
        paper = GoldenReferencePaper(
            paper_id=paper_id,
            filename=pdf_path.name,
            title=metadata['title'],
            journal=metadata['journal'],
            year=metadata['year'],
            sections=sections,
            level0_chunks=all_chunks,
            level1_summaries=level1_summaries,
            level2_summary=level2_summary,
            full_text=full_text,
            total_words=len(full_text.split())
        )

        return paper

    async def ingest_all(self, pdf_dir: Path, limit: Optional[int] = None):
        """Ingest all PDFs from directory."""

        pdfs = sorted(pdf_dir.glob("*.pdf"))

        if limit:
            pdfs = pdfs[:limit]
            print(f"Test mode: processing {limit} papers")
        else:
            print(f"Processing all {len(pdfs)} papers")

        print("\n" + "=" * 70)
        print("ADVANCED GOLDEN REFERENCE INGESTION")
        print("=" * 70)
        print(f"Total PDFs: {len(pdfs)}")
        print(f"Collection: golden_references_advanced")
        print("=" * 70)
        print("\n")

        results = {
            'success': [],
            'failed': [],
            'total_chunks_l0': 0,
            'total_chunks_l1': 0,
            'total_chunks_l2': 0
        }

        for idx, pdf_path in enumerate(pdfs, 1):
            print(f"[{idx}/{len(pdfs)}]\n")

            try:
                paper = await self.process_paper(pdf_path)

                if paper:
                    results['success'].append(pdf_path.name)
                    results['total_chunks_l0'] += len(paper.level0_chunks)
                    results['total_chunks_l1'] += len(paper.level1_summaries)
                    results['total_chunks_l2'] += (1 if paper.level2_summary else 0)

                    print(f"✅ Successfully processed: {paper.title}\n")
                else:
                    results['failed'].append(pdf_path.name)
                    print(f"✗ Failed to process: {pdf_path.name}\n")

            except Exception as e:
                print(f"✗ Error processing {pdf_path.name}: {e}\n")
                results['failed'].append(pdf_path.name)

            print(f"Progress: {len(results['success'])}/{len(pdfs)} completed\n")

        # Print summary
        print("=" * 70)
        print("INGESTION COMPLETE")
        print("=" * 70)
        print(f"✅ Success: {len(results['success'])}/{len(pdfs)}")
        print(f"✗ Failed: {len(results['failed'])}/{len(pdfs)}")
        print(f"\nChunks created:")
        print(f"  Level 0 (chunks): {results['total_chunks_l0']}")
        print(f"  Level 1 (sections): {results['total_chunks_l1']}")
        print(f"  Level 2 (papers): {results['total_chunks_l2']}")
        print(f"  Total: {results['total_chunks_l0'] + results['total_chunks_l1'] + results['total_chunks_l2']}")

        # Save results
        results_path = Path("data/reference_papers/ingestion_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_path}")

        print("\n" + "=" * 70)
        print("All done! 🎉")
        print("=" * 70)


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Advanced Golden Reference Ingestion")
    parser.add_argument("--test", action="store_true", help="Test mode: process 5 papers only")
    parser.add_argument("--all", action="store_true", help="Process all papers")
    args = parser.parse_args()

    # Default to test mode if no args
    if not args.test and not args.all:
        args.test = True

    pdf_dir = Path("data/reference_papers/pdfs")

    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return

    ingestor = AdvancedGoldenReferenceIngestor()

    if args.test:
        await ingestor.ingest_all(pdf_dir, limit=5)
    else:
        await ingestor.ingest_all(pdf_dir, limit=None)


if __name__ == "__main__":
    asyncio.run(main())
