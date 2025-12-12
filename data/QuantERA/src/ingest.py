"""
QuantERA QML-RAPTOR: Document Ingestion Module
Handles PDF processing, mathematical formula extraction, and multimodal content
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from datetime import datetime

import pypdf
import pdfplumber
from PIL import Image
import numpy as np
import tiktoken

# For mathematical formula handling
from pylatexenc.latex2text import LatexNodes2Text

# For text processing
from sentence_transformers import SentenceTransformer
import spacy


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    has_math: bool
    has_circuits: bool
    entity_count: int
    embedding_model: str


@dataclass
class ProcessedDocument:
    """Container for processed document data"""
    title: str
    authors: List[str]
    abstract: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    mathematical_elements: List[str]
    circuit_descriptions: List[str]
    total_pages: int
    processing_timestamp: str


class MathPreserver:
    """Handles mathematical formula preservation and extraction"""

    def __init__(self):
        self.latex_converter = LatexNodes2Text()
        # Common LaTeX math patterns
        self.math_patterns = [
            r'\$\$[^$]*\$\$',  # Display math
            r'\$[^$]*\$',      # Inline math
            r'\\begin\{equation\}.*?\\end\{equation\}',  # Equations
            r'\\begin\{align\}.*?\\end\{align\}',        # Align environments
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',  # Eqnarray
        ]

    def extract_math_elements(self, text: str) -> Tuple[str, List[str]]:
        """Extract mathematical elements and replace with placeholders"""
        math_elements = []
        processed_text = text

        for i, pattern in enumerate(self.math_patterns):
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                math_content = match.group(0)
                math_elements.append(math_content)
                placeholder = f"[MATH_{len(math_elements)}]"
                processed_text = processed_text.replace(math_content, placeholder, 1)

        return processed_text, math_elements

    def restore_math_elements(self, text: str, math_elements: List[str]) -> str:
        """Restore mathematical elements from placeholders"""
        restored_text = text
        for i, math_element in enumerate(math_elements, 1):
            placeholder = f"[MATH_{i}]"
            restored_text = restored_text.replace(placeholder, math_element)
        return restored_text


class CircuitDetector:
    """Detects and describes quantum circuit elements in text"""

    def __init__(self):
        # Quantum gate patterns
        self.gate_patterns = {
            'hadamard': r'\b(H|Hadamard)\b',
            'pauli_x': r'\b(X|Pauli-X|NOT)\b',
            'pauli_y': r'\b(Y|Pauli-Y)\b',
            'pauli_z': r'\b(Z|Pauli-Z)\b',
            'cnot': r'\b(CNOT|CX|controlled-X)\b',
            'cz': r'\b(CZ|controlled-Z)\b',
            'rz': r'\b(RZ|R_Z)\b',
            'ry': r'\b(RY|R_Y)\b',
            'rx': r'\b(RX|R_X)\b',
            'toffoli': r'\b(Toffoli|CCX)\b',
            'fredkin': r'\b(Fredkin|CSWAP)\b',
            'measurement': r'\b(measure|measurement|readout)\b'
        }

        # Circuit description patterns
        self.circuit_patterns = [
            r'quantum\s+circuit',
            r'circuit\s+diagram',
            r'gate\s+sequence',
            r'quantum\s+algorithm',
            r'ansatz',
            r'variational\s+circuit',
            r'parameterized\s+circuit'
        ]

    def detect_circuit_elements(self, text: str) -> List[str]:
        """Detect quantum circuit elements in text"""
        detected_elements = []

        # Check for gate patterns
        for gate_name, pattern in self.gate_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_elements.append(f"Gate: {gate_name}")

        # Check for circuit descriptions
        for pattern in self.circuit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                detected_elements.append(f"Circuit: {match}")

        return detected_elements


class MathAwareChunker:
    """Chunks text while preserving mathematical boundaries"""

    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, math_elements: List[str]) -> List[str]:
        """Chunk text while respecting mathematical boundaries"""
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_token_count + sentence_tokens > self.chunk_size and current_chunk:
                # If current sentence contains math placeholder, try to keep it together
                if "[MATH_" in sentence and sentence_tokens < self.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_token_count = sentence_tokens
                else:
                    # Add current chunk and start new one
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_token_count = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_token_count += sentence_tokens

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


class EntityExtractor:
    """Extracts quantum ML entities from text"""

    def __init__(self):
        # Load spaCy model (fallback to basic if not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
            logging.warning("spaCy model not found. Entity extraction will be limited.")

        # Quantum ML specific terms
        self.qml_entities = {
            'algorithms': ['VQE', 'QAOA', 'QSVM', 'QNN', 'QGaN', 'QGAN'],
            'concepts': ['barren plateau', 'ansatz', 'variational', 'parameterized',
                        'quantum advantage', 'quantum supremacy', 'NISQ'],
            'hardware': ['superconducting', 'ion trap', 'photonic', 'topological'],
            'metrics': ['fidelity', 'gate fidelity', 'coherence time', 'error rate']
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {category: [] for category in self.qml_entities}

        # Extract using predefined patterns
        for category, terms in self.qml_entities.items():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    entities[category].append(term)

        # Use spaCy for additional entity extraction if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE"]:
                    entities.setdefault('named_entities', []).append(ent.text)

        return entities


class QuantERAIngestor:
    """Main ingestion class for QuantERA papers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.math_handler = MathPreserver()
        self.circuit_detector = CircuitDetector()
        self.chunker = MathAwareChunker(
            chunk_size=self.config.get('chunk_size', 1500),
            overlap=self.config.get('overlap', 200)
        )
        self.entity_extractor = EntityExtractor()

        # Initialize embedding model if specified
        self.embedding_model = None
        if self.config.get('embedding_model'):
            try:
                self.embedding_model = SentenceTransformer(
                    self.config['embedding_model']
                )
            except Exception as e:
                self.logger.warning(f"Could not load embedding model: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'chunk_size': 1500,
            'overlap': 200,
            'preserve_math': True,
            'extract_circuits': True,
            'embedding_model': 'all-MiniLM-L6-v2'
        }

    def _generate_chunk_id(self, source_file: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        content = f"{source_file}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def extract_metadata_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {
            'title': '',
            'authors': [],
            'subject': '',
            'creation_date': None,
            'page_count': 0
        }

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata['page_count'] = len(pdf_reader.pages)

                if pdf_reader.metadata:
                    metadata['title'] = pdf_reader.metadata.get('/Title', '')
                    metadata['subject'] = pdf_reader.metadata.get('/Subject', '')
                    metadata['creation_date'] = pdf_reader.metadata.get('/CreationDate', '')

                    # Extract authors (simplified)
                    author_field = pdf_reader.metadata.get('/Author', '')
                    if author_field:
                        metadata['authors'] = [
                            name.strip() for name in author_field.split(',')
                        ]

        except Exception as e:
            self.logger.warning(f"Could not extract PDF metadata: {e}")

        return metadata

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """Extract text from PDF pages"""
        pages_text = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text() or ""
                        pages_text.append((page_num, text))
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num}: {e}")
                        pages_text.append((page_num, ""))

        except Exception as e:
            self.logger.error(f"Could not open PDF {pdf_path}: {e}")
            return []

        return pages_text

    def process_paper(self, pdf_path: str) -> ProcessedDocument:
        """Process a single research paper"""
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.logger.info(f"Processing paper: {pdf_path.name}")

        # Extract metadata
        metadata = self.extract_metadata_from_pdf(pdf_path)

        # Extract text from all pages
        pages_text = self.extract_text_from_pdf(pdf_path)
        full_text = " ".join([text for _, text in pages_text])

        # Extract mathematical elements
        if self.config.get('preserve_math', True):
            processed_text, math_elements = self.math_handler.extract_math_elements(full_text)
        else:
            processed_text = full_text
            math_elements = []

        # Detect circuit elements
        circuit_descriptions = []
        if self.config.get('extract_circuits', True):
            circuit_descriptions = self.circuit_detector.detect_circuit_elements(processed_text)

        # Chunk the text
        chunks_text = self.chunker.chunk_text(processed_text, math_elements)

        # Process each chunk
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            # Restore math elements
            if math_elements:
                chunk_text = self.math_handler.restore_math_elements(chunk_text, math_elements)

            # Extract entities
            entities = self.entity_extractor.extract_entities(chunk_text)

            # Generate embedding if model available
            embedding = None
            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(chunk_text).tolist()
                except Exception as e:
                    self.logger.warning(f"Could not generate embedding for chunk {i}: {e}")

            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(str(pdf_path), i),
                source_file=str(pdf_path),
                page_number=0,  # TODO: Map chunks to specific pages
                chunk_index=i,
                has_math=bool(re.search(r'\[MATH_\d+\]', chunk_text)),
                has_circuits=bool(circuit_descriptions),
                entity_count=sum(len(ents) for ents in entities.values()),
                embedding_model=self.config.get('embedding_model', 'none')
            )

            chunk_data = {
                'text': chunk_text,
                'metadata': chunk_metadata.__dict__,
                'entities': entities,
                'embedding': embedding
            }

            chunks.append(chunk_data)

        # Extract abstract (first few meaningful sentences)
        sentences = re.split(r'(?<=[.!?])\s+', processed_text)
        abstract_sentences = [s for s in sentences[:5] if len(s.split()) > 10]
        abstract = " ".join(abstract_sentences[:3])

        # Create processed document
        processed_doc = ProcessedDocument(
            title=metadata.get('title', pdf_path.stem),
            authors=metadata.get('authors', []),
            abstract=abstract,
            chunks=chunks,
            metadata=metadata,
            mathematical_elements=math_elements,
            circuit_descriptions=circuit_descriptions,
            total_pages=metadata.get('page_count', 0),
            processing_timestamp=str(datetime.now().isoformat())
        )

        self.logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks, "
                        f"{len(math_elements)} math elements, "
                        f"{len(circuit_descriptions)} circuit elements")

        return processed_doc

    def process_directory(self, directory_path: str, output_file: Optional[str] = None) -> List[ProcessedDocument]:
        """Process all PDFs in a directory"""
        directory_path = Path(directory_path)
        processed_docs = []

        pdf_files = list(directory_path.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            try:
                processed_doc = self.process_paper(str(pdf_file))
                processed_docs.append(processed_doc)
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file.name}: {e}")

        # Save results if output file specified
        if output_file and processed_docs:
            self._save_processed_docs(processed_docs, output_file)

        return processed_docs

    def _save_processed_docs(self, docs: List[ProcessedDocument], output_file: str):
        """Save processed documents to JSON file"""
        # Convert to serializable format
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

        self.logger.info(f"Saved {len(docs)} processed documents to {output_file}")


def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description="QuantERA Document Ingestion")
    parser.add_argument("--paper", help="Path to single PDF paper")
    parser.add_argument("--directory", help="Path to directory containing PDFs")
    parser.add_argument("--output", help="Output file for processed documents")
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Initialize ingestor
    ingestor = QuantERAIngestor(config)

    if args.paper:
        # Process single paper
        processed_doc = ingestor.process_paper(args.paper)
        print(f"Processed: {processed_doc.title}")
        print(f"Chunks: {len(processed_doc.chunks)}")
        print(f"Math elements: {len(processed_doc.mathematical_elements)}")

    elif args.directory:
        # Process directory
        processed_docs = ingestor.process_directory(args.directory, args.output)
        print(f"Processed {len(processed_docs)} documents")

    else:
        print("Please specify --paper or --directory")


if __name__ == "__main__":
    main()