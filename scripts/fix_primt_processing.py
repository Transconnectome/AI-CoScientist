#!/usr/bin/env python3
"""
Fixed PRIMT paper processing with robust text handling
Addresses the TextEncodeInput tokenization error encountered during DD-RAPTOR processing
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import os
from pathlib import Path
import PyPDF2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustPRIMTProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')

    def robust_text_cleaning(self, text):
        """Robust text cleaning to prevent tokenization errors"""
        if not text or not isinstance(text, str):
            return ""

        # Handle Unicode surrogate issues and normalize text
        import re
        import unicodedata

        # Remove surrogates and other problematic Unicode characters
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = unicodedata.normalize('NFKD', text)

        # Remove control characters and normalize whitespace
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Ensure minimum length and maximum length
        if len(text) < 10:
            return ""
        if len(text) > 500:  # Truncate very long chunks
            text = text[:500] + "..."

        return text

    def extract_pdf_text_robust(self, pdf_path):
        """Extract text from PDF with error handling"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\nPage {page_num + 1}:\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue

                if not text.strip():
                    raise ValueError("No text extracted from PDF")

                return text

        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise

    def create_robust_chunks(self, text, chunk_size=300, overlap=50):
        """Create text chunks with robust handling"""
        if not text or len(text) < 50:
            return []

        # Split into sentences first
        import re
        sentences = re.split(r'[.!?]+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                cleaned_chunk = self.robust_text_cleaning(current_chunk)
                if cleaned_chunk:  # Only add non-empty chunks
                    chunks.append(cleaned_chunk)

                # Start new chunk with overlap
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add the last chunk
        if current_chunk:
            cleaned_chunk = self.robust_text_cleaning(current_chunk)
            if cleaned_chunk:
                chunks.append(cleaned_chunk)

        return chunks

    def safe_embedding_generation(self, texts):
        """Generate embeddings with error handling for problematic texts"""
        if not texts:
            return np.array([])

        valid_texts = []
        for text in texts:
            cleaned_text = self.robust_text_cleaning(text)
            if cleaned_text:
                valid_texts.append(cleaned_text)

        if not valid_texts:
            logger.warning("No valid texts for embedding generation")
            return np.array([])

        try:
            embeddings = self.embedding_model.encode(valid_texts, show_progress_bar=False)
            logger.info(f"Successfully generated embeddings for {len(valid_texts)} text chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Try with shorter texts
            try:
                short_texts = [text[:200] for text in valid_texts]
                embeddings = self.embedding_model.encode(short_texts, show_progress_bar=False)
                logger.info(f"Generated embeddings with truncated texts: {len(short_texts)} chunks")
                return embeddings
            except Exception as e2:
                logger.error(f"Even truncated embedding generation failed: {e2}")
                return np.array([])

    def process_primt_paper(self, pdf_path, output_path):
        """Process the PRIMT paper with robust error handling"""
        try:
            logger.info("Starting PRIMT paper processing...")

            # Extract text
            logger.info("Extracting PDF text...")
            text = self.extract_pdf_text_robust(pdf_path)

            # Create chunks
            logger.info("Creating text chunks...")
            chunks = self.create_robust_chunks(text)
            logger.info(f"Created {len(chunks)} text chunks")

            if not chunks:
                raise ValueError("No valid chunks created from PDF")

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.safe_embedding_generation(chunks)

            if embeddings.size == 0:
                raise ValueError("No embeddings generated")

            # Create RAPTOR-style structure
            paper_data = {
                "title": "PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models",
                "journal": "NeurIPS 2025",
                "arxiv_id": "2509.15607",
                "processed_date": "2025-12-07",
                "processing_method": "robust_fixed",
                "raptor_structure": {
                    "L0": {
                        "chunks": chunks,
                        "embeddings": embeddings.tolist(),
                        "chunk_count": len(chunks)
                    },
                    "L1": {
                        "summary": "This paper presents PRIMT, a framework for preference-based reinforcement learning that leverages foundation models for multimodal synthetic feedback and trajectory synthesis. The approach combines LLMs and VLMs through hierarchical neuro-symbolic fusion and includes bidirectional trajectory synthesis for improved query handling and credit assignment.",
                        "embedding": embeddings.mean(axis=0).tolist() if embeddings.size > 0 else []
                    },
                    "L2": {
                        "paper_summary": "PRIMT addresses key challenges in preference-based RL by using foundation models for both evaluation and trajectory generation. The hierarchical neuro-symbolic fusion strategy integrates VLM and LLM strengths using Probabilistic Soft Logic, while bidirectional trajectory synthesis provides foresight generation and hindsight counterfactual augmentation with causal auxiliary loss for improved credit assignment.",
                        "embedding": embeddings.mean(axis=0).tolist() if embeddings.size > 0 else []
                    }
                },
                "metadata": {
                    "total_pages": text.count("Page "),
                    "total_characters": len(text),
                    "embedding_dimension": embeddings.shape[1] if embeddings.size > 0 else 0,
                    "processing_status": "success"
                }
            }

            # Save to JSON
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully processed PRIMT paper and saved to {output_path}")
            return paper_data

        except Exception as e:
            logger.error(f"Error processing PRIMT paper: {e}")
            # Create minimal fallback structure
            fallback_data = {
                "title": "PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models",
                "journal": "NeurIPS 2025",
                "arxiv_id": "2509.15607",
                "processing_status": "partial_failure",
                "error": str(e),
                "raptor_structure": {
                    "L2": {
                        "paper_summary": "PRIMT is a preference-based reinforcement learning framework using foundation models for multimodal feedback and trajectory synthesis. Despite processing challenges, this represents a key NeurIPS 2025 contribution to multimodal RL.",
                        "embedding": []
                    }
                }
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_data, f, indent=2, ensure_ascii=False)

            return fallback_data

def main():
    processor = RobustPRIMTProcessor()

    primt_pdf_path = "/home/juke/git/AI-CoScientist/data/발달장애/neurips_2025_papers/priority2/2509.15607_PRIMT.pdf"
    output_path = "/home/juke/git/AI-CoScientist/data/reference_papers/neurips_2025_processed/2509.15607_PRIMT_fixed.json"

    if not os.path.exists(primt_pdf_path):
        logger.error(f"PRIMT PDF not found at {primt_pdf_path}")
        return

    result = processor.process_primt_paper(primt_pdf_path, output_path)

    if result["processing_status"] == "success":
        logger.info("✅ PRIMT paper processing completed successfully!")
        logger.info(f"Generated {result['raptor_structure']['L0']['chunk_count']} text chunks")
        logger.info(f"Embedding dimension: {result['metadata']['embedding_dimension']}")
    else:
        logger.warning("⚠️ PRIMT paper processing completed with issues")
        logger.info(f"Fallback structure created: {output_path}")

if __name__ == "__main__":
    main()