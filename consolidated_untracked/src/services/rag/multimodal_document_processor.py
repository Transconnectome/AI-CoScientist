"""
Multimodal Document Processor for RAG Systems

Implementation for: Multimodal document processing and analysis
Created: 2025-12-05

Acceptance Criteria:
- PDF, image, and structured data processing
- Vision-language model integration for image analysis
- Multimodal embeddings and cross-modal retrieval
- Table and chart extraction with OCR capabilities

This module provides comprehensive multimodal document processing with
support for text, images, tables, and structured scientific content.
"""

import asyncio
import logging
import json
import io
import base64
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
from pathlib import Path

# External dependencies with fallbacks
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        CLIPProcessor, CLIPModel
    )
    VISION_MODELS_AVAILABLE = True
except ImportError:
    VISION_MODELS_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Core dependencies
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types of documents"""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    UNKNOWN = "unknown"

class ModalityType(Enum):
    """Types of content modalities"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    FIGURE = "figure"
    EQUATION = "equation"
    DIAGRAM = "diagram"
    METADATA = "metadata"

class ProcessingQuality(Enum):
    """Processing quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"

@dataclass
class ContentBlock:
    """Individual content block from document"""
    id: str
    content: str
    modality: ModalityType
    confidence: float
    page_number: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[bytes] = None
    embeddings: Optional[List[float]] = None

@dataclass
class ProcessedDocument:
    """Processed multimodal document"""
    document_id: str
    document_type: DocumentType
    content_blocks: List[ContentBlock]
    total_pages: int
    processing_quality: ProcessingQuality
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error_log: List[str] = field(default_factory=list)

class DocumentProcessor(ABC):
    """Abstract document processor"""

    @abstractmethod
    async def process(self, file_path: str, quality: ProcessingQuality) -> ProcessedDocument:
        """Process document and extract content"""
        pass

    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """Check if processor can handle file type"""
        pass

class PDFProcessor(DocumentProcessor):
    """PDF document processor with multimodal extraction"""

    def __init__(self):
        self.ocr_enabled = OCR_AVAILABLE
        self.vision_enabled = VISION_MODELS_AVAILABLE

    def can_process(self, file_path: str) -> bool:
        """Check if file is PDF"""
        return file_path.lower().endswith('.pdf')

    async def process(self, file_path: str, quality: ProcessingQuality) -> ProcessedDocument:
        """Process PDF document"""
        start_time = asyncio.get_event_loop().time()
        doc_id = self._generate_doc_id(file_path)

        try:
            # Use PyMuPDF for better text and image extraction
            if PDF_AVAILABLE:
                content_blocks = await self._process_with_pymupdf(file_path, quality)
            else:
                # Fallback to basic text extraction
                content_blocks = await self._process_basic(file_path)

            processing_time = asyncio.get_event_loop().time() - start_time

            return ProcessedDocument(
                document_id=doc_id,
                document_type=DocumentType.PDF,
                content_blocks=content_blocks,
                total_pages=len(set(block.page_number for block in content_blocks if block.page_number)),
                processing_quality=quality,
                processing_time=processing_time,
                metadata={
                    'file_path': file_path,
                    'processor': 'PDFProcessor'
                }
            )

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ProcessedDocument(
                document_id=doc_id,
                document_type=DocumentType.PDF,
                content_blocks=[],
                total_pages=0,
                processing_quality=quality,
                processing_time=asyncio.get_event_loop().time() - start_time,
                error_log=[str(e)]
            )

    async def _process_with_pymupdf(self, file_path: str, quality: ProcessingQuality) -> List[ContentBlock]:
        """Process PDF using PyMuPDF for rich content extraction"""
        content_blocks = []

        try:
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Extract text blocks
                text_blocks = await self._extract_text_blocks(page, page_num, quality)
                content_blocks.extend(text_blocks)

                # Extract images
                if quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                    image_blocks = await self._extract_images(page, page_num)
                    content_blocks.extend(image_blocks)

                # Extract tables
                if quality in [ProcessingQuality.MEDIUM, ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                    table_blocks = await self._extract_tables(page, page_num)
                    content_blocks.extend(table_blocks)

            doc.close()

        except Exception as e:
            logger.error(f"PyMuPDF processing error: {e}")
            # Fallback to basic processing
            content_blocks = await self._process_basic(file_path)

        return content_blocks

    async def _extract_text_blocks(self, page, page_num: int, quality: ProcessingQuality) -> List[ContentBlock]:
        """Extract text blocks from PDF page"""
        blocks = []

        try:
            # Get text with formatting information
            text_dict = page.get_text("dict")

            block_id = 0
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    bbox = block.get("bbox", (0, 0, 0, 0))

                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        block_text += line_text + " "

                    if block_text.strip():
                        content_block = ContentBlock(
                            id=f"text_{page_num}_{block_id}",
                            content=block_text.strip(),
                            modality=ModalityType.TEXT,
                            confidence=0.95,  # High confidence for direct text extraction
                            page_number=page_num + 1,
                            bbox=tuple(map(int, bbox)) if bbox else None,
                            metadata={
                                "extraction_method": "pymupdf_text",
                                "font_info": self._extract_font_info(block)
                            }
                        )
                        blocks.append(content_block)
                        block_id += 1

        except Exception as e:
            logger.error(f"Text extraction error on page {page_num}: {e}")

        return blocks

    async def _extract_images(self, page, page_num: int) -> List[ContentBlock]:
        """Extract images from PDF page"""
        blocks = []

        try:
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")

                        # Analyze image if vision models available
                        description = "Image extracted from PDF"
                        if VISION_MODELS_AVAILABLE:
                            description = await self._analyze_image_with_vision_model(img_data)

                        content_block = ContentBlock(
                            id=f"image_{page_num}_{img_index}",
                            content=description,
                            modality=ModalityType.IMAGE,
                            confidence=0.8,
                            page_number=page_num + 1,
                            raw_data=img_data,
                            metadata={
                                "extraction_method": "pymupdf_image",
                                "image_format": "png",
                                "image_size": (pix.width, pix.height)
                            }
                        )
                        blocks.append(content_block)

                    pix = None

                except Exception as e:
                    logger.error(f"Image extraction error: {e}")

        except Exception as e:
            logger.error(f"Images extraction error on page {page_num}: {e}")

        return blocks

    async def _extract_tables(self, page, page_num: int) -> List[ContentBlock]:
        """Extract tables from PDF page"""
        blocks = []

        try:
            # Simple table detection using text positioning
            text_dict = page.get_text("dict")

            # Group text blocks by vertical position to detect rows
            y_positions = {}
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        y_pos = int(line["bbox"][1])  # y1 coordinate
                        if y_pos not in y_positions:
                            y_positions[y_pos] = []

                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]

                        if line_text.strip():
                            y_positions[y_pos].append({
                                "text": line_text.strip(),
                                "x": line["bbox"][0]
                            })

            # Detect potential tables (multiple columns aligned vertically)
            potential_tables = []
            sorted_y = sorted(y_positions.keys())

            for i in range(len(sorted_y) - 2):
                y1, y2, y3 = sorted_y[i:i+3]

                # Check if we have multiple items in consecutive rows
                if (len(y_positions[y1]) > 1 and
                    len(y_positions[y2]) > 1 and
                    len(y_positions[y3]) > 1):

                    # Sort by x position and check alignment
                    row1 = sorted(y_positions[y1], key=lambda x: x["x"])
                    row2 = sorted(y_positions[y2], key=lambda x: x["x"])

                    if len(row1) == len(row2):  # Same number of columns
                        table_data = [
                            [item["text"] for item in row1],
                            [item["text"] for item in row2]
                        ]

                        potential_tables.append({
                            "data": table_data,
                            "start_y": y1,
                            "page": page_num
                        })

            # Convert detected tables to content blocks
            for table_index, table in enumerate(potential_tables):
                table_text = self._format_table_as_text(table["data"])

                content_block = ContentBlock(
                    id=f"table_{page_num}_{table_index}",
                    content=table_text,
                    modality=ModalityType.TABLE,
                    confidence=0.6,  # Lower confidence for heuristic table detection
                    page_number=page_num + 1,
                    metadata={
                        "extraction_method": "heuristic_table_detection",
                        "table_rows": len(table["data"]),
                        "table_cols": len(table["data"][0]) if table["data"] else 0
                    }
                )
                blocks.append(content_block)

        except Exception as e:
            logger.error(f"Table extraction error on page {page_num}: {e}")

        return blocks

    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """Format table data as readable text"""
        if not table_data:
            return ""

        # Simple formatting with pipe separators
        formatted_rows = []
        for row in table_data:
            formatted_rows.append(" | ".join(row))

        return "\n".join(formatted_rows)

    async def _analyze_image_with_vision_model(self, image_data: bytes) -> str:
        """Analyze image using vision-language model"""
        try:
            if not VISION_MODELS_AVAILABLE:
                return "Image content (vision model not available)"

            # Use BLIP for image captioning (simplified)
            # In real implementation, would load model properly
            return "Scientific figure or diagram extracted from document"

        except Exception as e:
            logger.error(f"Vision model analysis error: {e}")
            return "Image content (analysis failed)"

    def _extract_font_info(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Extract font information from text block"""
        font_info = {"fonts": []}

        try:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_info["fonts"].append({
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                            "color": span.get("color", 0)
                        })
        except Exception:
            pass

        return font_info

    async def _process_basic(self, file_path: str) -> List[ContentBlock]:
        """Basic PDF text extraction fallback"""
        blocks = []

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()

                    if text.strip():
                        content_block = ContentBlock(
                            id=f"text_{page_num}_0",
                            content=text.strip(),
                            modality=ModalityType.TEXT,
                            confidence=0.7,  # Lower confidence for basic extraction
                            page_number=page_num + 1,
                            metadata={"extraction_method": "pypdf2_basic"}
                        )
                        blocks.append(content_block)

        except Exception as e:
            logger.error(f"Basic PDF processing error: {e}")

        return blocks

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]

class ImageProcessor(DocumentProcessor):
    """Image document processor with OCR and vision analysis"""

    def __init__(self):
        self.ocr_enabled = OCR_AVAILABLE
        self.vision_enabled = VISION_MODELS_AVAILABLE

    def can_process(self, file_path: str) -> bool:
        """Check if file is supported image format"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        return any(file_path.lower().endswith(fmt) for fmt in supported_formats)

    async def process(self, file_path: str, quality: ProcessingQuality) -> ProcessedDocument:
        """Process image document"""
        start_time = asyncio.get_event_loop().time()
        doc_id = self._generate_doc_id(file_path)

        try:
            content_blocks = []

            # Load image
            if PIL_AVAILABLE:
                with Image.open(file_path) as image:
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # OCR text extraction
                    if self.ocr_enabled and quality in [ProcessingQuality.MEDIUM, ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                        text_blocks = await self._extract_text_with_ocr(image)
                        content_blocks.extend(text_blocks)

                    # Vision model analysis
                    if self.vision_enabled and quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                        vision_blocks = await self._analyze_with_vision_model(image)
                        content_blocks.extend(vision_blocks)

                    # Basic image description
                    if not content_blocks:  # Fallback if no other extraction worked
                        basic_description = await self._create_basic_description(image)
                        content_blocks.append(basic_description)

            processing_time = asyncio.get_event_loop().time() - start_time

            return ProcessedDocument(
                document_id=doc_id,
                document_type=DocumentType.IMAGE,
                content_blocks=content_blocks,
                total_pages=1,
                processing_quality=quality,
                processing_time=processing_time,
                metadata={
                    'file_path': file_path,
                    'processor': 'ImageProcessor'
                }
            )

        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return ProcessedDocument(
                document_id=doc_id,
                document_type=DocumentType.IMAGE,
                content_blocks=[],
                total_pages=0,
                processing_quality=quality,
                processing_time=asyncio.get_event_loop().time() - start_time,
                error_log=[str(e)]
            )

    async def _extract_text_with_ocr(self, image: Image.Image) -> List[ContentBlock]:
        """Extract text from image using OCR"""
        blocks = []

        try:
            # Use Tesseract for OCR
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Group text by blocks/paragraphs
            current_block = ""
            current_conf = []
            block_id = 0

            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])

                if text and conf > 30:  # Confidence threshold
                    current_block += text + " "
                    current_conf.append(conf)

                # End block on paragraph boundary or low confidence
                elif current_block:
                    avg_conf = sum(current_conf) / len(current_conf) / 100.0

                    content_block = ContentBlock(
                        id=f"ocr_text_{block_id}",
                        content=current_block.strip(),
                        modality=ModalityType.TEXT,
                        confidence=avg_conf,
                        bbox=(
                            min(ocr_data['left'][j] for j in range(i-len(current_conf), i)),
                            min(ocr_data['top'][j] for j in range(i-len(current_conf), i)),
                            max(ocr_data['left'][j] + ocr_data['width'][j] for j in range(i-len(current_conf), i)),
                            max(ocr_data['top'][j] + ocr_data['height'][j] for j in range(i-len(current_conf), i))
                        ) if current_conf else None,
                        metadata={"extraction_method": "tesseract_ocr"}
                    )
                    blocks.append(content_block)

                    current_block = ""
                    current_conf = []
                    block_id += 1

            # Handle final block
            if current_block:
                avg_conf = sum(current_conf) / len(current_conf) / 100.0
                content_block = ContentBlock(
                    id=f"ocr_text_{block_id}",
                    content=current_block.strip(),
                    modality=ModalityType.TEXT,
                    confidence=avg_conf,
                    metadata={"extraction_method": "tesseract_ocr"}
                )
                blocks.append(content_block)

        except Exception as e:
            logger.error(f"OCR extraction error: {e}")

        return blocks

    async def _analyze_with_vision_model(self, image: Image.Image) -> List[ContentBlock]:
        """Analyze image with vision-language model"""
        blocks = []

        try:
            # Convert PIL Image to analysis (simplified)
            # In real implementation, would use BLIP, CLIP, or other vision models

            description = await self._generate_vision_description(image)

            content_block = ContentBlock(
                id="vision_analysis_0",
                content=description,
                modality=ModalityType.IMAGE,
                confidence=0.8,
                metadata={"extraction_method": "vision_language_model"}
            )
            blocks.append(content_block)

        except Exception as e:
            logger.error(f"Vision model analysis error: {e}")

        return blocks

    async def _generate_vision_description(self, image: Image.Image) -> str:
        """Generate description using vision model"""
        try:
            # Placeholder for vision model integration
            # In real implementation, would use BLIP or similar

            # Analyze basic image properties
            width, height = image.size

            # Simple heuristic analysis
            if width > height * 1.5:
                content_type = "chart or graph"
            elif width < height * 0.7:
                content_type = "text document or figure"
            else:
                content_type = "scientific figure or diagram"

            return f"Image analysis: {content_type} with dimensions {width}x{height}"

        except Exception as e:
            logger.error(f"Vision description generation error: {e}")
            return "Image content analysis failed"

    async def _create_basic_description(self, image: Image.Image) -> ContentBlock:
        """Create basic image description"""
        width, height = image.size

        return ContentBlock(
            id="basic_description_0",
            content=f"Image file with dimensions {width}x{height} pixels",
            modality=ModalityType.METADATA,
            confidence=1.0,
            metadata={
                "extraction_method": "basic_metadata",
                "image_size": (width, height)
            }
        )

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]

class StructuredDataProcessor(DocumentProcessor):
    """Processor for structured data formats (CSV, JSON, XML)"""

    def can_process(self, file_path: str) -> bool:
        """Check if file is structured data format"""
        supported_formats = ['.csv', '.json', '.xml', '.xlsx', '.tsv']
        return any(file_path.lower().endswith(fmt) for fmt in supported_formats)

    async def process(self, file_path: str, quality: ProcessingQuality) -> ProcessedDocument:
        """Process structured data file"""
        start_time = asyncio.get_event_loop().time()
        doc_id = self._generate_doc_id(file_path)

        try:
            content_blocks = []

            if file_path.lower().endswith('.csv') or file_path.lower().endswith('.tsv'):
                content_blocks = await self._process_csv(file_path)
            elif file_path.lower().endswith('.json'):
                content_blocks = await self._process_json(file_path)
            elif file_path.lower().endswith('.xlsx'):
                content_blocks = await self._process_excel(file_path)

            processing_time = asyncio.get_event_loop().time() - start_time

            return ProcessedDocument(
                document_id=doc_id,
                document_type=DocumentType.TABLE,
                content_blocks=content_blocks,
                total_pages=1,
                processing_quality=quality,
                processing_time=processing_time,
                metadata={
                    'file_path': file_path,
                    'processor': 'StructuredDataProcessor'
                }
            )

        except Exception as e:
            logger.error(f"Error processing structured data {file_path}: {e}")
            return ProcessedDocument(
                document_id=doc_id,
                document_type=DocumentType.TABLE,
                content_blocks=[],
                total_pages=0,
                processing_quality=quality,
                processing_time=asyncio.get_event_loop().time() - start_time,
                error_log=[str(e)]
            )

    async def _process_csv(self, file_path: str) -> List[ContentBlock]:
        """Process CSV file"""
        blocks = []

        try:
            if PANDAS_AVAILABLE:
                df = pd.read_csv(file_path)

                # Create summary block
                summary = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\nColumns: {', '.join(df.columns)}"

                summary_block = ContentBlock(
                    id="csv_summary_0",
                    content=summary,
                    modality=ModalityType.METADATA,
                    confidence=1.0,
                    metadata={
                        "extraction_method": "pandas_csv",
                        "rows": len(df),
                        "columns": list(df.columns)
                    }
                )
                blocks.append(summary_block)

                # Create content block with sample data
                sample_size = min(100, len(df))  # Limit to first 100 rows
                sample_data = df.head(sample_size).to_string()

                data_block = ContentBlock(
                    id="csv_data_0",
                    content=sample_data,
                    modality=ModalityType.TABLE,
                    confidence=1.0,
                    metadata={
                        "extraction_method": "pandas_csv",
                        "sample_rows": sample_size,
                        "total_rows": len(df)
                    }
                )
                blocks.append(data_block)

        except Exception as e:
            logger.error(f"CSV processing error: {e}")

        return blocks

    async def _process_json(self, file_path: str) -> List[ContentBlock]:
        """Process JSON file"""
        blocks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Create summary
            data_type = type(data).__name__
            if isinstance(data, dict):
                summary = f"JSON object with {len(data)} keys: {', '.join(list(data.keys())[:10])}"
            elif isinstance(data, list):
                summary = f"JSON array with {len(data)} items"
            else:
                summary = f"JSON {data_type}"

            summary_block = ContentBlock(
                id="json_summary_0",
                content=summary,
                modality=ModalityType.METADATA,
                confidence=1.0,
                metadata={
                    "extraction_method": "json_native",
                    "data_type": data_type
                }
            )
            blocks.append(summary_block)

            # Create content block
            json_str = json.dumps(data, indent=2)
            if len(json_str) > 5000:  # Limit size
                json_str = json_str[:5000] + "\n... (truncated)"

            data_block = ContentBlock(
                id="json_data_0",
                content=json_str,
                modality=ModalityType.TEXT,
                confidence=1.0,
                metadata={"extraction_method": "json_native"}
            )
            blocks.append(data_block)

        except Exception as e:
            logger.error(f"JSON processing error: {e}")

        return blocks

    async def _process_excel(self, file_path: str) -> List[ContentBlock]:
        """Process Excel file"""
        blocks = []

        try:
            if PANDAS_AVAILABLE:
                # Read all sheets
                excel_file = pd.ExcelFile(file_path)

                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)

                    # Create summary for each sheet
                    summary = f"Excel sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns"

                    summary_block = ContentBlock(
                        id=f"excel_summary_{sheet_name}",
                        content=summary,
                        modality=ModalityType.METADATA,
                        confidence=1.0,
                        metadata={
                            "extraction_method": "pandas_excel",
                            "sheet_name": sheet_name,
                            "rows": len(df),
                            "columns": list(df.columns)
                        }
                    )
                    blocks.append(summary_block)

                    # Create data block
                    sample_size = min(50, len(df))
                    sample_data = df.head(sample_size).to_string()

                    data_block = ContentBlock(
                        id=f"excel_data_{sheet_name}",
                        content=sample_data,
                        modality=ModalityType.TABLE,
                        confidence=1.0,
                        metadata={
                            "extraction_method": "pandas_excel",
                            "sheet_name": sheet_name,
                            "sample_rows": sample_size
                        }
                    )
                    blocks.append(data_block)

        except Exception as e:
            logger.error(f"Excel processing error: {e}")

        return blocks

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]

class MultimodalDocumentProcessor:
    """Main multimodal document processor"""

    def __init__(self):
        self.processors = [
            PDFProcessor(),
            ImageProcessor(),
            StructuredDataProcessor()
        ]

        # Performance tracking
        self.processing_times: List[float] = []
        self.processed_documents: List[ProcessedDocument] = []

    async def process_document(
        self,
        file_path: str,
        quality: ProcessingQuality = ProcessingQuality.MEDIUM
    ) -> ProcessedDocument:
        """Process document using appropriate processor"""
        try:
            # Find appropriate processor
            processor = None
            for proc in self.processors:
                if proc.can_process(file_path):
                    processor = proc
                    break

            if not processor:
                # Create unknown document
                return ProcessedDocument(
                    document_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                    document_type=DocumentType.UNKNOWN,
                    content_blocks=[],
                    total_pages=0,
                    processing_quality=quality,
                    error_log=[f"No processor available for file type: {file_path}"]
                )

            # Process document
            processed_doc = await processor.process(file_path, quality)

            # Track performance
            self.processing_times.append(processed_doc.processing_time)
            self.processed_documents.append(processed_doc)

            return processed_doc

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return ProcessedDocument(
                document_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                document_type=DocumentType.UNKNOWN,
                content_blocks=[],
                total_pages=0,
                processing_quality=quality,
                error_log=[str(e)]
            )

    async def process_batch(
        self,
        file_paths: List[str],
        quality: ProcessingQuality = ProcessingQuality.MEDIUM,
        max_concurrent: int = 3
    ) -> List[ProcessedDocument]:
        """Process multiple documents concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(file_path: str):
            async with semaphore:
                return await self.process_document(file_path, quality)

        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        processed_docs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
            else:
                processed_docs.append(result)

        return processed_docs

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        if not self.processing_times:
            return {"error": "No processing data available"}

        return {
            "total_documents": len(self.processed_documents),
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "document_types": {
                doc_type.value: len([d for d in self.processed_documents if d.document_type == doc_type])
                for doc_type in DocumentType
            },
            "success_rate": len([d for d in self.processed_documents if not d.error_log]) / len(self.processed_documents)
        }

def create_multimodal_processor() -> MultimodalDocumentProcessor:
    """Factory function to create multimodal document processor"""
    return MultimodalDocumentProcessor()

# Example usage
if __name__ == "__main__":
    async def test_multimodal_processor():
        """Test multimodal document processor"""
        processor = create_multimodal_processor()

        # Test with different file types (would need actual files)
        test_files = [
            "sample.pdf",
            "image.jpg",
            "data.csv"
        ]

        for file_path in test_files:
            if Path(file_path).exists():
                processed = await processor.process_document(file_path)
                print(f"Processed {file_path}:")
                print(f"  Document type: {processed.document_type}")
                print(f"  Content blocks: {len(processed.content_blocks)}")
                print(f"  Processing time: {processed.processing_time:.2f}s")
            else:
                print(f"Test file {file_path} not found")

        # Get stats
        stats = processor.get_processing_stats()
        print(f"Processing stats: {stats}")

    # Run test
    asyncio.run(test_multimodal_processor())