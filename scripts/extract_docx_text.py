#!/usr/bin/env python3
"""Extract full text from .docx file."""

import sys
from docx import Document

def extract_text(docx_path: str) -> str:
    """Extract all text from .docx file."""
    doc = Document(docx_path)

    # Extract all paragraphs
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

    return "\n\n".join(paragraphs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_text.py <path_to_docx>")
        sys.exit(1)

    text = extract_text(sys.argv[1])
    print(text)
