#!/usr/bin/env python3
"""Convert text file to .docx with proper formatting."""

import sys
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def txt_to_docx(txt_path, docx_path):
    """Convert txt to docx with formatting."""
    # Read text
    with open(txt_path) as f:
        text = f.read()

    # Create document
    doc = Document()

    # Split into paragraphs
    paragraphs = text.split('\n\n')

    for para_text in paragraphs:
        if not para_text.strip():
            continue

        # Add paragraph
        p = doc.add_paragraph(para_text.strip())

        # Format based on content
        if para_text.strip().isupper() and len(para_text.strip()) < 100:
            # Section header
            p.style = 'Heading 1'
        elif para_text.startswith('Abstract') or para_text.startswith('Introduction:') or para_text.startswith('Discussion:'):
            # Major section
            p.style = 'Heading 1'
        elif len(para_text.strip()) < 80 and '\n' not in para_text and para_text.strip().endswith(':'):
            # Subsection
            p.style = 'Heading 2'

    # Save
    doc.save(docx_path)
    print(f"✅ Converted {txt_path} → {docx_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python txt_to_docx.py <input.txt> <output.docx>")
        sys.exit(1)

    txt_to_docx(sys.argv[1], sys.argv[2])
