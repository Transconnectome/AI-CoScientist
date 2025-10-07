#!/usr/bin/env python3
"""Extract text from PDF file."""

import sys
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    text = ""

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        print(f"📄 PDF Info: {num_pages} pages", file=sys.stderr)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
            text += "\n\n---PAGE BREAK---\n\n"

    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <pdf_file>", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    extracted_text = extract_text_from_pdf(pdf_path)

    # Print to stdout
    print(extracted_text)
