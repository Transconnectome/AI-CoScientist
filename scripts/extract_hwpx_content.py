#!/usr/bin/env python3
"""Extract readable content from HWPX section0.xml file."""

import xml.etree.ElementTree as ET
import sys
from pathlib import Path

def extract_text_from_element(elem, namespace_map=None):
    """Recursively extract text from XML element."""
    texts = []

    # Get text from current element
    if elem.text:
        texts.append(elem.text.strip())

    # Get text from child elements
    for child in elem:
        child_texts = extract_text_from_element(child, namespace_map)
        texts.extend(child_texts)

        # Get tail text
        if child.tail:
            texts.append(child.tail.strip())

    return [t for t in texts if t]  # Filter empty strings

def extract_hwpx_content(xml_file):
    """Parse HWPX section0.xml and extract readable content."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract all text content
        all_text = extract_text_from_element(root)

        # Join with newlines and clean up
        content = '\n'.join(all_text)

        # Remove excessive whitespace
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        return '\n'.join(lines)

    except Exception as e:
        print(f"Error parsing XML: {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    xml_file = Path('/Users/jiookcha/Documents/git/AI-CoScientist/temp_grant_extract/Contents/section0.xml')

    if not xml_file.exists():
        print(f"File not found: {xml_file}", file=sys.stderr)
        sys.exit(1)

    content = extract_hwpx_content(xml_file)

    if content:
        # Save to output file
        output_file = Path('/Users/jiookcha/Documents/git/AI-CoScientist/temp_grant_extract/extracted_content.txt')
        output_file.write_text(content, encoding='utf-8')
        print(f"Content extracted to: {output_file}")
        print(f"\nFirst 2000 characters:")
        print(content[:2000])
    else:
        print("Failed to extract content", file=sys.stderr)
        sys.exit(1)
