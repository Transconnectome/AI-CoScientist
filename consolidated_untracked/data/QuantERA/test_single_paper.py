#!/usr/bin/env python3
"""Test ingestion on single paper and save output"""
import json
import logging
from pathlib import Path
from src.ingest import QuantERAIngestor

logging.basicConfig(level=logging.INFO)

# Initialize ingestor
ingestor = QuantERAIngestor()

# Process Cerezo 2021 paper
paper_path = "Papers/Cerezo-2021-Variational quantum algorithms.pdf"
print(f"Processing: {paper_path}")

processed_doc = ingestor.process_paper(paper_path)

# Convert to serializable format
doc_dict = {
    'title': processed_doc.title,
    'authors': processed_doc.authors,
    'abstract': processed_doc.abstract,
    'chunks': processed_doc.chunks,
    'metadata': processed_doc.metadata,
    'mathematical_elements': processed_doc.mathematical_elements,
    'circuit_descriptions': processed_doc.circuit_descriptions,
    'total_pages': processed_doc.total_pages,
    'processing_timestamp': processed_doc.processing_timestamp
}

# Save to JSON
output_file = "test_cerezo_output.json"
with open(output_file, 'w') as f:
    json.dump([doc_dict], f, indent=2)

print(f"\n✓ Successfully processed:")
print(f"  Title: {processed_doc.title}")
print(f"  Authors: {len(processed_doc.authors)} author(s)")
print(f"  Total pages: {processed_doc.total_pages}")
print(f"  Chunks: {len(processed_doc.chunks)}")
print(f"  Math elements: {len(processed_doc.mathematical_elements)}")
print(f"  Circuit descriptions: {len(processed_doc.circuit_descriptions)}")
print(f"  Output saved to: {output_file}")
