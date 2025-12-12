#!/usr/bin/env python3
"""Analyze and identify duplicate documents in papers_collection."""

import hashlib
from pathlib import Path
from collections import defaultdict
import re

def calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()

def normalize_filename(filename: str) -> str:
    """Normalize filename for comparison."""
    # Remove extensions
    name = Path(filename).stem
    # Convert to lowercase
    name = name.lower()
    # Remove common suffixes
    name = re.sub(r'[_\s]+(ver|v|final|수정|초안|최종)[\d\._]*$', '', name, flags=re.IGNORECASE)
    # Remove special characters
    name = re.sub(r'[^\w가-힣]+', '', name)
    return name

def main():
    papers_dir = Path("/Users/jiookcha/Documents/git/AI-CoScientist/papers_collection")

    # Find all documents
    pdf_files = list(papers_dir.glob("*.pdf"))
    docx_files = list(papers_dir.glob("*.docx"))
    doc_files = list(papers_dir.glob("*.doc"))

    all_files = pdf_files + docx_files + doc_files

    print(f"📊 Document Analysis")
    print("=" * 70)
    print(f"Total files found: {len(all_files)}")
    print(f"  PDF: {len(pdf_files)}")
    print(f"  DOCX: {len(docx_files)}")
    print(f"  DOC: {len(doc_files)}\n")

    # Group by hash (exact duplicates)
    hash_groups = defaultdict(list)
    print("🔍 Calculating file hashes...")

    for i, file_path in enumerate(all_files):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(all_files)}")
        try:
            file_hash = calculate_file_hash(file_path)
            hash_groups[file_hash].append(file_path)
        except Exception as e:
            print(f"   ⚠️  Error reading {file_path.name}: {e}")

    # Find exact duplicates
    exact_duplicates = {h: files for h, files in hash_groups.items() if len(files) > 1}

    print(f"\n📋 Exact Duplicates (by content hash):")
    print(f"Found {len(exact_duplicates)} groups of exact duplicates\n")

    for i, (hash_val, files) in enumerate(exact_duplicates.items(), 1):
        print(f"Group {i}: {len(files)} files")
        for f in files:
            print(f"  - {f.name}")
        print()

    # Group by normalized filename (similar names)
    name_groups = defaultdict(list)
    for file_path in all_files:
        normalized = normalize_filename(file_path.name)
        if normalized:  # Skip empty names
            name_groups[normalized].append(file_path)

    similar_names = {n: files for n, files in name_groups.items() if len(files) > 1}

    print(f"\n📝 Similar Names (likely versions):")
    print(f"Found {len(similar_names)} groups with similar names\n")

    for i, (norm_name, files) in enumerate(list(similar_names.items())[:10], 1):
        print(f"Group {i}: '{norm_name}'")
        for f in files:
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        print()

    # Recommendations
    unique_by_hash = len(hash_groups)
    total_duplicates = len(all_files) - unique_by_hash

    print("=" * 70)
    print(f"📊 Summary:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Unique by hash: {unique_by_hash}")
    print(f"  Exact duplicates to remove: {total_duplicates}")
    print(f"  Similar name groups: {len(similar_names)}")
    print("=" * 70)

    # Save unique file list
    unique_files = [files[0] for files in hash_groups.values()]
    output_file = papers_dir.parent / "unique_documents.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in sorted(unique_files):
            f.write(f"{file_path}\n")

    print(f"\n✅ Unique file list saved to: {output_file}")
    print(f"   Total unique files: {len(unique_files)}")

if __name__ == "__main__":
    main()
