#!/usr/bin/env python3
"""Filter documents to keep only latest versions."""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime


class DocumentVersion:
    """Represents a document with version metadata."""

    def __init__(self, file_path: Path):
        self.path = file_path
        self.name = file_path.name
        self.stem = file_path.stem
        self.suffix = file_path.suffix

        # File metadata
        if file_path.exists():
            stat = file_path.stat()
            self.size = stat.st_size
            self.mtime = datetime.fromtimestamp(stat.st_mtime)
        else:
            self.size = 0
            self.mtime = datetime.min

        # Version analysis
        self.base_name = self._normalize_name()
        self.version_info = self._extract_version_info()
        self.score = self._calculate_score()

    def _normalize_name(self) -> str:
        """Normalize filename for grouping."""
        name = self.stem.lower()

        # Remove version patterns
        patterns = [
            r'[_\s\-]?v\.?\d+\.?\d*',  # v1, v1.0, v2.1
            r'[_\s\-]?ver\.?\d+',       # ver1, ver2
            r'\s*\(\d+\)',              # (1), (2)
            r'[_\s]\d+$',               # _1, _2 at end
            r'[_\s\-]?(초안|수정본?|최종본?|draft|revised?|final|updated?|new)',  # Keywords
            r'[_\s\-]?copy',            # copy
        ]

        for pattern in patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)

        # Remove special characters except Korean/English
        name = re.sub(r'[^\w가-힣]+', '', name)

        return name.strip()

    def _extract_version_info(self) -> Dict:
        """Extract version indicators from filename."""
        info = {
            'version_major': 0,
            'version_minor': 0,
            'has_final': False,
            'has_revised': False,
            'has_draft': False,
            'has_duplicate_suffix': False,
            'keywords': []
        }

        name_lower = self.stem.lower()

        # Extract version numbers
        version_match = re.search(r'v\.?(\d+)\.?(\d+)?', name_lower)
        if version_match:
            info['version_major'] = int(version_match.group(1))
            info['version_minor'] = int(version_match.group(2) or 0)

        # Check for keywords
        if re.search(r'(final|최종)', name_lower):
            info['has_final'] = True
            info['keywords'].append('final')

        if re.search(r'(revised?|수정)', name_lower):
            info['has_revised'] = True
            info['keywords'].append('revised')

        if re.search(r'(draft|초안)', name_lower):
            info['has_draft'] = True
            info['keywords'].append('draft')

        if re.search(r'(updated?|new)', name_lower):
            info['keywords'].append('updated')

        # Check for duplicate suffixes
        if re.search(r'\s*\(\d+\)|\s+\d+$|copy', name_lower):
            info['has_duplicate_suffix'] = True

        return info

    def _calculate_score(self) -> float:
        """Calculate quality score for this document."""
        score = 100.0  # Base score

        info = self.version_info

        # Version number bonus
        score += info['version_major'] * 50
        score += info['version_minor'] * 5

        # Keyword bonuses/penalties
        if info['has_final']:
            score += 40
        if info['has_revised']:
            score += 30
        if 'updated' in info['keywords']:
            score += 20
        if info['has_draft']:
            score -= 10

        # Duplicate suffix penalty
        if info['has_duplicate_suffix']:
            score -= 20

        # File size as tiebreaker (larger is usually better)
        score += (self.size / 1024) * 0.1  # +0.1 per KB

        # Modification time as tiebreaker (newer is better)
        days_since_epoch = (self.mtime - datetime(1970, 1, 1)).days
        score += days_since_epoch * 0.001

        # Prefer PDF over DOCX (more stable format)
        if self.suffix.lower() == '.pdf':
            score += 5

        return score

    def __repr__(self):
        return f"DocumentVersion({self.name}, score={self.score:.2f})"


def group_documents(doc_versions: List[DocumentVersion]) -> Dict[str, List[DocumentVersion]]:
    """Group documents by normalized base name."""
    groups = defaultdict(list)

    for doc in doc_versions:
        groups[doc.base_name].append(doc)

    return dict(groups)


def select_best_versions(groups: Dict[str, List[DocumentVersion]]) -> List[DocumentVersion]:
    """Select the best version from each group."""
    selected = []

    for base_name, versions in groups.items():
        if len(versions) == 1:
            # Only one version, automatically include
            selected.append(versions[0])
        else:
            # Multiple versions, select best by score
            best = max(versions, key=lambda v: v.score)
            selected.append(best)

    return selected


def main():
    """Main entry point."""
    # Load unique documents
    unique_docs_file = Path("/Users/jiookcha/Documents/git/AI-CoScientist/unique_documents.txt")

    if not unique_docs_file.exists():
        print("❌ unique_documents.txt not found")
        return

    print("=" * 80)
    print("📋 Document Version Filtering")
    print("=" * 80)

    # Read all files
    with open(unique_docs_file, 'r', encoding='utf-8') as f:
        file_paths = [Path(line.strip()) for line in f if line.strip()]

    print(f"\n📂 Total unique files: {len(file_paths)}")

    # Filter only supported formats
    supported_files = [
        f for f in file_paths
        if f.suffix.lower() in ['.pdf', '.docx']
    ]

    print(f"📄 Supported formats (PDF, DOCX): {len(supported_files)}")

    # Analyze versions
    print(f"\n🔍 Analyzing versions...")
    doc_versions = [DocumentVersion(f) for f in supported_files]

    # Group by base name
    groups = group_documents(doc_versions)

    print(f"\n📊 Grouping Analysis:")
    print(f"   Total document groups: {len(groups)}")
    print(f"   Single-version groups: {sum(1 for g in groups.values() if len(g) == 1)}")
    print(f"   Multi-version groups: {sum(1 for g in groups.values() if len(g) > 1)}")

    # Show multi-version groups
    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}

    if multi_groups:
        print(f"\n📝 Multi-Version Groups (showing top 20):")
        for i, (base_name, versions) in enumerate(sorted(
            multi_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20], 1):
            print(f"\n{i}. Base: '{base_name}' ({len(versions)} versions)")

            # Sort by score
            sorted_versions = sorted(versions, key=lambda v: v.score, reverse=True)

            for j, ver in enumerate(sorted_versions, 1):
                marker = "✅ SELECTED" if j == 1 else "❌ Filtered"
                print(f"   {marker} [{j}] {ver.name}")
                print(f"      Score: {ver.score:.2f} | Size: {ver.size/1024:.1f}KB | "
                      f"Version: v{ver.version_info['version_major']}.{ver.version_info['version_minor']} | "
                      f"Keywords: {', '.join(ver.version_info['keywords']) or 'none'}")

    # Select best versions
    print(f"\n🎯 Selecting best versions...")
    selected = select_best_versions(groups)

    # Save filtered list
    output_file = Path("/Users/jiookcha/Documents/git/AI-CoScientist/filtered_documents.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in sorted(selected, key=lambda d: d.path):
            f.write(f"{doc.path}\n")

    # Statistics
    total_filtered = len(supported_files) - len(selected)

    print(f"\n{'='*80}")
    print(f"✅ FILTERING COMPLETE")
    print(f"{'='*80}")
    print(f"Original files: {len(supported_files)}")
    print(f"Selected files: {len(selected)}")
    print(f"Filtered out: {total_filtered}")
    print(f"Reduction: {total_filtered/len(supported_files)*100:.1f}%")
    print(f"\n💾 Filtered list saved to: {output_file}")
    print(f"{'='*80}\n")

    # Show file type distribution
    pdf_count = sum(1 for d in selected if d.suffix.lower() == '.pdf')
    docx_count = sum(1 for d in selected if d.suffix.lower() == '.docx')

    print(f"📊 Selected Files by Type:")
    print(f"   PDF: {pdf_count}")
    print(f"   DOCX: {docx_count}")


if __name__ == "__main__":
    main()
