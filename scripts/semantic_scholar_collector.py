#!/usr/bin/env python3
"""
Collect papers from Semantic Scholar API.
Searches for papers from specific journals with required keywords.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict
import aiohttp

# Semantic Scholar API endpoint
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Journal names for Semantic Scholar
JOURNALS = [
    "Nature",
    "Nature Medicine",
    "Nature Biomedical Engineering",
    "Nature Human Behaviour",
    "Science"
]

# Search keywords
KEYWORDS = [
    "foundation model",
    "large language model",
    "transformer model"
]


async def search_semantic_scholar(
    query: str,
    venue: str,
    limit: int = 100,
    year_from: int = 2020,
    year_to: int = 2025
) -> List[Dict]:
    """Search Semantic Scholar for papers matching criteria."""

    params = {
        'query': query,
        'venue': venue,
        'year': f'{year_from}-{year_to}',
        'limit': limit,
        'fields': 'paperId,externalIds,title,abstract,venue,year,authors,openAccessPdf'
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(S2_API, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('data', [])
            else:
                print(f"Error {response.status} for venue '{venue}': {await response.text()}")
                return []


async def collect_papers_for_journal(journal: str) -> List[Dict]:
    """Collect papers for a specific journal."""
    print(f"\nSearching {journal}...")

    all_papers = []
    seen_ids = set()

    # Try each keyword
    for keyword in KEYWORDS:
        print(f"  Keyword: '{keyword}'")
        papers = await search_semantic_scholar(
            query=keyword,
            venue=journal,
            limit=100
        )

        # Deduplicate and collect
        for paper in papers:
            paper_id = paper.get('paperId')
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                all_papers.append(paper)

        print(f"    Found {len(papers)} papers (total unique: {len(all_papers)})")

        # Rate limiting (S2 allows 100 req/5min for public API)
        await asyncio.sleep(1)

    return all_papers


async def main():
    """Main collection workflow."""

    print("="*70)
    print("SEMANTIC SCHOLAR PAPER COLLECTION")
    print("="*70)
    print(f"Journals: {', '.join(JOURNALS)}")
    print(f"Keywords: {', '.join(KEYWORDS)}")
    print(f"Period: 2020-2025")
    print("="*70)

    all_collected = {}

    # Collect from each journal
    for journal in JOURNALS:
        papers = await collect_papers_for_journal(journal)
        all_collected[journal] = papers
        await asyncio.sleep(2)  # Extra delay between journals

    # Convert to paper_urls.json format
    url_file = Path("data/reference_papers/paper_urls_s2.json")

    # Load existing if present
    if url_file.exists():
        with open(url_file, 'r') as f:
            existing_data = json.load(f)
        existing_papers = existing_data.get('papers', [])
        print(f"\nStarting with {len(existing_papers)} existing papers")
    else:
        existing_papers = []
        print("\nStarting fresh")

    # Convert S2 papers to our format
    new_papers = []
    for journal, papers in all_collected.items():
        for paper in papers:
            # Try to get DOI-based URL
            external_ids = paper.get('externalIds', {})
            doi = external_ids.get('DOI')

            if doi:
                url = f"https://doi.org/{doi}"
            else:
                # Fallback to S2 URL
                paper_id = paper.get('paperId')
                url = f"https://www.semanticscholar.org/paper/{paper_id}"

            new_papers.append({
                'url': url,
                'title': paper.get('title', 'Unknown'),
                'venue': journal,
                'year': paper.get('year'),
                'has_open_access': bool(paper.get('openAccessPdf'))
            })

    # Merge with existing (deduplicate by URL)
    existing_urls = {p['url'] for p in existing_papers}
    unique_new = [p for p in new_papers if p['url'] not in existing_urls]

    all_papers = existing_papers + unique_new

    # Save
    output_data = {'papers': all_papers}
    with open(url_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("COLLECTION SUMMARY")
    print(f"{'='*70}")
    for journal in JOURNALS:
        count = sum(1 for p in all_papers if p.get('venue') == journal)
        open_count = sum(1 for p in all_papers if p.get('venue') == journal and p.get('has_open_access'))
        print(f"{journal}: {count} papers ({open_count} open access)")

    print(f"\nTotal papers: {len(all_papers)}")
    print(f"New papers added: {len(unique_new)}")
    print(f"Open access papers: {sum(1 for p in all_papers if p.get('has_open_access'))}")
    print(f"\n✓ Saved to: {url_file}")


if __name__ == '__main__':
    asyncio.run(main())
