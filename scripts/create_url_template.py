#!/usr/bin/env python3
"""
Simplified Paper URL Extractor
Extract paper URLs from Nature search results page that you've already opened in your browser.
"""

import json
from pathlib import Path
from datetime import datetime


def create_url_list_template():
    """Create a template file for manual URL entry."""
    template = {
        "instructions": "Paste paper URLs from Nature search results here",
        "search_url": "https://www-nature-com-ssl.libproxy.snu.ac.kr/search?order=relevance&date_range=2020-2025",
        "papers": [
            {
                "url": "https://www-nature-com-ssl.libproxy.snu.ac.kr/articles/...",
                "title": "Paper title (optional)",
                "notes": "Any notes (optional)"
            }
        ]
    }
    
    output_file = Path("data/reference_papers/paper_urls.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Template created: {output_file}")
    print("\nNext steps:")
    print("1. Open Nature search in your browser (already logged in)")
    print("2. Search with date range 2020-2025")
    print("3. Copy paper URLs from search results")
    print("4. Paste URLs into paper_urls.json")
    print("5. Run: poetry run python scripts/download_papers_from_urls.py")


if __name__ == '__main__':
    create_url_list_template()
