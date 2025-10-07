#!/usr/bin/env python3
"""
Top-tier Venue Paper Downloader
Downloads papers from Nature, Science, NeurIPS, ICLR, AAAI, etc.
Using Semantic Scholar API and direct venue sources
"""

import requests
import os
import time
import json
from urllib.parse import quote

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/neuroscience_ml_papers"
MAX_PAPERS = 50
DELAY = 2  # seconds between requests

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

# Target venues
TARGET_VENUES = [
    "Nature",
    "Science", 
    "Nature Neuroscience",
    "Nature Methods",
    "NeurIPS",
    "ICLR",
    "AAAI",
    "ICML",
    "Nature Communications"
]

# Search keywords
KEYWORDS = [
    "brain imaging machine learning",
    "fMRI deep learning",
    "neuroscience artificial intelligence",
    "neuroimaging",
    "brain decoding",
    "computational neuroscience"
]

def search_semantic_scholar(query, venue=None):
    """Search papers using Semantic Scholar API"""
    url = f"{S2_API_BASE}/paper/search"
    
    params = {
        'query': query,
        'limit': 100,
        'fields': 'title,authors,year,venue,openAccessPdf,externalIds,citationCount'
    }
    
    if venue:
        params['venue'] = venue
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        print(f"Error searching: {e}")
        return []

def download_pdf(url, filepath):
    """Download PDF from URL"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

def main():
    """Main download function"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    downloaded_papers = []
    downloaded_count = 0
    seen_ids = set()
    
    print(f"Searching for top-tier papers...")
    print(f"Target venues: {', '.join(TARGET_VENUES)}\n")
    
    # Search for each keyword
    for keyword in KEYWORDS:
        if downloaded_count >= MAX_PAPERS:
            break
            
        print(f"\nSearching: {keyword}")
        
        # Search all papers for this keyword
        papers = search_semantic_scholar(keyword)
        
        # Filter by venue and availability
        for paper in papers:
            if downloaded_count >= MAX_PAPERS:
                break
            
            venue = paper.get('venue', '')
            paper_id = paper.get('paperId', '')
            
            # Skip if already downloaded
            if paper_id in seen_ids:
                continue
            
            # Check if from target venue
            is_target_venue = any(target.lower() in venue.lower() for target in TARGET_VENUES)
            
            if not is_target_venue:
                continue
            
            # Check if PDF is available
            pdf_info = paper.get('openAccessPdf')
            if not pdf_info or not pdf_info.get('url'):
                continue
            
            seen_ids.add(paper_id)
            
            # Prepare filename
            title = paper.get('title', 'Untitled')
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title[:80]
            year = paper.get('year', 'unknown')
            filename = f"{safe_title}_{year}.pdf"
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            
            # Download
            print(f"\n[{downloaded_count + 1}/{MAX_PAPERS}]")
            print(f"Title: {title[:70]}...")
            print(f"Venue: {venue}")
            print(f"Year: {year}")
            print(f"Citations: {paper.get('citationCount', 0)}")
            
            if download_pdf(pdf_info['url'], filepath):
                print(f"✓ Downloaded successfully")
                
                downloaded_papers.append({
                    'title': title,
                    'authors': ', '.join([a.get('name', '') for a in paper.get('authors', [])]),
                    'year': year,
                    'venue': venue,
                    'citations': paper.get('citationCount', 0),
                    'doi': paper.get('externalIds', {}).get('DOI', ''),
                    'filename': filename
                })
                
                downloaded_count += 1
            else:
                print(f"✗ Failed to download")
            
            time.sleep(DELAY)
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "top_tier_papers_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Top-Tier Papers Downloaded: {downloaded_count}\n")
        f.write(f"Target Venues: {', '.join(TARGET_VENUES)}\n")
        f.write("="*80 + "\n\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   Authors: {paper['authors']}\n")
            f.write(f"   Year: {paper['year']}\n")
            f.write(f"   Venue: {paper['venue']}\n")
            f.write(f"   Citations: {paper['citations']}\n")
            f.write(f"   DOI: {paper['doi']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\n{'='*80}")
    print(f"Download Complete!")
    print(f"Total papers downloaded: {downloaded_count}")
    print(f"Papers saved to: {DOWNLOAD_DIR}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")
    
    # Print venue breakdown
    venue_counts = {}
    for paper in downloaded_papers:
        venue = paper['venue']
        venue_counts[venue] = venue_counts.get(venue, 0) + 1
    
    print("\nPapers by Venue:")
    for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {venue}: {count}")

if __name__ == "__main__":
    main()
