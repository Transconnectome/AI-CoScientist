#!/usr/bin/env python3
"""
Open Access Journals Paper Downloader
Downloads papers from Scientific Reports, PLOS ONE, and MDPI journals
Focus: Brain imaging, Neuroscience, Machine Learning
"""

import requests
import os
import time
import json
from datetime import datetime

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/open_access_papers"
MAX_PAPERS = 50
DELAY = 2  # seconds between downloads

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

# Target journals
TARGET_JOURNALS = [
    "Scientific Reports",
    "PLOS ONE",
    "PLoS ONE",  # Alternative name
    "MDPI"
]

# Search keywords (same as before)
KEYWORDS = [
    "brain imaging machine learning",
    "fMRI deep learning",
    "neuroscience artificial intelligence",
    "neuroimaging",
    "brain decoding",
    "computational neuroscience deep learning",
    "neural networks brain",
    "EEG machine learning",
    "MEG neuroimaging"
]

def search_semantic_scholar(query, limit=100):
    """Search papers using Semantic Scholar API"""
    url = f"{S2_API_BASE}/paper/search"
    
    params = {
        'query': query,
        'limit': limit,
        'fields': 'title,authors,year,venue,openAccessPdf,externalIds,citationCount,publicationDate',
        'publicationDateOrYear': '2018-'  # Papers from 2018 onwards
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        print(f"Error searching: {e}")
        return []

def download_pdf(url, filepath):
    """Download PDF from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify it's a valid PDF
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                os.remove(filepath)
                return False
        
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def is_target_journal(venue):
    """Check if venue matches target journals"""
    venue_lower = venue.lower()
    
    # Check for exact or partial matches
    if 'scientific reports' in venue_lower:
        return True
    if 'plos one' in venue_lower:
        return True
    if 'mdpi' in venue_lower:
        return True
    
    # Check if it's an MDPI journal (they have many journals)
    mdpi_journals = [
        'sensors', 'applied sciences', 'brain sciences', 'diagnostics',
        'entropy', 'healthcare', 'jcm', 'biomedicines', 'life',
        'bioengineering', 'algorithms', 'mathematics', 'symmetry'
    ]
    
    for mdpi_journal in mdpi_journals:
        if mdpi_journal in venue_lower:
            return True
    
    return False

def main():
    """Main download function"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    downloaded_papers = []
    downloaded_count = 0
    seen_ids = set()
    
    print(f"Searching for Open Access papers...")
    print(f"Target journals: Scientific Reports, PLOS ONE, MDPI journals\n")
    
    # Search for each keyword
    for keyword in KEYWORDS:
        if downloaded_count >= MAX_PAPERS:
            break
            
        print(f"\n{'='*80}")
        print(f"Searching: {keyword}")
        print(f"{'='*80}")
        
        # Search papers
        papers = search_semantic_scholar(keyword)
        print(f"Found {len(papers)} papers")
        
        # Filter and download
        for paper in papers:
            if downloaded_count >= MAX_PAPERS:
                break
            
            venue = paper.get('venue', '')
            paper_id = paper.get('paperId', '')
            
            # Skip if already downloaded
            if paper_id in seen_ids:
                continue
            
            # Check if from target journal
            if not is_target_journal(venue):
                continue
            
            # Check if PDF is available
            pdf_info = paper.get('openAccessPdf')
            if not pdf_info or not pdf_info.get('url'):
                continue
            
            seen_ids.add(paper_id)
            
            # Prepare filename
            title = paper.get('title', 'Untitled')
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title[:70]
            year = paper.get('year', 'unknown')
            
            # Clean venue name for filename
            venue_short = venue.replace(' ', '_')[:20]
            filename = f"{venue_short}_{year}_{safe_title}.pdf"
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            
            # Download
            print(f"\n[{downloaded_count + 1}/{MAX_PAPERS}]")
            print(f"Title: {title[:70]}...")
            print(f"Venue: {venue}")
            print(f"Year: {year}")
            print(f"Citations: {paper.get('citationCount', 0)}")
            print(f"URL: {pdf_info['url'][:60]}...")
            
            if download_pdf(pdf_info['url'], filepath):
                print(f"✓ Downloaded successfully to {filename}")
                
                downloaded_papers.append({
                    'title': title,
                    'authors': ', '.join([a.get('name', '') for a in paper.get('authors', [])][:3]),
                    'year': year,
                    'venue': venue,
                    'citations': paper.get('citationCount', 0),
                    'doi': paper.get('externalIds', {}).get('DOI', ''),
                    'pubDate': paper.get('publicationDate', ''),
                    'filename': filename,
                    'keyword': keyword
                })
                
                downloaded_count += 1
            else:
                print(f"✗ Failed to download")
            
            time.sleep(DELAY)
        
        print(f"\nProgress: {downloaded_count}/{MAX_PAPERS} papers downloaded")
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "papers_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Open Access Papers Downloaded: {downloaded_count}\n")
        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Journals: Scientific Reports, PLOS ONE, MDPI\n")
        f.write("="*80 + "\n\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   Authors: {paper['authors']}\n")
            f.write(f"   Year: {paper['year']}\n")
            f.write(f"   Venue: {paper['venue']}\n")
            f.write(f"   Citations: {paper['citations']}\n")
            f.write(f"   Publication Date: {paper['pubDate']}\n")
            f.write(f"   DOI: {paper['doi']}\n")
            f.write(f"   Keyword: {paper['keyword']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    # Save JSON metadata
    json_file = os.path.join(DOWNLOAD_DIR, "papers_metadata.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(downloaded_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Download Complete!")
    print(f"Total papers downloaded: {downloaded_count}")
    print(f"Papers saved to: {DOWNLOAD_DIR}")
    print(f"Summary saved to: {summary_file}")
    print(f"Metadata saved to: {json_file}")
    print(f"{'='*80}")
    
    # Print journal breakdown
    journal_counts = {}
    for paper in downloaded_papers:
        venue = paper['venue']
        journal_counts[venue] = journal_counts.get(venue, 0) + 1
    
    print("\nPapers by Journal:")
    for journal, count in sorted(journal_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {journal}: {count}")
    
    # Print keyword breakdown
    keyword_counts = {}
    for paper in downloaded_papers:
        keyword = paper['keyword']
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    print("\nPapers by Keyword:")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {keyword}: {count}")

if __name__ == "__main__":
    main()
