#!/usr/bin/env python3
"""
STRICT Top-5 Venue Paper Downloader
Downloads ONLY from: Nature, Science, NeurIPS, ICLR, AAAI
Using Semantic Scholar API with strict venue filtering
"""

import requests
import os
import time

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/neuroscience_ml_papers"
MAX_PAPERS = 50
DELAY = 2
S2_API = "https://api.semanticscholar.org/graph/v1"

# STRICT venue filter - only exact matches
STRICT_VENUES = {
    'nature', 'science', 'neurips', 'nips', 'iclr', 'aaai'
}

KEYWORDS = [
    "brain imaging machine learning",
    "fMRI deep learning",
    "neuroimaging artificial intelligence",
    "brain decoding neural networks",
    "computational neuroscience deep learning",
    "brain connectivity machine learning"
]

def is_top5_venue(venue_str):
    """STRICT check: only these 5 venues allowed"""
    if not venue_str:
        return False
    
    venue_lower = venue_str.lower().strip()
    
    # Remove common suffixes/prefixes
    venue_clean = venue_lower.replace('conference on', '').replace('proceedings of', '').strip()
    
    # Check if ANY of the strict venue names match
    for allowed in STRICT_VENUES:
        if allowed in venue_clean.split():
            return True
        if venue_clean == allowed:
            return True
    
    return False

def search_papers(query):
    """Search Semantic Scholar with strict venue filtering"""
    url = f"{S2_API}/paper/search"
    
    params = {
        'query': query,
        'limit': 100,
        'fields': 'title,authors,year,venue,openAccessPdf,externalIds,citationCount,publicationTypes'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # STRICT filtering
        filtered = []
        for paper in data.get('data', []):
            venue = paper.get('venue', '')
            
            # Must be from top 5 venues
            if not is_top5_venue(venue):
                continue
            
            # Must have open access PDF
            pdf_info = paper.get('openAccessPdf')
            if not pdf_info or not pdf_info.get('url'):
                continue
            
            # Must be published (not preprint)
            pub_types = paper.get('publicationTypes', [])
            if 'JournalArticle' not in pub_types and 'Conference' not in pub_types:
                continue
            
            filtered.append(paper)
        
        return filtered
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def download_pdf(url, filepath):
    """Download PDF file"""
    try:
        print(f"  Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file size
        size = os.path.getsize(filepath)
        if size < 10000:  # Less than 10KB is suspicious
            os.remove(filepath)
            return False
        
        return True
        
    except Exception as e:
        print(f"  Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    """Main download function with strict venue filtering"""
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    print("="*80)
    print("STRICT Top-5 Venue Paper Downloader")
    print("ONLY downloading from: Nature, Science, NeurIPS, ICLR, AAAI")
    print("="*80)
    print()
    
    downloaded = []
    downloaded_count = 0
    seen_ids = set()
    
    for keyword in KEYWORDS:
        if downloaded_count >= MAX_PAPERS:
            break
        
        print(f"\n{'='*80}")
        print(f"Searching: {keyword}")
        print(f"{'='*80}\n")
        
        papers = search_papers(keyword)
        print(f"Found {len(papers)} papers from top-5 venues with open access PDFs\n")
        
        for paper in papers:
            if downloaded_count >= MAX_PAPERS:
                break
            
            paper_id = paper.get('paperId', '')
            if paper_id in seen_ids:
                continue
            
            seen_ids.add(paper_id)
            
            # Extract info
            title = paper.get('title', 'Untitled')
            venue = paper.get('venue', '')
            year = paper.get('year', 'unknown')
            citations = paper.get('citationCount', 0)
            pdf_url = paper['openAccessPdf']['url']
            authors = ', '.join([a.get('name', '') for a in paper.get('authors', [])])
            
            # Create filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title[:70].strip()
            filename = f"{safe_title}_{year}.pdf"
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            
            # Download
            print(f"[{downloaded_count + 1}/{MAX_PAPERS}]")
            print(f"Title: {title[:75]}...")
            print(f"Venue: {venue}")
            print(f"Year: {year} | Citations: {citations}")
            
            if download_pdf(pdf_url, filepath):
                print(f"✓ Successfully downloaded\n")
                
                downloaded.append({
                    'title': title,
                    'authors': authors,
                    'venue': venue,
                    'year': year,
                    'citations': citations,
                    'doi': paper.get('externalIds', {}).get('DOI', ''),
                    'filename': filename
                })
                
                downloaded_count += 1
            else:
                print(f"✗ Download failed\n")
            
            time.sleep(DELAY)
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "TOP5_VENUES_SUMMARY.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("STRICT TOP-5 VENUES ONLY\n")
        f.write("Nature | Science | NeurIPS | ICLR | AAAI\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Downloaded: {downloaded_count}\n")
        f.write(f"Download Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Venue breakdown
        venue_counts = {}
        for p in downloaded:
            v = p['venue']
            venue_counts[v] = venue_counts.get(v, 0) + 1
        
        f.write("Papers by Venue:\n")
        for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {venue}: {count}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed list
        for i, paper in enumerate(downloaded, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   Venue: {paper['venue']} ({paper['year']})\n")
            f.write(f"   Authors: {paper['authors'][:200]}...\n" if len(paper['authors']) > 200 else f"   Authors: {paper['authors']}\n")
            f.write(f"   Citations: {paper['citations']}\n")
            if paper['doi']:
                f.write(f"   DOI: {paper['doi']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print(f"Total papers: {downloaded_count}")
    print(f"Saved to: {DOWNLOAD_DIR}")
    print(f"Summary: {summary_file}")
    print("="*80)
    
    print("\nVenue Breakdown:")
    for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {venue}: {count}")
    print()

if __name__ == "__main__":
    main()
