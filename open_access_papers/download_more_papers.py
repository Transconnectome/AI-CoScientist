#!/usr/bin/env python3
"""
Extended Open Access Paper Downloader
Downloads more papers from Scientific Reports, PLOS ONE, MDPI journals
Using broader search terms
"""

import requests
import os
import time
from datetime import datetime

DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/open_access_papers"
TARGET_TOTAL = 50
DELAY = 3  # seconds between requests
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

# Expanded search keywords
KEYWORDS = [
    "neuroimaging convolutional neural network",
    "brain fMRI artificial intelligence", 
    "EEG classification deep learning",
    "MEG source localization machine learning",
    "resting state fMRI connectivity",
    "diffusion MRI tractography",
    "brain parcellation segmentation",
    "neurological disorder prediction",
    "cognitive neuroscience neural networks",
    "brain-computer interface deep learning"
]

def is_target_journal(venue):
    """Check if venue is one of our target journals"""
    venue_lower = venue.lower()
    
    target_names = [
        'scientific reports', 'plos one', 'sensors', 'applied sciences',
        'brain sciences', 'diagnostics', 'entropy', 'healthcare',
        'biomedicines', 'life', 'bioengineering', 'algorithms',
        'mathematics', 'symmetry', 'jcm', 'mdpi'
    ]
    
    return any(name in venue_lower for name in target_names)

def search_semantic_scholar(query, limit=100):
    """Search papers using Semantic Scholar API"""
    url = f"{S2_API_BASE}/paper/search"
    
    params = {
        'query': query,
        'limit': limit,
        'fields': 'title,authors,year,venue,openAccessPdf,citationCount',
        'year': '2018-'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        print(f"Search error: {e}")
        return []

def download_pdf(url, filepath):
    """Download PDF"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify PDF
        with open(filepath, 'rb') as f:
            if f.read(4) != b'%PDF':
                os.remove(filepath)
                return False
        return True
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    # Count existing papers
    existing = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.pdf')]
    current_count = len(existing)
    
    print(f"Current papers: {current_count}/{TARGET_TOTAL}")
    print(f"Need {TARGET_TOTAL - current_count} more papers\n")
    
    if current_count >= TARGET_TOTAL:
        print("Target already reached!")
        return
    
    downloaded_papers = []
    seen_ids = set()
    
    for keyword in KEYWORDS:
        if current_count >= TARGET_TOTAL:
            break
        
        print(f"\nSearching: {keyword}")
        papers = search_semantic_scholar(keyword)
        print(f"Found {len(papers)} papers")
        
        for paper in papers:
            if current_count >= TARGET_TOTAL:
                break
            
            venue = paper.get('venue', '')
            paper_id = paper.get('paperId', '')
            
            if paper_id in seen_ids or not is_target_journal(venue):
                continue
            
            pdf_info = paper.get('openAccessPdf')
            if not pdf_info or not pdf_info.get('url'):
                continue
            
            seen_ids.add(paper_id)
            
            title = paper.get('title', 'Untitled')
            safe_title = "".join(c for c in title if c.isalnum() or c in ' -_')[:60]
            year = paper.get('year', 'unknown')
            filename = f"{venue.replace(' ', '_')[:20]}_{year}_{safe_title}.pdf"
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            
            if os.path.exists(filepath):
                continue
            
            print(f"  [{current_count + 1}/{TARGET_TOTAL}] {title[:50]}... ", end='')
            
            if download_pdf(pdf_info['url'], filepath):
                print("✓")
                current_count += 1
                downloaded_papers.append({
                    'title': title,
                    'venue': venue,
                    'year': year,
                    'citations': paper.get('citationCount', 0),
                    'keyword': keyword
                })
            else:
                print("✗")
            
            time.sleep(DELAY)
        
        print(f"Progress: {current_count}/{TARGET_TOTAL}")
    
    # Save summary
    if downloaded_papers:
        summary_file = os.path.join(DOWNLOAD_DIR, "extended_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Additional Papers Downloaded: {len(downloaded_papers)}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for i, p in enumerate(downloaded_papers, 1):
                f.write(f"{i}. {p['title']}\n")
                f.write(f"   Venue: {p['venue']} ({p['year']})\n")
                f.write(f"   Citations: {p['citations']}\n")
                f.write(f"   Keyword: {p['keyword']}\n\n")
    
    print(f"\n{'='*60}")
    print(f"Final count: {current_count}/{TARGET_TOTAL} papers")
    print(f"Downloaded this session: {len(downloaded_papers)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
