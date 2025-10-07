#!/usr/bin/env python3
"""
arXiv Paper Downloader for Brain Imaging, Neuroscience, and Machine Learning
Downloads papers from arXiv that are publicly available
"""

import arxiv
import os
import time
from datetime import datetime

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/neuroscience_ml_papers"
MAX_PAPERS = 50
DELAY_BETWEEN_DOWNLOADS = 3  # seconds, to be respectful to arXiv servers

# Search queries for different topics
search_queries = [
    "brain imaging machine learning",
    "fMRI deep learning",
    "neuroscience artificial intelligence",
    "neural networks brain",
    "neuroimaging AI",
    "brain decoding machine learning",
    "computational neuroscience deep learning",
    "brain connectivity neural networks",
]

def download_papers():
    """Download papers from arXiv based on search queries"""
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    downloaded_count = 0
    paper_info = []
    seen_ids = set()
    
    print(f"Starting paper download to: {DOWNLOAD_DIR}")
    print(f"Target: {MAX_PAPERS} papers\n")
    
    for query in search_queries:
        if downloaded_count >= MAX_PAPERS:
            break
            
        print(f"Searching for: {query}")
        
        # Search arXiv
        search = arxiv.Search(
            query=query,
            max_results=20,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in search.results():
            if downloaded_count >= MAX_PAPERS:
                break
            
            # Skip if already downloaded
            if result.entry_id in seen_ids:
                continue
            
            seen_ids.add(result.entry_id)
            
            try:
                # Create safe filename
                safe_title = "".join(c for c in result.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title[:100]  # Limit length
                filename = f"{safe_title}_{result.entry_id.split('/')[-1]}.pdf"
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                
                # Download paper
                print(f"\n[{downloaded_count + 1}/{MAX_PAPERS}] Downloading: {result.title[:80]}...")
                result.download_pdf(dirpath=DOWNLOAD_DIR, filename=filename)
                
                # Store paper information
                paper_info.append({
                    'title': result.title,
                    'authors': ', '.join([author.name for author in result.authors]),
                    'published': result.published.strftime('%Y-%m-%d'),
                    'categories': ', '.join(result.categories),
                    'url': result.entry_id,
                    'filename': filename
                })
                
                downloaded_count += 1
                print(f"✓ Downloaded successfully")
                
                # Be respectful to arXiv servers
                time.sleep(DELAY_BETWEEN_DOWNLOADS)
                
            except Exception as e:
                print(f"✗ Error downloading {result.title[:80]}: {str(e)}")
                continue
        
        print(f"\nCompleted query: {query}")
        print(f"Total downloaded so far: {downloaded_count}/{MAX_PAPERS}\n")
    
    # Save paper information to a summary file
    summary_file = os.path.join(DOWNLOAD_DIR, "papers_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Papers Downloaded: {downloaded_count}\n")
        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for i, paper in enumerate(paper_info, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   Authors: {paper['authors']}\n")
            f.write(f"   Published: {paper['published']}\n")
            f.write(f"   Categories: {paper['categories']}\n")
            f.write(f"   URL: {paper['url']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\n{'='*80}")
    print(f"Download Complete!")
    print(f"Total papers downloaded: {downloaded_count}")
    print(f"Papers saved to: {DOWNLOAD_DIR}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    download_papers()
