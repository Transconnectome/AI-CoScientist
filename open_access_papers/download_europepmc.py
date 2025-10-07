#!/usr/bin/env python3
"""
Europe PMC Paper Downloader
Uses Europe PMC API - more reliable for open access papers
Target journals: Scientific Reports, PLOS ONE, MDPI
"""

import requests
import os
import time
import json
from datetime import datetime

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/open_access_papers"
MAX_PAPERS = 50
DELAY = 1  # seconds between requests

# Europe PMC API
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Search queries with journal filters
SEARCH_QUERIES = [
    'TITLE_ABS:("brain imaging" OR neuroimaging) AND TITLE_ABS:("machine learning" OR "deep learning") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
    'TITLE_ABS:(fMRI) AND TITLE_ABS:("deep learning" OR "neural network") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
    'TITLE_ABS:(neuroscience) AND TITLE_ABS:("artificial intelligence") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
    'TITLE_ABS:("brain decoding") AND TITLE_ABS:("machine learning") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
    'TITLE_ABS:(EEG) AND TITLE_ABS:("machine learning" OR "deep learning") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
    'TITLE_ABS:("computational neuroscience") AND TITLE_ABS:("deep learning") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
    'TITLE_ABS:(MEG OR magnetoencephalography) AND TITLE_ABS:("machine learning") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
    'TITLE_ABS:("brain connectivity") AND TITLE_ABS:("neural network") AND (SRC:"Scientific Reports" OR SRC:"PLoS One" OR SRC:"MDPI")',
]

def search_europe_pmc(query, page_size=100):
    """Search Europe PMC"""
    params = {
        'query': query + ' AND OPEN_ACCESS:Y',  # Only open access
        'format': 'json',
        'pageSize': page_size,
        'cursorMark': '*'
    }
    
    try:
        response = requests.get(EPMC_SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return data.get('resultList', {}).get('result', [])
    except Exception as e:
        print(f"Search error: {e}")
        return []

def download_pdf(url, filepath):
    """Download PDF from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify PDF
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

def main():
    """Main download function"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    # Load existing papers
    existing_files = set(os.listdir(DOWNLOAD_DIR))
    
    downloaded_papers = []
    downloaded_count = 0
    seen_ids = set()
    
    print(f"Downloading papers from Europe PMC...")
    print(f"Target: {MAX_PAPERS} papers")
    print(f"Target journals: Scientific Reports, PLOS ONE, MDPI\n")
    
    for query_idx, query in enumerate(SEARCH_QUERIES):
        if downloaded_count >= MAX_PAPERS:
            break
        
        print(f"\n{'='*80}")
        print(f"Query {query_idx + 1}/{len(SEARCH_QUERIES)}")
        print(f"{'='*80}")
        
        # Search
        results = search_europe_pmc(query)
        print(f"Found {len(results)} papers")
        
        if not results:
            continue
        
        time.sleep(DELAY)
        
        # Process results
        for result in results:
            if downloaded_count >= MAX_PAPERS:
                break
            
            # Get IDs
            paper_id = result.get('id', '')
            pmcid = result.get('pmcid', '')
            
            # Skip if already processed
            if paper_id in seen_ids:
                continue
            
            seen_ids.add(paper_id)
            
            # Check if PDF available
            has_pdf = result.get('hasPDF', 'N') == 'Y'
            full_text_urls = result.get('fullTextUrlList', {}).get('fullTextUrl', [])
            
            # Find PDF URL
            pdf_url = None
            for url_info in full_text_urls:
                if url_info.get('documentStyle') == 'pdf':
                    pdf_url = url_info.get('url')
                    break
            
            if not pdf_url or not has_pdf:
                continue
            
            # Get metadata
            title = result.get('title', 'Untitled')
            journal = result.get('journalTitle', 'Unknown')
            year = result.get('pubYear', 'Unknown')
            
            # Authors
            authors = []
            for author in result.get('authorList', {}).get('author', [])[:5]:
                full_name = author.get('fullName', '')
                if full_name:
                    authors.append(full_name)
            
            # Create filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title[:60]
            journal_short = journal.replace(' ', '_')[:20]
            
            filename = f"EPMC_{journal_short}_{year}_{safe_title}.pdf"
            
            # Skip if exists
            if filename in existing_files:
                print(f"Skipping (exists): {title[:60]}...")
                continue
            
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            
            # Download
            print(f"\n[{downloaded_count + 1}/{MAX_PAPERS}]")
            print(f"Title: {title[:70]}...")
            print(f"Journal: {journal}")
            print(f"Year: {year}")
            if pmcid:
                print(f"PMC ID: {pmcid}")
            print(f"PDF URL: {pdf_url[:60]}...")
            
            if download_pdf(pdf_url, filepath):
                print(f"✓ Downloaded successfully")
                
                downloaded_papers.append({
                    'id': paper_id,
                    'pmcid': pmcid,
                    'title': title,
                    'authors': ', '.join(authors),
                    'year': year,
                    'journal': journal,
                    'doi': result.get('doi', ''),
                    'filename': filename
                })
                
                downloaded_count += 1
            else:
                print(f"✗ Failed to download")
            
            time.sleep(DELAY)
        
        print(f"\nProgress: {downloaded_count}/{MAX_PAPERS} papers")
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "europepmc_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Europe PMC Papers Downloaded: {downloaded_count}\n")
        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: Europe PMC\n")
        f.write("="*80 + "\n\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   ID: {paper['id']}\n")
            if paper['pmcid']:
                f.write(f"   PMC ID: {paper['pmcid']}\n")
            f.write(f"   Authors: {paper['authors']}\n")
            f.write(f"   Year: {paper['year']}\n")
            f.write(f"   Journal: {paper['journal']}\n")
            f.write(f"   DOI: {paper['doi']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    # Save JSON
    json_file = os.path.join(DOWNLOAD_DIR, "europepmc_metadata.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(downloaded_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Download Complete!")
    print(f"Total papers downloaded: {downloaded_count}")
    print(f"Papers saved to: {DOWNLOAD_DIR}")
    print(f"Summary: {summary_file}")
    print(f"Metadata: {json_file}")
    print(f"{'='*80}")
    
    # Statistics
    if downloaded_papers:
        journal_counts = {}
        for paper in downloaded_papers:
            journal = paper['journal']
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
        
        print("\nPapers by Journal:")
        for journal, count in sorted(journal_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {journal}: {count}")

if __name__ == "__main__":
    main()
