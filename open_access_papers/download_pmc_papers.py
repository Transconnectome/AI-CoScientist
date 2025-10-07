#!/usr/bin/env python3
"""
PubMed Central Open Access Paper Downloader
Uses NCBI E-utilities API to download papers from PMC
Target journals: Scientific Reports, PLOS ONE, MDPI
"""

import requests
import xml.etree.ElementTree as ET
import os
import time
from datetime import datetime

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/open_access_papers"
MAX_PAPERS = 50
DELAY = 0.34  # ~3 requests per second (NCBI guideline without API key)

# NCBI E-utilities endpoints
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_PDF_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles/"

# Search queries
SEARCH_QUERIES = [
    '("brain imaging"[Title/Abstract] OR "neuroimaging"[Title/Abstract]) AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract])',
    '"fMRI"[Title/Abstract] AND ("deep learning"[Title/Abstract] OR "neural network"[Title/Abstract])',
    '"neuroscience"[Title/Abstract] AND "artificial intelligence"[Title/Abstract]',
    '"brain decoding"[Title/Abstract] AND "machine learning"[Title/Abstract]',
    '"EEG"[Title/Abstract] AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract])',
    '"computational neuroscience"[Title/Abstract] AND "deep learning"[Title/Abstract]',
]

# Target journals filter
JOURNAL_FILTERS = [
    'AND ("Scientific Reports"[Journal])',
    'AND ("PLoS One"[Journal])',
    'AND ("MDPI"[Journal] OR "Sensors"[Journal] OR "Brain Sciences"[Journal] OR "Applied Sciences"[Journal])',
]

def search_pmc(query, retmax=100):
    """Search PubMed Central for papers"""
    params = {
        'db': 'pmc',
        'term': query + ' AND open access[filter]',  # Only open access papers
        'retmax': retmax,
        'retmode': 'xml',
        'sort': 'relevance'
    }
    
    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        id_list = [id_elem.text for id_elem in root.findall('.//Id')]
        
        return id_list
    except Exception as e:
        print(f"Search error: {e}")
        return []

def fetch_paper_metadata(pmc_ids):
    """Fetch metadata for PMC papers"""
    if not pmc_ids:
        return []
    
    # Batch fetch (max 200 IDs at a time)
    params = {
        'db': 'pmc',
        'id': ','.join(pmc_ids[:200]),
        'retmode': 'xml'
    }
    
    try:
        response = requests.get(EFETCH_URL, params=params, timeout=60)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        papers = []
        
        for article in root.findall('.//article'):
            try:
                # Extract metadata
                title_elem = article.find('.//article-title')
                title = ''.join(title_elem.itertext()) if title_elem is not None else 'Untitled'
                
                # Get PMC ID
                pmc_id = None
                for article_id in article.findall('.//article-id'):
                    if article_id.get('pub-id-type') == 'pmc':
                        pmc_id = article_id.text
                        break
                
                if not pmc_id:
                    continue
                
                # Get journal name
                journal_elem = article.find('.//journal-title')
                journal = ''.join(journal_elem.itertext()) if journal_elem is not None else 'Unknown'
                
                # Get year
                year_elem = article.find('.//pub-date/year')
                year = year_elem.text if year_elem is not None else 'Unknown'
                
                # Get authors
                authors = []
                for contrib in article.findall('.//contrib[@contrib-type="author"]'):
                    surname = contrib.find('.//surname')
                    given_names = contrib.find('.//given-names')
                    if surname is not None:
                        name = surname.text
                        if given_names is not None:
                            name = f"{given_names.text} {name}"
                        authors.append(name)
                
                papers.append({
                    'pmc_id': pmc_id,
                    'title': title,
                    'journal': journal,
                    'year': year,
                    'authors': ', '.join(authors[:5])
                })
                
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue
        
        return papers
        
    except Exception as e:
        print(f"Fetch error: {e}")
        return []

def download_pmc_pdf(pmc_id, filepath):
    """Download PDF from PMC"""
    # Try PDF download URL
    pdf_url = f"{PMC_PDF_BASE}PMC{pmc_id}/pdf/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(pdf_url, headers=headers, timeout=60, allow_redirects=True)
        
        # Check if we got redirected to the actual PDF
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('application/pdf'):
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify PDF
            with open(filepath, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    os.remove(filepath)
                    return False
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  Download error: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    """Main download function"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    # Load existing papers to avoid duplicates
    existing_files = set(os.listdir(DOWNLOAD_DIR))
    
    downloaded_papers = []
    downloaded_count = 0
    seen_pmc_ids = set()
    
    print(f"Downloading papers from PubMed Central...")
    print(f"Target: {MAX_PAPERS} papers")
    print(f"Target journals: Scientific Reports, PLOS ONE, MDPI\n")
    
    # Try each query
    for query in SEARCH_QUERIES:
        if downloaded_count >= MAX_PAPERS:
            break
        
        print(f"\n{'='*80}")
        print(f"Query: {query[:70]}...")
        print(f"{'='*80}")
        
        # Try with each journal filter
        for journal_filter in JOURNAL_FILTERS:
            if downloaded_count >= MAX_PAPERS:
                break
                
            full_query = query + journal_filter
            print(f"\nSearching with filter: {journal_filter}")
            
            # Search
            pmc_ids = search_pmc(full_query, retmax=50)
            print(f"Found {len(pmc_ids)} papers")
            
            if not pmc_ids:
                continue
            
            time.sleep(DELAY)
            
            # Fetch metadata
            papers = fetch_paper_metadata(pmc_ids)
            print(f"Retrieved metadata for {len(papers)} papers")
            
            time.sleep(DELAY)
            
            # Download papers
            for paper in papers:
                if downloaded_count >= MAX_PAPERS:
                    break
                
                pmc_id = paper['pmc_id']
                
                # Skip if already processed
                if pmc_id in seen_pmc_ids:
                    continue
                
                seen_pmc_ids.add(pmc_id)
                
                # Create filename
                title = paper['title']
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
                safe_title = safe_title[:60]
                year = paper['year']
                journal_short = paper['journal'].replace(' ', '_')[:20]
                
                filename = f"PMC{pmc_id}_{journal_short}_{year}_{safe_title}.pdf"
                
                # Skip if already exists
                if filename in existing_files:
                    print(f"\nSkipping (already exists): {title[:60]}...")
                    continue
                
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                
                # Download
                print(f"\n[{downloaded_count + 1}/{MAX_PAPERS}]")
                print(f"PMC ID: PMC{pmc_id}")
                print(f"Title: {title[:70]}...")
                print(f"Journal: {paper['journal']}")
                print(f"Year: {year}")
                print(f"Authors: {paper['authors'][:80]}...")
                
                if download_pmc_pdf(pmc_id, filepath):
                    print(f"✓ Downloaded successfully")
                    
                    downloaded_papers.append({
                        'pmc_id': pmc_id,
                        'title': title,
                        'authors': paper['authors'],
                        'year': year,
                        'journal': paper['journal'],
                        'filename': filename,
                        'url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
                    })
                    
                    downloaded_count += 1
                else:
                    print(f"✗ Failed to download")
                
                time.sleep(DELAY)
        
        print(f"\nProgress: {downloaded_count}/{MAX_PAPERS} papers")
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "pmc_papers_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"PubMed Central Papers Downloaded: {downloaded_count}\n")
        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: PubMed Central (PMC)\n")
        f.write("="*80 + "\n\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   PMC ID: PMC{paper['pmc_id']}\n")
            f.write(f"   Authors: {paper['authors']}\n")
            f.write(f"   Year: {paper['year']}\n")
            f.write(f"   Journal: {paper['journal']}\n")
            f.write(f"   URL: {paper['url']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\n{'='*80}")
    print(f"Download Complete!")
    print(f"Total papers downloaded: {downloaded_count}")
    print(f"Papers saved to: {DOWNLOAD_DIR}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")
    
    # Print journal breakdown
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
