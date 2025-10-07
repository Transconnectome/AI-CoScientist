#!/usr/bin/env python3
"""
NCBI PubMed Central Paper Downloader
Uses E-utilities API to download open access papers
Focus: Brain imaging, Neuroscience, Machine Learning
Journals: Scientific Reports, PLOS ONE, MDPI
"""

import requests
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/open_access_papers"
MAX_PAPERS = 50
DELAY = 0.5  # NCBI requests max 3 requests per second

# NCBI E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Email for NCBI (required for API usage)
EMAIL = "your_email@example.com"  # Please use your real email

# Search queries combining keywords and journals
SEARCH_QUERIES = [
    '("brain imaging"[Title/Abstract] OR "neuroimaging"[Title/Abstract]) AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract]) AND ("Scientific Reports"[Journal] OR "PLoS ONE"[Journal] OR "PLOS ONE"[Journal])',
    '("fMRI"[Title/Abstract] OR "functional MRI"[Title/Abstract]) AND ("deep learning"[Title/Abstract] OR "neural network"[Title/Abstract]) AND ("Scientific Reports"[Journal] OR "PLoS ONE"[Journal])',
    '("neuroscience"[Title/Abstract]) AND ("artificial intelligence"[Title/Abstract] OR "machine learning"[Title/Abstract]) AND ("Scientific Reports"[Journal] OR "PLoS ONE"[Journal] OR "PLOS ONE"[Journal])',
    '("brain decoding"[Title/Abstract] OR "neural decoding"[Title/Abstract]) AND ("Scientific Reports"[Journal] OR "PLoS ONE"[Journal] OR "PLOS ONE"[Journal])',
    '("EEG"[Title/Abstract] OR "MEG"[Title/Abstract]) AND ("machine learning"[Title/Abstract]) AND ("Scientific Reports"[Journal] OR "PLoS ONE"[Journal] OR "PLOS ONE"[Journal])',
    '("computational neuroscience"[Title/Abstract]) AND ("deep learning"[Title/Abstract]) AND ("Scientific Reports"[Journal] OR "PLoS ONE"[Journal])',
    # MDPI journals
    '("brain imaging"[Title/Abstract] OR "neuroimaging"[Title/Abstract]) AND ("machine learning"[Title/Abstract]) AND ("Sensors (Basel)"[Journal] OR "Brain Sci"[Journal] OR "Appl Sci"[Journal])',
    '("neuroscience"[Title/Abstract]) AND ("machine learning"[Title/Abstract]) AND ("Entropy (Basel)"[Journal] OR "Diagnostics (Basel)"[Journal])',
]

def search_pubmed(query, retmax=100):
    """Search PubMed Central using E-utilities"""
    params = {
        'db': 'pmc',  # PubMed Central for full-text articles
        'term': query,
        'retmax': retmax,
        'retmode': 'xml',
        'email': EMAIL,
        'sort': 'relevance',
        'mindate': '2018/01/01',  # From 2018 onwards
        'maxdate': '2024/12/31'
    }
    
    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        id_list = [id_elem.text for id_elem in root.findall('.//Id')]
        
        return id_list
    except Exception as e:
        print(f"  Error searching: {e}")
        return []

def get_paper_metadata(pmc_ids):
    """Get metadata for papers using ESummary"""
    if not pmc_ids:
        return []
    
    params = {
        'db': 'pmc',
        'id': ','.join(pmc_ids),
        'retmode': 'xml',
        'email': EMAIL
    }
    
    try:
        response = requests.get(ESUMMARY_URL, params=params, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        papers = []
        
        for doc_sum in root.findall('.//DocSum'):
            pmc_id = doc_sum.find('./Id').text
            
            # Extract metadata
            title = ''
            authors = ''
            journal = ''
            pub_date = ''
            doi = ''
            
            for item in doc_sum.findall('.//Item'):
                name = item.get('Name')
                if name == 'Title':
                    title = item.text or ''
                elif name == 'AuthorList':
                    author_list = [a.text for a in item.findall('.//Item') if a.text]
                    authors = ', '.join(author_list[:3])  # First 3 authors
                elif name == 'FullJournalName':
                    journal = item.text or ''
                elif name == 'PubDate':
                    pub_date = item.text or ''
                elif name == 'DOI':
                    doi = item.text or ''
            
            papers.append({
                'pmc_id': pmc_id,
                'title': title,
                'authors': authors,
                'journal': journal,
                'pub_date': pub_date,
                'doi': doi
            })
        
        return papers
    except Exception as e:
        print(f"  Error getting metadata: {e}")
        return []

def download_pdf(pmc_id, filepath):
    """Download PDF from PubMed Central"""
    # Try to get PDF link
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(pdf_url, stream=True, timeout=60, headers=headers, allow_redirects=True)
        
        # Check if we got a PDF
        content_type = response.headers.get('content-type', '')
        if 'application/pdf' not in content_type:
            # Try alternative URL format
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/main.pdf"
            response = requests.get(pdf_url, stream=True, timeout=60, headers=headers, allow_redirects=True)
            content_type = response.headers.get('content-type', '')
            
            if 'application/pdf' not in content_type:
                return False
        
        response.raise_for_status()
        
        # Save PDF
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

def main():
    """Main download function"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    downloaded_papers = []
    downloaded_count = 0
    seen_ids = set()
    
    # Count existing files to continue from where we left off
    existing_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.pdf')]
    downloaded_count = len(existing_files)
    
    print(f"NCBI PubMed Central Paper Downloader")
    print(f"Target: {MAX_PAPERS} papers")
    print(f"Already downloaded: {downloaded_count} papers")
    print(f"Remaining: {MAX_PAPERS - downloaded_count} papers\n")
    
    # Search with each query
    for query_idx, query in enumerate(SEARCH_QUERIES, 1):
        if downloaded_count >= MAX_PAPERS:
            break
        
        print(f"\n{'='*80}")
        print(f"Query {query_idx}/{len(SEARCH_QUERIES)}")
        print(f"{'='*80}")
        
        # Search for papers
        pmc_ids = search_pubmed(query)
        print(f"Found {len(pmc_ids)} papers")
        
        if not pmc_ids:
            continue
        
        # Get metadata in batches
        batch_size = 20
        for i in range(0, len(pmc_ids), batch_size):
            if downloaded_count >= MAX_PAPERS:
                break
            
            batch_ids = pmc_ids[i:i+batch_size]
            papers = get_paper_metadata(batch_ids)
            
            print(f"\nProcessing batch {i//batch_size + 1} ({len(batch_ids)} papers)...")
            
            # Download each paper
            for paper in papers:
                if downloaded_count >= MAX_PAPERS:
                    break
                
                pmc_id = paper['pmc_id']
                
                # Skip if already seen
                if pmc_id in seen_ids:
                    continue
                
                seen_ids.add(pmc_id)
                
                # Create filename
                title = paper['title']
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
                safe_title = safe_title[:60]
                
                journal_short = paper['journal'].replace(' ', '_')[:20]
                year = paper['pub_date'][:4] if paper['pub_date'] else 'unknown'
                
                filename = f"PMC{pmc_id}_{journal_short}_{year}_{safe_title}.pdf"
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                
                # Skip if file exists
                if os.path.exists(filepath):
                    print(f"  [SKIP] Already exists: {filename}")
                    continue
                
                # Download
                print(f"\n[{downloaded_count + 1}/{MAX_PAPERS}]")
                print(f"PMC ID: PMC{pmc_id}")
                print(f"Title: {title[:70]}...")
                print(f"Journal: {paper['journal']}")
                print(f"Year: {year}")
                print(f"DOI: {paper['doi']}")
                
                if download_pdf(pmc_id, filepath):
                    print(f"✓ Downloaded successfully")
                    
                    paper['filename'] = filename
                    downloaded_papers.append(paper)
                    downloaded_count += 1
                else:
                    print(f"✗ Failed to download (may not have PDF available)")
                
                # Respect NCBI rate limits (max 3 requests/second)
                time.sleep(DELAY)
            
            # Small delay between batches
            time.sleep(1)
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "pubmed_central_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"PubMed Central Papers Downloaded: {len(downloaded_papers)}\n")
        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Journals: Scientific Reports, PLOS ONE, MDPI\n")
        f.write("="*80 + "\n\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   PMC ID: PMC{paper['pmc_id']}\n")
            f.write(f"   Authors: {paper['authors']}\n")
            f.write(f"   Journal: {paper['journal']}\n")
            f.write(f"   Publication Date: {paper['pub_date']}\n")
            f.write(f"   DOI: {paper['doi']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\n{'='*80}")
    print(f"Download Complete!")
    print(f"Total papers downloaded this session: {len(downloaded_papers)}")
    print(f"Total papers in folder: {downloaded_count}")
    print(f"Papers saved to: {DOWNLOAD_DIR}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")
    
    # Print journal breakdown
    if downloaded_papers:
        journal_counts = {}
        for paper in downloaded_papers:
            journal = paper['journal']
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
        
        print("\nPapers by Journal (this session):")
        for journal, count in sorted(journal_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {journal}: {count}")

if __name__ == "__main__":
    main()
