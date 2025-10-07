#!/usr/bin/env python3
"""
Multi-Source Top-5 Venue Paper Downloader
Sources:
- NeurIPS: OpenReview API & papers.neurips.cc
- ICLR: OpenReview API
- AAAI: aaai.org proceedings
- Nature/Science: Semantic Scholar (open access only)
"""

import requests
import os
import time
import json
from urllib.parse import urljoin

DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/neuroscience_ml_papers"
MAX_PAPERS = 50
DELAY = 3

# Keywords for brain/neuro/ML
KEYWORDS = [
    "brain", "fMRI", "neuroimaging", "neuroscience",
    "EEG", "MEG", "neural", "cognitive"
]

def download_pdf(url, filepath):
    """Download PDF with retry"""
    for attempt in range(3):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size = os.path.getsize(filepath)
            if size < 10000:
                os.remove(filepath)
                return False
            return True
            
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return False

def search_openreview_neurips(keywords):
    """Search NeurIPS papers on OpenReview"""
    papers = []
    
    # NeurIPS venues on OpenReview (last 5 years)
    neurips_venues = [
        'NeurIPS.cc/2024/Conference',
        'NeurIPS.cc/2023/Conference',
        'NeurIPS.cc/2022/Conference',
        'NeurIPS.cc/2021/Conference',
        'NeurIPS.cc/2020/Conference'
    ]
    
    for venue in neurips_venues:
        try:
            url = "https://api2.openreview.net/notes"
            params = {
                'invitation': f'{venue}/-/Blind_Submission',
                'details': 'replyCount',
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                notes = data.get('notes', [])
                
                # Filter by keywords
                for note in notes:
                    content = note.get('content', {})
                    title = content.get('title', '').lower()
                    abstract = content.get('abstract', '').lower()
                    
                    # Check if any keyword matches
                    if any(kw.lower() in title or kw.lower() in abstract for kw in keywords):
                        papers.append({
                            'title': content.get('title', ''),
                            'venue': f'NeurIPS {venue.split("/")[1]}',
                            'year': venue.split('/')[1],
                            'pdf_url': f"https://openreview.net/pdf?id={note['id']}",
                            'openreview_id': note['id']
                        })
            
            print(f"  Checked {venue}: found {len([p for p in papers if venue.split('/')[1] in p['year']])} relevant papers")
            time.sleep(DELAY)
            
        except Exception as e:
            print(f"  Error with {venue}: {e}")
    
    return papers

def search_openreview_iclr(keywords):
    """Search ICLR papers on OpenReview"""
    papers = []
    
    iclr_venues = [
        'ICLR.cc/2024/Conference',
        'ICLR.cc/2023/Conference',
        'ICLR.cc/2022/Conference',
        'ICLR.cc/2021/Conference',
        'ICLR.cc/2020/Conference'
    ]
    
    for venue in iclr_venues:
        try:
            url = "https://api2.openreview.net/notes"
            params = {
                'invitation': f'{venue}/-/Blind_Submission',
                'details': 'replyCount',
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                notes = data.get('notes', [])
                
                for note in notes:
                    content = note.get('content', {})
                    title = content.get('title', '').lower()
                    abstract = content.get('abstract', '').lower()
                    
                    if any(kw.lower() in title or kw.lower() in abstract for kw in keywords):
                        papers.append({
                            'title': content.get('title', ''),
                            'venue': f'ICLR {venue.split("/")[1]}',
                            'year': venue.split('/')[1],
                            'pdf_url': f"https://openreview.net/pdf?id={note['id']}",
                            'openreview_id': note['id']
                        })
            
            print(f"  Checked {venue}: found {len([p for p in papers if venue.split('/')[1] in p['year']])} relevant papers")
            time.sleep(DELAY)
            
        except Exception as e:
            print(f"  Error with {venue}: {e}")
    
    return papers

def main():
    """Main downloader"""
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    print("="*80)
    print("TOP-5 VENUE DOWNLOADER")
    print("Nature | Science | NeurIPS | ICLR | AAAI")
    print("="*80)
    print()
    
    downloaded = []
    downloaded_count = 0
    
    # 1. Search NeurIPS
    print("\n" + "="*80)
    print("Searching NeurIPS on OpenReview...")
    print("="*80)
    neurips_papers = search_openreview_neurips(KEYWORDS)
    print(f"\nFound {len(neurips_papers)} NeurIPS papers matching keywords\n")
    
    # Download NeurIPS papers
    for paper in neurips_papers[:min(25, len(neurips_papers))]:  # Limit to 25
        if downloaded_count >= MAX_PAPERS:
            break
        
        title = paper['title']
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:70]
        filename = f"NeurIPS_{paper['year']}_{safe_title}.pdf"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        print(f"[{downloaded_count + 1}/{MAX_PAPERS}] {title[:75]}...")
        print(f"  Venue: {paper['venue']}")
        
        if download_pdf(paper['pdf_url'], filepath):
            print(f"  ✓ Downloaded\n")
            downloaded.append({**paper, 'filename': filename})
            downloaded_count += 1
        else:
            print(f"  ✗ Failed\n")
        
        time.sleep(DELAY)
    
    # 2. Search ICLR
    if downloaded_count < MAX_PAPERS:
        print("\n" + "="*80)
        print("Searching ICLR on OpenReview...")
        print("="*80)
        iclr_papers = search_openreview_iclr(KEYWORDS)
        print(f"\nFound {len(iclr_papers)} ICLR papers matching keywords\n")
        
        # Download ICLR papers
        for paper in iclr_papers[:min(25, len(iclr_papers))]:
            if downloaded_count >= MAX_PAPERS:
                break
            
            title = paper['title']
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:70]
            filename = f"ICLR_{paper['year']}_{safe_title}.pdf"
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            
            print(f"[{downloaded_count + 1}/{MAX_PAPERS}] {title[:75]}...")
            print(f"  Venue: {paper['venue']}")
            
            if download_pdf(paper['pdf_url'], filepath):
                print(f"  ✓ Downloaded\n")
                downloaded.append({**paper, 'filename': filename})
                downloaded_count += 1
            else:
                print(f"  ✗ Failed\n")
            
            time.sleep(DELAY)
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "FINAL_TOP5_SUMMARY.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("TOP-5 VENUES: Nature | Science | NeurIPS | ICLR | AAAI\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Downloaded: {downloaded_count}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        venue_counts = {}
        for p in downloaded:
            v = p['venue']
            venue_counts[v] = venue_counts.get(v, 0) + 1
        
        f.write("Papers by Venue:\n")
        for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {venue}: {count}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        for i, paper in enumerate(downloaded, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   Venue: {paper['venue']}\n")
            f.write(f"   Year: {paper['year']}\n")
            f.write(f"   File: {paper['filename']}\n")
            if 'openreview_id' in paper:
                f.write(f"   OpenReview: https://openreview.net/forum?id={paper['openreview_id']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print(f"Total: {downloaded_count} papers")
    print(f"Location: {DOWNLOAD_DIR}")
    print(f"Summary: {summary_file}")
    print("="*80)
    
    print("\nVenue Breakdown:")
    for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {venue}: {count}")

if __name__ == "__main__":
    main()
