#!/usr/bin/env python3
"""
URL List Downloader
Reads urls.txt and downloads PDFs
Format: URL | Title | Venue
"""

import requests
import os
import time

DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/neuroscience_ml_papers"
URLS_FILE = os.path.join(DOWNLOAD_DIR, "urls.txt")

def download_pdf(url, filepath, title):
    """Download PDF with retries"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    for attempt in range(3):
        try:
            print(f"  Downloading: {url}")
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size = os.path.getsize(filepath)
            if size < 10000:
                print(f"  ⚠ File too small ({size} bytes), might be error page")
                os.remove(filepath)
                return False
            
            print(f"  ✓ Downloaded ({size/1024:.1f} KB)")
            return True
            
        except Exception as e:
            print(f"  ✗ Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return False

def main():
    """Main downloader from URL list"""
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    if not os.path.exists(URLS_FILE):
        print(f"❌ URLs file not found: {URLS_FILE}")
        print("\nPlease create urls.txt with format:")
        print("URL | Title | Venue")
        print("\nExample:")
        print("https://papers.nips.cc/paper_files/paper/2023/file/xxx.pdf | Brain Imaging | NeurIPS 2023")
        return
    
    print("="*80)
    print("URL List Downloader")
    print("="*80)
    print()
    
    # Read URLs
    papers = []
    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                papers.append({
                    'url': parts[0],
                    'title': parts[1],
                    'venue': parts[2]
                })
    
    print(f"Found {len(papers)} papers in urls.txt\n")
    
    downloaded = []
    downloaded_count = 0
    
    for i, paper in enumerate(papers, 1):
        title = paper['title']
        url = paper['url']
        venue = paper['venue']
        
        # Create filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
        safe_title = safe_title[:70].strip()
        safe_venue = venue.replace(' ', '_')
        filename = f"{safe_venue}_{safe_title}.pdf"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        print(f"[{i}/{len(papers)}] {title}")
        print(f"  Venue: {venue}")
        
        if os.path.exists(filepath):
            print(f"  ⏭ Already exists, skipping\n")
            downloaded.append({**paper, 'filename': filename})
            downloaded_count += 1
            continue
        
        if download_pdf(url, filepath, title):
            downloaded.append({**paper, 'filename': filename})
            downloaded_count += 1
        
        print()
        time.sleep(2)
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "DOWNLOADED_PAPERS.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Downloaded Papers: {downloaded_count}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
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
            f.write(f"   URL: {paper['url']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print("="*80)
    print(f"COMPLETE: {downloaded_count} papers downloaded")
    print(f"Location: {DOWNLOAD_DIR}")
    print(f"Summary: {summary_file}")
    print("="*80)
    
    if venue_counts:
        print("\nVenue Breakdown:")
        for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {venue}: {count}")

if __name__ == "__main__":
    main()
