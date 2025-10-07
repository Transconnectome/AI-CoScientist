#!/usr/bin/env python3
"""
Web Scraping Paper Downloader
Directly scrapes Scientific Reports, PLOS ONE, and MDPI websites
Using Playwright for browser automation
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import os
import time
import json
from datetime import datetime
import re

# Configuration
DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/open_access_papers"
MAX_PAPERS = 50
DELAY = 2  # seconds between downloads

# Search keywords
KEYWORDS = [
    "brain imaging machine learning",
    "fMRI deep learning",
    "neuroscience artificial intelligence",
    "neuroimaging",
    "EEG machine learning",
    "brain decoding"
]

def scrape_scientific_reports(page, keyword, max_results=20):
    """Scrape Scientific Reports"""
    papers = []
    
    try:
        # Navigate to search
        search_url = f"https://www.nature.com/search?q={keyword.replace(' ', '%20')}&journal=srep&order=relevance"
        print(f"  Searching Scientific Reports: {search_url}")
        page.goto(search_url, wait_until='networkidle', timeout=30000)
        
        time.sleep(3)
        
        # Get article links
        article_links = page.query_selector_all('article a[data-track-action="view article"]')
        
        for i, link in enumerate(article_links[:max_results]):
            if len(papers) >= max_results:
                break
                
            try:
                href = link.get_attribute('href')
                if not href:
                    continue
                
                full_url = f"https://www.nature.com{href}" if href.startswith('/') else href
                
                # Visit article page
                page.goto(full_url, wait_until='networkidle', timeout=30000)
                time.sleep(2)
                
                # Get metadata
                title_elem = page.query_selector('h1.c-article-title')
                title = title_elem.inner_text() if title_elem else 'Untitled'
                
                # Get PDF link
                pdf_link = page.query_selector('a[data-track-action="download pdf"]')
                if not pdf_link:
                    pdf_link = page.query_selector('a[href*=".pdf"]')
                
                if pdf_link:
                    pdf_url = pdf_link.get_attribute('href')
                    if pdf_url and pdf_url.startswith('/'):
                        pdf_url = f"https://www.nature.com{pdf_url}"
                    
                    # Get year
                    year_elem = page.query_selector('time')
                    year = year_elem.get_attribute('datetime')[:4] if year_elem else 'Unknown'
                    
                    papers.append({
                        'title': title,
                        'url': full_url,
                        'pdf_url': pdf_url,
                        'journal': 'Scientific Reports',
                        'year': year
                    })
                    
                    print(f"    Found: {title[:60]}...")
                    
            except Exception as e:
                print(f"    Error processing article: {e}")
                continue
        
    except Exception as e:
        print(f"  Error scraping Scientific Reports: {e}")
    
    return papers

def scrape_plos_one(page, keyword, max_results=20):
    """Scrape PLOS ONE"""
    papers = []
    
    try:
        # Navigate to search
        search_url = f"https://journals.plos.org/plosone/search?q={keyword.replace(' ', '+')}&sortOrder=RELEVANCE"
        print(f"  Searching PLOS ONE: {search_url}")
        page.goto(search_url, wait_until='networkidle', timeout=30000)
        
        time.sleep(3)
        
        # Get article links
        article_links = page.query_selector_all('li.search-results-item h2 a')
        
        for i, link in enumerate(article_links[:max_results]):
            if len(papers) >= max_results:
                break
                
            try:
                href = link.get_attribute('href')
                if not href:
                    continue
                
                full_url = href if href.startswith('http') else f"https://journals.plos.org{href}"
                
                # Get title
                title = link.inner_text()
                
                # Construct PDF URL (PLOS ONE has predictable PDF URLs)
                # Example: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123456
                # PDF: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0123456&type=printable
                
                if 'article?id=' in full_url:
                    pdf_url = full_url.replace('article?id=', 'article/file?id=') + '&type=printable'
                    
                    # Extract year from article page or DOI
                    year = 'Unknown'
                    doi_match = re.search(r'journal\.pone\.(\d{4})\d+', full_url)
                    if doi_match:
                        # First 4 digits after pone. are often the year, but not always reliable
                        pass
                    
                    # Visit page to get year
                    try:
                        page.goto(full_url, wait_until='networkidle', timeout=20000)
                        time.sleep(1)
                        year_elem = page.query_selector('meta[name="citation_publication_date"]')
                        if year_elem:
                            year = year_elem.get_attribute('content')[:4]
                    except:
                        pass
                    
                    papers.append({
                        'title': title,
                        'url': full_url,
                        'pdf_url': pdf_url,
                        'journal': 'PLOS ONE',
                        'year': year
                    })
                    
                    print(f"    Found: {title[:60]}...")
                    
            except Exception as e:
                print(f"    Error processing article: {e}")
                continue
        
    except Exception as e:
        print(f"  Error scraping PLOS ONE: {e}")
    
    return papers

def scrape_mdpi(page, keyword, max_results=20):
    """Scrape MDPI journals"""
    papers = []
    
    try:
        # Navigate to search
        search_url = f"https://www.mdpi.com/search?q={keyword.replace(' ', '+')}&journal=&volume=&authors=&article_type=&search=1&sort=pubdate"
        print(f"  Searching MDPI: {search_url}")
        page.goto(search_url, wait_until='networkidle', timeout=30000)
        
        time.sleep(3)
        
        # Get article links
        article_divs = page.query_selector_all('div.article-content')
        
        for i, div in enumerate(article_divs[:max_results]):
            if len(papers) >= max_results:
                break
                
            try:
                # Get title and link
                title_link = div.query_selector('a.title-link')
                if not title_link:
                    continue
                
                title = title_link.inner_text()
                href = title_link.get_attribute('href')
                
                if not href:
                    continue
                
                full_url = href if href.startswith('http') else f"https://www.mdpi.com{href}"
                
                # MDPI PDF URL pattern: /article-number/pdf
                pdf_url = full_url.rstrip('/') + '/pdf'
                
                # Get journal name and year
                journal_elem = div.query_selector('div.journal-name')
                journal = journal_elem.inner_text() if journal_elem else 'MDPI'
                
                year_elem = div.query_selector('div.pubdate')
                year = 'Unknown'
                if year_elem:
                    year_text = year_elem.inner_text()
                    year_match = re.search(r'(\d{4})', year_text)
                    if year_match:
                        year = year_match.group(1)
                
                papers.append({
                    'title': title,
                    'url': full_url,
                    'pdf_url': pdf_url,
                    'journal': journal,
                    'year': year
                })
                
                print(f"    Found: {title[:60]}...")
                
            except Exception as e:
                print(f"    Error processing article: {e}")
                continue
        
    except Exception as e:
        print(f"  Error scraping MDPI: {e}")
    
    return papers


def download_pdf(page, pdf_url, filepath):
    """Download PDF using Playwright"""
    try:
        # Navigate to PDF URL
        response = page.goto(pdf_url, wait_until='networkidle', timeout=60000)
        
        if response and response.status == 200:
            # Check if it's a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type.lower():
                # Download using page.pdf() or save content
                # For direct PDF download, we can use requests after getting the URL
                import requests
                pdf_response = requests.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(pdf_response.content)
                
                # Verify PDF
                with open(filepath, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        os.remove(filepath)
                        return False
                
                return True
        
        return False
        
    except Exception as e:
        print(f"  Download error: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    """Main scraping function"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    # Load existing files
    existing_files = set(os.listdir(DOWNLOAD_DIR))
    
    all_papers = []
    downloaded_count = 0
    
    print(f"Starting web scraping...")
    print(f"Target: {MAX_PAPERS} papers")
    print(f"Target journals: Scientific Reports, PLOS ONE, MDPI\n")
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        )
        page = context.new_page()
        
        # Scrape each source
        for keyword in KEYWORDS:
            if downloaded_count >= MAX_PAPERS:
                break
            
            print(f"\n{'='*80}")
            print(f"Keyword: {keyword}")
            print(f"{'='*80}")
            
            # Scrape Scientific Reports
            print("\nScraping Scientific Reports...")
            sr_papers = scrape_scientific_reports(page, keyword, max_results=10)
            all_papers.extend(sr_papers)
            
            time.sleep(DELAY)
            
            # Scrape PLOS ONE
            print("\nScraping PLOS ONE...")
            plos_papers = scrape_plos_one(page, keyword, max_results=10)
            all_papers.extend(plos_papers)
            
            time.sleep(DELAY)
            
            # Scrape MDPI
            print("\nScraping MDPI...")
            mdpi_papers = scrape_mdpi(page, keyword, max_results=10)
            all_papers.extend(mdpi_papers)
            
            print(f"\nTotal papers found so far: {len(all_papers)}")
        
        # Download papers
        print(f"\n{'='*80}")
        print("Downloading PDFs...")
        print(f"{'='*80}\n")
        
        downloaded_papers = []
        seen_titles = set()
        
        for paper in all_papers:
            if downloaded_count >= MAX_PAPERS:
                break
            
            # Skip duplicates
            if paper['title'] in seen_titles:
                continue
            
            seen_titles.add(paper['title'])
            
            # Create filename
            safe_title = "".join(c for c in paper['title'] if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title[:60]
            journal_short = paper['journal'].replace(' ', '_')[:20]
            year = paper['year']
            
            filename = f"WEB_{journal_short}_{year}_{safe_title}.pdf"
            
            # Skip if exists
            if filename in existing_files:
                print(f"Skipping (exists): {paper['title'][:60]}...")
                continue
            
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            
            # Download
            print(f"[{downloaded_count + 1}/{MAX_PAPERS}]")
            print(f"Title: {paper['title'][:70]}...")
            print(f"Journal: {paper['journal']}")
            print(f"Year: {year}")
            print(f"PDF URL: {paper['pdf_url'][:60]}...")
            
            if download_pdf(page, paper['pdf_url'], filepath):
                print(f"✓ Downloaded successfully\n")
                
                downloaded_papers.append({
                    'title': paper['title'],
                    'journal': paper['journal'],
                    'year': year,
                    'url': paper['url'],
                    'filename': filename
                })
                
                downloaded_count += 1
            else:
                print(f"✗ Failed to download\n")
            
            time.sleep(DELAY)
        
        browser.close()
    
    # Save summary
    summary_file = os.path.join(DOWNLOAD_DIR, "web_scraped_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Web Scraped Papers Downloaded: {downloaded_count}\n")
        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: Direct web scraping\n")
        f.write("="*80 + "\n\n")
        
        for i, paper in enumerate(downloaded_papers, 1):
            f.write(f"{i}. {paper['title']}\n")
            f.write(f"   Journal: {paper['journal']}\n")
            f.write(f"   Year: {paper['year']}\n")
            f.write(f"   URL: {paper['url']}\n")
            f.write(f"   File: {paper['filename']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    # Save JSON
    json_file = os.path.join(DOWNLOAD_DIR, "web_scraped_metadata.json")
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
