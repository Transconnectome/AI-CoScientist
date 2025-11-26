#!/usr/bin/env python3
"""
Download Papers from URL List
Downloads PDFs from a list of Nature paper URLs using your authenticated browser session.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import yaml
from playwright.async_api import async_playwright, Browser, BrowserContext


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/reference_papers/download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleDownloader:
    """Download papers from a list of URLs using existing browser profile."""
    
    def __init__(self, url_file: str = "data/reference_papers/paper_urls.json"):
        self.url_file = Path(url_file)
        self.output_dir = Path("data/reference_papers/pdfs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = Path("data/reference_papers/metadata.json")
        self.metadata = self._load_metadata()
        
        self.browser: Browser = None
        self.context: BrowserContext = None
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'papers': [], 'downloaded_urls': []}
    
    def _save_metadata(self):
        """Save metadata."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Metadata saved: {len(self.metadata['papers'])} papers")
    
    def load_urls(self) -> List[Dict]:
        """Load URLs from JSON file."""
        if not self.url_file.exists():
            logger.error(f"URL file not found: {self.url_file}")
            logger.info("Run: poetry run python scripts/create_url_template.py")
            return []
        
        with open(self.url_file, 'r') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        logger.info(f"Loaded {len(papers)} paper URLs")
        return papers
    
    async def initialize_browser(self, use_saved_auth: bool = True):
        """Initialize browser with optional saved authentication."""
        playwright = await async_playwright().start()
        
        auth_file = Path('data/reference_papers/auth_state.json')
        
        if use_saved_auth and auth_file.exists():
            logger.info(f"Using saved authentication from {auth_file}")
            self.browser = await playwright.chromium.launch(headless=False)
            self.context = await self.browser.new_context(
                storage_state=str(auth_file)
            )
        else:
            logger.info("Starting fresh browser session")
            self.browser = await playwright.chromium.launch(headless=False)
            self.context = await self.browser.new_context()
        
        logger.info("Browser initialized")
    
    async def download_paper(self, paper_info: Dict) -> bool:
        """Download a single paper."""
        url = paper_info.get('url', '')
        
        if not url or url in self.metadata['downloaded_urls']:
            logger.info(f"Skipping: {url}")
            return False
        
        logger.info(f"Processing: {url}")
        
        try:
            page = await self.context.new_page()
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await asyncio.sleep(2)
            
            # Check if login is needed
            if 'login' in page.url.lower():
                logger.warning("Login required! Please log in manually in the browser.")
                logger.info("Waiting 60 seconds for manual login...")
                await asyncio.sleep(60)
                
                # Save auth state after manual login
                await self.context.storage_state(
                    path='data/reference_papers/auth_state.json'
                )
                logger.info("Auth state saved!")
            
            # Extract title if not provided
            title = paper_info.get('title', '')
            if not title:
                try:
                    title_elem = await page.query_selector('h1.c-article-title')
                    if title_elem:
                        title = await title_elem.inner_text()
                except:
                    title = "unknown"
            
            # Look for PDF download link
            pdf_link = None
            selectors = [
                'a[data-track-action="download pdf"]',
                'a:has-text("Download PDF")',
                'a.c-pdf-download__link',
                'a[href*=".pdf"]'
            ]
            
            for selector in selectors:
                try:
                    pdf_link = await page.query_selector(selector)
                    if pdf_link:
                        break
                except:
                    continue
            
            if not pdf_link:
                logger.warning(f"No PDF link found for: {title[:60]}")
                await page.close()
                return False
            
            # Generate filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:50]
            filename = f"nature_{safe_title}.pdf"
            filepath = self.output_dir / filename
            
            # Download PDF
            logger.info(f"Downloading: {filename}")
            
            download_success = False
            
            try:
                # Try standard download event first
                async with page.expect_download(timeout=5000) as download_info:
                    await pdf_link.click()
                
                download = await download_info.value
                await download.save_as(filepath)
                download_success = True
                
            except Exception:
                # If download event didn't fire, check if it opened in a new tab or redirected
                # Wait a bit for navigation
                await asyncio.sleep(5)
                
                # Check if current page is PDF
                if str(page.url).endswith('.pdf'):
                    logger.info("PDF opened in current tab, downloading via requests...")
                    cookies = await self.context.cookies()
                    cookie_dict = {c['name']: c['value'] for c in cookies}
                    
                    import requests
                    response = requests.get(page.url, cookies=cookie_dict, stream=True)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        download_success = True
                
                # Check for new pages (tabs)
                elif len(self.context.pages) > 1:
                    new_page = self.context.pages[-1]
                    await new_page.wait_for_load_state()
                    if str(new_page.url).endswith('.pdf'):
                        logger.info("PDF opened in new tab, downloading via requests...")
                        cookies = await self.context.cookies()
                        cookie_dict = {c['name']: c['value'] for c in cookies}
                        
                        import requests
                        response = requests.get(new_page.url, cookies=cookie_dict, stream=True)
                        if response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            download_success = True
                        await new_page.close()

            if download_success:
                # Update metadata
                paper_data = {
                    'url': url,
                    'title': title,
                    'pdf_path': str(filepath),
                    'downloaded_at': datetime.now().isoformat()
                }
                
                self.metadata['papers'].append(paper_data)
                self.metadata['downloaded_urls'].append(url)
                
                logger.info(f"✓ Downloaded: {filename}")
                await page.close()
                await asyncio.sleep(4)  # Rate limiting
                return True
            else:
                logger.warning(f"Failed to download PDF for: {title[:60]}")
                await page.close()
                return False
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            try:
                await page.close()
            except:
                pass
            return False
    
    async def download_all(self):
        """Download all papers from URL list."""
        papers = self.load_urls()
        
        if not papers:
            return
        
        await self.initialize_browser()
        
        success_count = 0
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n--- Paper {i}/{len(papers)} ---")
            
            if await self.download_paper(paper):
                success_count += 1
                
                # Save metadata periodically
                if success_count % 5 == 0:
                    self._save_metadata()
        
        # Final save
        self._save_metadata()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Download complete!")
        logger.info(f"Successfully downloaded: {success_count}/{len(papers)} papers")
        logger.info(f"PDFs saved to: {self.output_dir}")
        logger.info(f"{'='*60}")
        
        if self.browser:
            await self.browser.close()


async def main():
    parser = argparse.ArgumentParser(description='Download papers from URL list')
    parser.add_argument('--url-file', type=str, 
                       default='data/reference_papers/paper_urls.json',
                       help='Path to URL list JSON file')
    
    args = parser.parse_args()
    
    downloader = SimpleDownloader(url_file=args.url_file)
    await downloader.download_all()


if __name__ == '__main__':
    asyncio.run(main())
