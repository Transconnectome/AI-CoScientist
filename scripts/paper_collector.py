#!/usr/bin/env python3
"""
Automated Paper Collection Script for Nature Journal
Collects recent papers via SNU library proxy using Playwright automation.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import yaml
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/reference_papers/collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperCollector:
    """Automated paper collector using Playwright."""
    
    def __init__(self, config_path: str = "config/paper_collection_config.yaml"):
        """Initialize the collector with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.nature_config = self.config['journals']['nature']
        self.download_config = self.config['download']
        
        # Create output directory
        self.output_dir = Path(self.download_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata tracking
        self.metadata_file = Path(self.download_config['metadata_file'])
        self.metadata = self._load_metadata()
        
        # Session persistence
        self.auth_state_file = Path('data/reference_papers/auth_state.json')
        
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.context = None
        
    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'papers': [],
            'downloaded_urls': set(),
            'last_updated': None
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        # Convert set to list for JSON serialization
        save_data = self.metadata.copy()
        save_data['downloaded_urls'] = list(save_data['downloaded_urls'])
        save_data['last_updated'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Metadata saved: {len(self.metadata['papers'])} papers tracked")
    
    async def initialize_browser(self, headless: bool = False):
        """Initialize Playwright browser with session persistence."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=headless
        )
        
        # Load saved authentication state if available
        if self.auth_state_file.exists():
            logger.info(f"Loading saved authentication from {self.auth_state_file}")
            self.context = await self.browser.new_context(
                storage_state=str(self.auth_state_file)
            )
        else:
            logger.info("No saved authentication found, starting fresh session")
            self.context = await self.browser.new_context()
        
        self.page = await self.context.new_page()
        logger.info("Browser initialized")
    
    async def save_auth_state(self):
        """Save current authentication state."""
        if self.context:
            await self.context.storage_state(path=str(self.auth_state_file))
            logger.info(f"Authentication state saved to {self.auth_state_file}")
    
    async def handle_login(self):
        """Handle manual login if needed."""
        current_url = self.page.url
        
        # Check if we're on a login page
        if 'login' in current_url.lower() or 'sso' in current_url.lower():
            logger.info("\n" + "="*60)
            logger.info("LOGIN REQUIRED")
            logger.info("="*60)
            logger.info("Please complete the login process in the browser window.")
            logger.info("After logging in successfully, the script will continue automatically.")
            logger.info("Waiting for login...")
            logger.info("="*60 + "\n")
            
            # Wait for navigation away from login page
            try:
                await self.page.wait_for_url(
                    lambda url: 'login' not in url.lower() and 'sso' not in url.lower(),
                    timeout=300000  # 5 minutes for user to login
                )
                logger.info("✓ Login successful!")
                
                # Save authentication state for future use
                await self.save_auth_state()
                
                # Wait a bit for session to stabilize
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Login timeout or error: {e}")
                raise
        else:
            logger.info("Already authenticated")
    
    async def close_browser(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()
            logger.info("Browser closed")
    
    async def search_papers(self, limit: int = 100) -> List[Dict]:
        """
        Search for papers on Nature advanced search.
        
        Args:
            limit: Maximum number of papers to collect
            
        Returns:
            List of paper metadata dictionaries
        """
        search_url = f"{self.nature_config['url']}{self.nature_config['search_url']}"
        logger.info(f"Navigating to: {search_url}")
        
        await self.page.goto(search_url, wait_until='domcontentloaded', timeout=60000)
        await asyncio.sleep(3)  # Extra wait for dynamic content
        
        # Handle login if redirected
        await self.handle_login()
        
        # Handle cookie consent if present
        try:
            cookie_button = await self.page.wait_for_selector(
                'button:has-text("Accept")', 
                timeout=5000
            )
            if cookie_button:
                await cookie_button.click()
                logger.info("Accepted cookies")
                await asyncio.sleep(1)
        except Exception as e:
            logger.info(f"No cookie banner found or already accepted: {e}")
        
        # Wait for the advanced search form to be fully loaded
        logger.info("Waiting for search form to load...")
        await self.page.wait_for_selector('#start_year', state='visible', timeout=30000)
        await self.page.wait_for_selector('#end_year', state='visible', timeout=30000)
        
        # Set date range filters
        start_year = self.nature_config['date_range']['start']
        end_year = self.nature_config['date_range']['end']
        
        logger.info(f"Setting date range: {start_year} - {end_year}")
        
        # Select start year
        await self.page.select_option('#start_year', str(start_year))
        await asyncio.sleep(0.5)
        
        # Select end year
        await self.page.select_option('#end_year', str(end_year))
        await asyncio.sleep(0.5)
        
        # Optional: Add keywords for research articles
        # await self.page.fill('#advanced-search-keywords', 'research')
        
        # Scroll down to make submit button visible
        await self.page.evaluate('window.scrollBy(0, 500)')
        await asyncio.sleep(0.5)
        
        # Submit search - look for the Search button
        await self.page.click('button.c-search__button:has-text("Search")')
        
        # Wait for navigation with increased timeout
        try:
            await self.page.wait_for_load_state('domcontentloaded', timeout=60000)
        except Exception as e:
            logger.warning(f"Page load timeout, continuing anyway: {e}")
        
        await asyncio.sleep(3)  # Extra wait for results to render
        
        logger.info("Search submitted, parsing results...")
        
        # Parse search results
        papers = await self._parse_search_results(limit)
        
        return papers
    
    async def _parse_search_results(self, limit: int) -> List[Dict]:
        """Parse search results page and extract paper information."""
        papers = []
        page_num = 1
        
        while len(papers) < limit:
            logger.info(f"Parsing page {page_num}...")
            
            # Wait for results to load
            await self.page.wait_for_selector('article', timeout=10000)
            
            # Extract article information
            articles = await self.page.query_selector_all('article')
            
            for article in articles:
                if len(papers) >= limit:
                    break
                
                try:
                    # Extract title and link
                    title_elem = await article.query_selector('h3 a, h2 a')
                    if not title_elem:
                        continue
                    
                    title = await title_elem.inner_text()
                    href = await title_elem.get_attribute('href')
                    
                    # Build full URL
                    if href.startswith('/'):
                        url = f"{self.nature_config['url']}{href}"
                    else:
                        url = href
                    
                    # Skip if already downloaded
                    if url in self.metadata.get('downloaded_urls', set()):
                        logger.info(f"Skipping already downloaded: {title[:50]}...")
                        continue
                    
                    # Extract authors (if available)
                    authors = []
                    author_elems = await article.query_selector_all('[data-test="author-name"]')
                    for author_elem in author_elems:
                        author_name = await author_elem.inner_text()
                        authors.append(author_name)
                    
                    # Extract year (if available)
                    year = None
                    time_elem = await article.query_selector('time')
                    if time_elem:
                        date_str = await time_elem.get_attribute('datetime')
                        if date_str:
                            year = date_str.split('-')[0]
                    
                    paper_info = {
                        'title': title.strip(),
                        'url': url,
                        'authors': authors,
                        'year': year,
                        'journal': 'Nature',
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    papers.append(paper_info)
                    logger.info(f"Found paper {len(papers)}/{limit}: {title[:60]}...")
                    
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            # Check if there's a next page
            if len(papers) < limit:
                try:
                    next_button = await self.page.query_selector('a[rel="next"], button:has-text("Next")')
                    if next_button:
                        await next_button.click()
                        await self.page.wait_for_load_state('networkidle')
                        page_num += 1
                        await asyncio.sleep(2)  # Rate limiting
                    else:
                        logger.info("No more pages available")
                        break
                except Exception as e:
                    logger.warning(f"Error navigating to next page: {e}")
                    break
        
        logger.info(f"Found {len(papers)} papers total")
        return papers
    
    async def download_paper_pdf(self, paper: Dict, dry_run: bool = False) -> bool:
        """
        Download PDF for a single paper.
        
        Args:
            paper: Paper metadata dictionary
            dry_run: If True, don't actually download
            
        Returns:
            True if successful, False otherwise
        """
        url = paper['url']
        title = paper['title']
        
        logger.info(f"Processing: {title[:60]}...")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would download from: {url}")
            return True
        
        try:
            # Navigate to article page
            await self.page.goto(url, wait_until='networkidle')
            await asyncio.sleep(2)
            
            # Look for PDF download link
            pdf_link = None
            
            # Try different selectors for PDF download
            selectors = [
                'a[data-track-action="download pdf"]',
                'a:has-text("Download PDF")',
                'a[href*=".pdf"]',
                'a.c-pdf-download__link'
            ]
            
            for selector in selectors:
                try:
                    pdf_link = await self.page.query_selector(selector)
                    if pdf_link:
                        logger.info(f"Found PDF link with selector: {selector}")
                        break
                except:
                    continue
            
            if not pdf_link:
                logger.warning(f"No PDF download link found for: {title[:60]}")
                return False
            
            # Get PDF URL
            pdf_href = await pdf_link.get_attribute('href')
            if pdf_href.startswith('/'):
                pdf_url = f"{self.nature_config['url']}{pdf_href}"
            else:
                pdf_url = pdf_href
            
            # Generate filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:50]
            year = paper.get('year', 'unknown')
            first_author = paper['authors'][0].split()[-1] if paper['authors'] else 'unknown'
            filename = f"nature_{year}_{first_author}_{safe_title}.pdf"
            filepath = self.output_dir / filename
            
            # Download PDF
            logger.info(f"Downloading PDF to: {filename}")
            
            async with self.page.expect_download() as download_info:
                await pdf_link.click()
            
            download = await download_info.value
            await download.save_as(filepath)
            
            # Update metadata
            paper['pdf_path'] = str(filepath)
            paper['downloaded_at'] = datetime.now().isoformat()
            self.metadata['papers'].append(paper)
            self.metadata['downloaded_urls'].add(url)
            
            logger.info(f"✓ Successfully downloaded: {filename}")
            
            # Rate limiting
            await asyncio.sleep(self.download_config['rate_limit_seconds'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {title[:60]}: {e}")
            return False
    
    async def collect_papers(self, limit: int = 100, dry_run: bool = False):
        """
        Main collection workflow.
        
        Args:
            limit: Number of papers to collect
            dry_run: If True, don't download PDFs
        """
        try:
            await self.initialize_browser()
            
            # Search for papers
            papers = await self.search_papers(limit)
            
            if not papers:
                logger.warning("No papers found!")
                return
            
            logger.info(f"Starting download of {len(papers)} papers...")
            
            # Download each paper
            success_count = 0
            for i, paper in enumerate(papers, 1):
                logger.info(f"\n--- Paper {i}/{len(papers)} ---")
                
                if await self.download_paper_pdf(paper, dry_run):
                    success_count += 1
                    
                    # Save metadata periodically
                    if success_count % 10 == 0:
                        self._save_metadata()
            
            # Final save
            self._save_metadata()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Collection complete!")
            logger.info(f"Successfully downloaded: {success_count}/{len(papers)} papers")
            logger.info(f"PDFs saved to: {self.output_dir}")
            logger.info(f"Metadata saved to: {self.metadata_file}")
            logger.info(f"{'='*60}")
            
        finally:
            await self.close_browser()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Collect papers from Nature journal')
    parser.add_argument('--limit', type=int, default=100, help='Number of papers to collect')
    parser.add_argument('--dry-run', action='store_true', help='Test run without downloading')
    parser.add_argument('--config', type=str, default='config/paper_collection_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    collector = PaperCollector(config_path=args.config)
    await collector.collect_papers(limit=args.limit, dry_run=args.dry_run)


if __name__ == '__main__':
    asyncio.run(main())
