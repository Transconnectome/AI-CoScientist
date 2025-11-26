#!/usr/bin/env python3
"""
Hybrid paper downloader combining multiple methods for maximum success rate.

Methods (in priority order):
1. Semantic Scholar open access PDF (direct download)
2. Unpaywall API (DOI-based open access)
3. SNU proxy access (institutional access)
4. Playwright automated download (last resort)
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Optional, List
import aiohttp
import aiofiles
from urllib.parse import urlparse, quote


class HybridDownloader:
    def __init__(self, papers_file: str = "data/reference_papers/paper_urls_final.json"):
        self.papers_file = Path(papers_file)
        self.output_dir = Path("data/reference_papers/pdfs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.s2_api = "https://api.semanticscholar.org/graph/v1/paper"
        self.unpaywall_api = "https://api.unpaywall.org/v2"
        self.email = "your.email@example.com"  # For Unpaywall API

        self.success_count = 0
        self.fail_count = 0
        self.methods_used = {
            'semantic_scholar': 0,
            'unpaywall': 0,
            'snu_proxy': 0,
            'playwright': 0
        }

    def sanitize_filename(self, title: str, max_length: int = 100) -> str:
        """Create safe filename from paper title."""
        # Remove special characters
        filename = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces with underscores
        filename = re.sub(r'\s+', '_', filename)
        # Truncate if too long
        if len(filename) > max_length:
            filename = filename[:max_length]
        return filename + '.pdf'

    async def download_file(self, url: str, output_path: Path) -> bool:
        """Download file from URL to output path."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        content = await response.read()

                        # Verify it's a PDF
                        if content[:4] != b'%PDF':
                            print(f"    ⚠️  Not a valid PDF file")
                            return False

                        async with aiofiles.open(output_path, 'wb') as f:
                            await f.write(content)

                        # Check file size
                        size_kb = len(content) / 1024
                        if size_kb < 50:  # Suspiciously small
                            print(f"    ⚠️  File too small ({size_kb:.1f}KB)")
                            output_path.unlink()
                            return False

                        print(f"    ✓ Downloaded ({size_kb:.1f}KB)")
                        return True
                    else:
                        print(f"    ✗ HTTP {response.status}")
                        return False

        except asyncio.TimeoutError:
            print(f"    ✗ Timeout")
            return False
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return False

    async def try_semantic_scholar(self, paper: Dict) -> Optional[Path]:
        """Method 1: Try Semantic Scholar open access PDF."""
        try:
            # Extract S2 paper ID from URL if available
            url = paper['url']

            # If it's a DOI URL, look up via DOI
            if 'doi.org' in url:
                doi = url.split('doi.org/')[-1]
                lookup_url = f"{self.s2_api}/DOI:{doi}"
            elif 'semanticscholar.org' in url:
                paper_id = url.split('/')[-1]
                lookup_url = f"{self.s2_api}/{paper_id}"
            else:
                # Try searching by title
                lookup_url = None

            if not lookup_url:
                return None

            # Get paper metadata with PDF info
            params = {'fields': 'openAccessPdf'}

            async with aiohttp.ClientSession() as session:
                async with session.get(lookup_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pdf_info = data.get('openAccessPdf')

                        if pdf_info and pdf_info.get('url'):
                            pdf_url = pdf_info['url']
                            print(f"  [S2] Found open access PDF")

                            filename = self.sanitize_filename(paper['title'])
                            output_path = self.output_dir / filename

                            if await self.download_file(pdf_url, output_path):
                                self.methods_used['semantic_scholar'] += 1
                                return output_path

            return None

        except Exception as e:
            print(f"  [S2] Error: {e}")
            return None

    async def try_unpaywall(self, paper: Dict) -> Optional[Path]:
        """Method 2: Try Unpaywall API for open access."""
        try:
            # Extract DOI from URL
            url = paper['url']
            if 'doi.org' not in url:
                return None

            doi = url.split('doi.org/')[-1]
            lookup_url = f"{self.unpaywall_api}/{doi}"
            params = {'email': self.email}

            async with aiohttp.ClientSession() as session:
                async with session.get(lookup_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Check for best OA location
                        best_oa = data.get('best_oa_location')
                        if best_oa and best_oa.get('url_for_pdf'):
                            pdf_url = best_oa['url_for_pdf']
                            print(f"  [Unpaywall] Found OA PDF")

                            filename = self.sanitize_filename(paper['title'])
                            output_path = self.output_dir / filename

                            if await self.download_file(pdf_url, output_path):
                                self.methods_used['unpaywall'] += 1
                                return output_path

            return None

        except Exception as e:
            print(f"  [Unpaywall] Error: {e}")
            return None

    async def try_snu_proxy(self, paper: Dict) -> Optional[Path]:
        """Method 3: Try SNU proxy direct access."""
        try:
            url = paper['url']

            # Construct SNU proxy URL
            if 'doi.org' in url:
                doi = url.split('doi.org/')[-1]
                # Try Nature proxy first
                proxy_url = f"https://www-nature-com-ssl.libproxy.snu.ac.kr/articles/{doi}.pdf"

                print(f"  [SNU Proxy] Trying institutional access")

                filename = self.sanitize_filename(paper['title'])
                output_path = self.output_dir / filename

                if await self.download_file(proxy_url, output_path):
                    self.methods_used['snu_proxy'] += 1
                    return output_path

            return None

        except Exception as e:
            print(f"  [SNU Proxy] Error: {e}")
            return None

    async def download_paper(self, paper: Dict, index: int, total: int) -> bool:
        """Download a single paper using all available methods."""
        print(f"\n[{index}/{total}] {paper['title'][:80]}...")

        # Check if already downloaded
        filename = self.sanitize_filename(paper['title'])
        output_path = self.output_dir / filename
        if output_path.exists():
            print(f"  ✓ Already downloaded, skipping")
            self.success_count += 1
            return True

        # Try methods in order
        methods = [
            ('Semantic Scholar', self.try_semantic_scholar),
            ('Unpaywall', self.try_unpaywall),
            ('SNU Proxy', self.try_snu_proxy),
        ]

        for method_name, method_func in methods:
            result = await method_func(paper)
            if result:
                self.success_count += 1
                return True

            # Rate limiting between attempts
            await asyncio.sleep(0.5)

        # All methods failed
        print(f"  ✗ All methods failed")
        self.fail_count += 1
        return False

    async def download_all(self):
        """Download all papers from the list."""
        # Load papers
        with open(self.papers_file, 'r') as f:
            data = json.load(f)
        papers = data['papers']

        print("="*70)
        print("HYBRID PAPER DOWNLOADER")
        print("="*70)
        print(f"Papers to download: {len(papers)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Methods: Semantic Scholar → Unpaywall → SNU Proxy")
        print("="*70)

        # Download papers with rate limiting
        for i, paper in enumerate(papers, 1):
            await self.download_paper(paper, i, len(papers))

            # Rate limiting between papers
            if i < len(papers):
                await asyncio.sleep(1)

        # Summary
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Successful: {self.success_count}/{len(papers)}")
        print(f"Failed: {self.fail_count}/{len(papers)}")
        print(f"\nMethods used:")
        for method, count in self.methods_used.items():
            if count > 0:
                print(f"  {method}: {count}")

        success_rate = (self.success_count / len(papers)) * 100 if papers else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")


async def main():
    downloader = HybridDownloader()
    await downloader.download_all()


if __name__ == '__main__':
    asyncio.run(main())
