#!/usr/bin/env python3
"""
Download papers using Unpaywall API for open access content.
Unpaywall has better OA detection than Semantic Scholar.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Optional
import aiohttp
import aiofiles


class UnpaywallDownloader:
    def __init__(self, papers_file: str = "data/reference_papers/paper_urls_diverse.json"):
        self.papers_file = Path(papers_file)
        self.output_dir = Path("data/reference_papers/pdfs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.unpaywall_api = "https://api.unpaywall.org/v2"
        self.email = "jiook.cha@snu.ac.kr"  # Required for Unpaywall API

        self.success_count = 0
        self.fail_count = 0
        self.skip_count = 0

    def sanitize_filename(self, title: str, max_length: int = 100) -> str:
        """Create safe filename from paper title."""
        filename = re.sub(r'[^\w\s-]', '', title)
        filename = re.sub(r'\s+', '_', filename)
        if len(filename) > max_length:
            filename = filename[:max_length]
        return filename + '.pdf'

    async def download_file(self, url: str, output_path: Path) -> bool:
        """Download file from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60), allow_redirects=True) as response:
                    if response.status == 200:
                        content = await response.read()

                        # Verify it's a PDF
                        if content[:4] != b'%PDF':
                            print(f"    ⚠️  Not a PDF (got {len(content)} bytes)")
                            return False

                        async with aiofiles.open(output_path, 'wb') as f:
                            await f.write(content)

                        size_kb = len(content) / 1024
                        if size_kb < 50:
                            print(f"    ⚠️  File too small ({size_kb:.1f}KB)")
                            output_path.unlink()
                            return False

                        print(f"    ✓ Downloaded ({size_kb:.1f}KB)")
                        return True
                    else:
                        print(f"    ✗ HTTP {response.status}")
                        return False

        except Exception as e:
            print(f"    ✗ Error: {type(e).__name__}")
            return False

    async def try_unpaywall(self, paper: Dict) -> Optional[Path]:
        """Try Unpaywall API for open access PDF."""
        try:
            url = paper['url']

            # Extract DOI
            if 'doi.org/' in url:
                doi = url.split('doi.org/')[-1]
            elif '/articles/' in url:
                # Nature article ID
                article_id = url.split('/articles/')[-1]
                doi = article_id
            else:
                return None

            # Query Unpaywall
            lookup_url = f"{self.unpaywall_api}/{doi}"
            params = {'email': self.email}

            async with aiohttp.ClientSession() as session:
                async with session.get(lookup_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Try multiple OA locations
                        oa_locations = []

                        # Best OA location
                        best_oa = data.get('best_oa_location')
                        if best_oa and best_oa.get('url_for_pdf'):
                            oa_locations.append(('best', best_oa['url_for_pdf']))

                        # All OA locations
                        for loc in data.get('oa_locations', []):
                            if loc.get('url_for_pdf'):
                                oa_locations.append(('all', loc['url_for_pdf']))

                        # Try each location
                        for source, pdf_url in oa_locations:
                            print(f"  [Unpaywall-{source}] Trying OA PDF...")

                            filename = self.sanitize_filename(paper['title'])
                            output_path = self.output_dir / filename

                            if await self.download_file(pdf_url, output_path):
                                return output_path

                    elif response.status == 404:
                        print(f"  [Unpaywall] Paper not found in database")
                    else:
                        print(f"  [Unpaywall] API error {response.status}")

            return None

        except Exception as e:
            print(f"  [Unpaywall] Error: {type(e).__name__}: {e}")
            return None

    async def download_paper(self, paper: Dict, index: int, total: int) -> bool:
        """Download a single paper."""
        print(f"\n[{index}/{total}] {paper['title'][:80]}...")
        print(f"  Journal: {paper.get('venue', 'Unknown')}")

        # Check if already downloaded
        filename = self.sanitize_filename(paper['title'])
        output_path = self.output_dir / filename
        if output_path.exists():
            print(f"  ✓ Already downloaded, skipping")
            self.skip_count += 1
            return True

        # Try Unpaywall
        result = await self.try_unpaywall(paper)
        if result:
            self.success_count += 1
            return True

        print(f"  ✗ No open access PDF found")
        self.fail_count += 1
        return False

    async def download_all(self):
        """Download all papers."""
        with open(self.papers_file, 'r') as f:
            data = json.load(f)
        papers = data['papers']

        print("="*70)
        print("UNPAYWALL PAPER DOWNLOADER")
        print("="*70)
        print(f"Papers to download: {len(papers)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Email for API: {self.email}")
        print("="*70)

        for i, paper in enumerate(papers, 1):
            await self.download_paper(paper, i, len(papers))
            await asyncio.sleep(1)  # Rate limiting

        # Summary
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Already downloaded: {self.skip_count}/{len(papers)}")
        print(f"Newly downloaded: {self.success_count}/{len(papers)}")
        print(f"Failed: {self.fail_count}/{len(papers)}")
        print(f"Total PDFs: {self.skip_count + self.success_count}/{len(papers)}")

        total_success = self.skip_count + self.success_count
        success_rate = (total_success / len(papers)) * 100 if papers else 0
        print(f"\nOverall success rate: {success_rate:.1f}%")


async def main():
    downloader = UnpaywallDownloader()
    await downloader.download_all()


if __name__ == '__main__':
    asyncio.run(main())
