#!/usr/bin/env python3
"""
Download papers from PubMed Central (PMC) for biomedical papers.
PMC has a large open access collection.
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional
import aiohttp
import aiofiles


class PMCDownloader:
    def __init__(self, papers_file: str = "data/reference_papers/paper_urls_diverse.json"):
        self.papers_file = Path(papers_file)
        self.output_dir = Path("data/reference_papers/pdfs")
        self.output_dir.mkdir(exist_ok=True)

        self.pmc_api = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.success_count = 0
        self.fail_count = 0
        self.skip_count = 0

    def sanitize_filename(self, title: str) -> str:
        """Create safe filename."""
        filename = re.sub(r'[^\w\s-]', '', title)
        filename = re.sub(r'\s+', '_', filename)[:100]
        return filename + '.pdf'

    async def get_pmcid_from_doi(self, doi: str) -> str:
        """Get PMC ID from DOI using PubMed ID Converter."""
        url = f"{self.pmc_api}/esearch.fcgi"
        params = {
            'db': 'pmc',
            'term': f'{doi}[DOI]',
            'retmode': 'xml'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        root = ET.fromstring(xml_content)

                        id_list = root.find('.//IdList')
                        if id_list is not None:
                            pmc_id = id_list.find('Id')
                            if pmc_id is not None:
                                return pmc_id.text
        except:
            pass
        return None

    async def download_from_pmc(self, pmcid: str, output_path: Path) -> bool:
        """Download PDF from PMC."""
        # PMC OA PDF URL format
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=60), allow_redirects=True) as response:
                    if response.status == 200:
                        content = await response.read()

                        if content[:4] == b'%PDF' and len(content) > 50000:
                            async with aiofiles.open(output_path, 'wb') as f:
                                await f.write(content)

                            size_kb = len(content) / 1024
                            print(f"    ✓ Downloaded from PMC ({size_kb:.1f}KB)")
                            return True
        except:
            pass
        return False

    async def download_paper(self, paper: Dict, index: int, total: int) -> bool:
        """Download a single paper from PMC."""
        print(f"\n[{index}/{total}] {paper['title'][:70]}...")
        print(f"  Journal: {paper.get('venue', 'Unknown')}")

        filename = self.sanitize_filename(paper['title'])
        output_path = self.output_dir / filename

        if output_path.exists():
            print(f"  ✓ Already downloaded")
            self.skip_count += 1
            return True

        # Extract DOI
        url = paper['url']
        if 'doi.org/' not in url:
            self.fail_count += 1
            return False

        doi = url.split('doi.org/')[-1]

        # Get PMC ID
        print(f"  [PMC] Looking up DOI: {doi}")
        pmcid = await self.get_pmcid_from_doi(doi)

        if pmcid:
            print(f"  [PMC] Found PMC ID: PMC{pmcid}")
            if await self.download_from_pmc(pmcid, output_path):
                self.success_count += 1
                return True

        print(f"  ✗ Not available in PMC")
        self.fail_count += 1
        return False

    async def download_all(self):
        """Download all papers."""
        with open(self.papers_file, 'r') as f:
            papers = json.load(f)['papers']

        # Focus on biomedical journals (more likely in PMC)
        biomedical_journals = ['Nature Medicine', 'Nature Biomedical Engineering']
        papers = [p for p in papers if p.get('venue') in biomedical_journals]

        print("="*70)
        print("PMC PAPER DOWNLOADER")
        print("="*70)
        print(f"Papers to download: {len(papers)}")
        print(f"Focusing on: {', '.join(biomedical_journals)}")
        print("="*70)

        for i, paper in enumerate(papers, 1):
            await self.download_paper(paper, i, len(papers))
            await asyncio.sleep(0.5)  # Rate limiting

        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Already downloaded: {self.skip_count}/{len(papers)}")
        print(f"Newly downloaded: {self.success_count}/{len(papers)}")
        print(f"Failed: {self.fail_count}/{len(papers)}")


async def main():
    downloader = PMCDownloader()
    await downloader.download_all()


if __name__ == '__main__':
    asyncio.run(main())
