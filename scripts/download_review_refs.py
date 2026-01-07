
import asyncio
import json
from pathlib import Path
from unpaywall_downloader import UnpaywallDownloader

# Configure paths
REFS_FILE = "data/reference_papers/review_npp_refs.json"
OUTPUT_DIR = "data/review-npp"

async def main():
    print(f"Starting download for {REFS_FILE}...")
    
    # Instantiate downloader with correct file
    downloader = UnpaywallDownloader(papers_file=REFS_FILE)
    
    # Override output directory
    downloader.output_dir = Path(OUTPUT_DIR)
    downloader.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run
    await downloader.download_all()

if __name__ == "__main__":
    asyncio.run(main())
