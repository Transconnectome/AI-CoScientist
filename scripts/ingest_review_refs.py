
import asyncio
from pathlib import Path
import sys

# Add scripts dir to path to import ingestion module
sys.path.append("/home/juke/git/AI-CoScientist/scripts")
from ingest_golden_references_advanced import AdvancedGoldenReferenceIngestor

async def main():
    print("Starting ingestion of review references...")
    ingestor = AdvancedGoldenReferenceIngestor()
    # Ensure this matches where download_review_refs.py puts files
    pdf_dir = Path("data/review-npp")
    
    if not pdf_dir.exists():
        print(f"Directory {pdf_dir} does not exist!")
        return

    await ingestor.ingest_all(pdf_dir, limit=None)

if __name__ == "__main__":
    asyncio.run(main())
