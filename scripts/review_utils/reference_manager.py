import re
import asyncio
import aiohttp
import arxiv
from pathlib import Path
from typing import List, Dict

# SOTA Venues to prioritize
TOP_VENUES = [
    "CVPR", "ICCV", "ECCV", "NeurIPS", "ICML", "ICLR", "AAAI", "MICCAI", 
    "Nature", "Science", "Nature Neuroscience", "Nature Medicine"
]

class ReferenceManager:
    """
    Manages the extraction, validation, and downloading of references 
    to build a SOTA context for reviewing.
    """
    
    def __init__(self, download_dir: str = "data/review_references"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_references_from_text(self, text: str) -> List[str]:
        """
        Heuristic extraction of reference strings from paper text.
        Assumes standard format [1] Author et al. Title. Venue Year.
        """
        # Simple fallback regex for demonstration. 
        # In production, we'd use the LLM to parse specific lines or a proper parser.
        # Looking for lines that look like citations.
        
        # Regex for [1] ... or 1. ...
        ref_pattern = r'^\s*\[?\d+\]?\.?\s+(.+?(\d{4}).*?)$'
        matches = []
        for line in text.split('\n'):
            line = line.strip()
            match = re.match(ref_pattern, line)
            if match and len(line) > 50:
                matches.append(line)
        
        # Try to find "References" section first
        references_text = text
        parts = re.split(r'\nReferences\n|\nBibliography\n', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            references_text = parts[-1]
            
        # Regex for lines starting with [1], 1., or just AuthorName (heuristic)
        # We'll just take non-empty lines from the reference section as a fallback
        # because PDF extraction often mangles newlines in references.
        
        matches = []
        # split by patterns like [1], [2]
        # or just split by newlines if simple
        
        # Robust split: Look for [number] at start of line
        ref_split = re.split(r'\[\d+\]', references_text)
        if len(ref_split) > 5: # Successful split
            matches = [r.strip() for r in ref_split if len(r.strip()) > 20]
        else:
            # Fallback to newline splitting
            matches = [l.strip() for l in references_text.split('\n') if len(l.strip()) > 30]
            
        print(f"  (Debug) Found {len(matches)} reference strings.")
        return matches[:40] # Check more references
        
    async def filter_and_download_sota(self, references: List[str], max_download: int = 10) -> List[str]:
        """
        Check which references are from TOP VENUES and download them.
        Returns list of paths to downloaded PDFs.
        """
        print(f"Filtering {len(references)} references for Top Venues...")
        
        candidates = []
        
        for ref in references:
            # Check for venue keywords in the string itself (fastest)
            # e.g. "Proc. CVPR" or "In NeurIPS"
            # Normalize text
            ref_lower = ref.lower()
            is_top = any(v.lower() in ref_lower for v in TOP_VENUES)
            
            # Debug check
            # if "cvpr" in ref_lower: print(f"  Matched CVPR: {ref[:50]}...")
            
            if is_top:
                # Extract Title heuristic (between Author and Venue/Year)
                candidates.append(ref)
                
        print(f"  Found {len(candidates)} Top-Venue candidates.")
        
        downloaded_paths = []
        
        # Download top N candidates
        for i, ref_text in enumerate(candidates[:max_download]):
            print(f"  Processing {i+1}/{max_download}: {ref_text[:60]}...")
            
            # Heuristic title extraction: clean up [1] and Author names roughly
            # We will use Arxiv Search with the full string, it's surprisingly robust.
            
            # Clean leading numbers
            query = re.sub(r'^\[?\d+\]?\.?\s*', '', ref_text)
            # Truncate to first 200 chars to avoid noise
            query = query[:200]
            
            path = await self._download_from_arxiv(query)
            if path:
                downloaded_paths.append(str(path))
            
        return downloaded_paths
    
    async def _download_from_arxiv(self, query: str) -> Path:
        """Search and download from Arxiv."""
        try:
            search = arxiv.Search(
                query = query,
                max_results = 1,
                sort_by = arxiv.SortCriterion.Relevance
            )
            
            client = arxiv.Client()
            for result in client.results(search):
                # Sanity check: Title overlap?
                # Sanity check: Title overlap?
                # If the result title is completely different, skip.
                # But for now, we trust Arxiv relevance.
                
                valid_filename = "".join(c for c in result.title if c.isalnum() or c in " _-")[:50] + ".pdf"
                dest = self.download_dir / valid_filename
                
                if dest.exists():
                    print(f"    (Cached) {valid_filename}")
                    return dest
                
                print(f"    found: {result.title}")
                result.download_pdf(dirpath=self.download_dir, filename=valid_filename)
                print(f"    ✓ Downloaded")
                return dest
                
        except Exception as e:
            print(f"    ✗ Arxiv failed: {e}")
            
        return None

if __name__ == "__main__":
    # Test stub
    mgr = ReferenceManager()
    dummy_refs = [
        "[1] He, K. et al. Deep Residual Learning for Image Recognition. CVPR 2016.",
        "[2] Vaswani, A. et al. Attention is All You Need. NeurIPS 2017.",
        "[3] Random Paper in Random Journal 2000."
    ]
    asyncio.run(mgr.filter_and_download_sota(dummy_refs))
