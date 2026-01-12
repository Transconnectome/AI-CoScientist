import os
import requests
import json
from pathlib import Path

# Define key papers to download (Title -> PDF URL or DOI to find)
# Since direct PDF download from generic DOIs is hard without auth, 
# we will use known open access links or placeholder logic where we can't.
# For this task, we will try to find open access versions capable of being downloaded.
# In a real rigorous setting, we might need manual downloads or Unpaywall API.

# 3 key papers mentioned in research + recent works
JI_PAPERS = [
    {
        "title": "Swin fMRI Transformer Predicts Early Neurodevelopmental Outcomes from Neonatal fMRI",
        "url": "https://arxiv.org/pdf/2501.00001.pdf", # Placeholder-ish, usually we search arxiv
        "filename": "swin_fmri_transformer.pdf"
    },
    {
         "title": "Diagnosis and prognosis of Alzheimer's disease using brain morphometry and white matter connectomes",
         "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6889445/pdf/main.pdf", # Example open access
         "filename": "alzheimers_connectomes.pdf"
    },
    {
        "title": "Neural Correlates of Aggression in Medication-Naive Children with ADHD",
        "url": "https://www.nature.com/articles/npp201550.pdf", # Often guarded, but let's try or use Unpaywall
        "filename": "adhd_aggression.pdf"
    }
]

# We will use Unpaywall to be more robust if direct links fail or are unknown
# But for now, let's try a robust search/download function using semantic scholar or arxiv where possible.
# Actually, the user has the 'paper_collector.py' script. Let's reuse that logic or simplify here.

OUTPUT_DIR = Path("/home/juke/git/AI-CoScientist/data/jiook_cha_papers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    headers = {'User-Agent': 'Mozilla/5.0 (ScienceBot/1.0)'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            # Check content type if possible
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print("  ✓ Success")
                return True
            else:
                print(f"  Warning: Content-Type is {response.headers.get('Content-Type')}")
                if len(response.content) > 10000: # Heuristic
                     with open(filepath, 'wb') as f:
                        f.write(response.content)
                     print("  ✓ Saved (Check if valid PDF)")
                     return True
        print(f"  ✗ Failed: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    return False

import arxiv

def search_and_download_arxiv(query, filename):
    print(f"Searching ArXiv for: {query}")
    search = arxiv.Search(
        query = query,
        max_results = 1,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    for result in search.results():
        print(f"  Found: {result.title} ({result.entry_id})")
        pdf_path = OUTPUT_DIR / filename
        result.download_pdf(dirpath=OUTPUT_DIR, filename=filename)
        print(f"  ✓ Downloaded to {pdf_path}")
        return True
    
    print("  ✗ Not found on ArXiv")
    return False

def main():
    print("Fetching Dr. Jiook Cha's Persona Papers...")
    
    # 1. Swin fMRI Transformer (Recent)
    # Trying Arxiv search first
    if not search_and_download_arxiv("Swin fMRI Transformer", "swin_fmri_transformer.pdf"):
         pass # Backup?

    # 2. Generative AI for Memory (Recent)
    search_and_download_arxiv("Revisiting Your Memory: Reconstruction of Affect-Contextualized Memory", "generative_memory_reconstruction.pdf")

    # 3. ADHD Aggression (Try direct or use known PMC if available)
    # This is an older Nature Neuropsychopharmacology paper. Might not be on Arxiv.
    # We will skip manual download for now and rely on what we can get from Arxiv/Open sources for his AI work.
    
    # 4. Psychology/AI integration
    search_and_download_arxiv("Awe is characterized as an ambivalent experience", "awe_experience_vr.pdf")
    
    print(f"\nPersona Knowledge Base Assets in {OUTPUT_DIR}:")
    for f in OUTPUT_DIR.glob("*.pdf"):
        print(f" - {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
