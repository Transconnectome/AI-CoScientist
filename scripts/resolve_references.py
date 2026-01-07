
import re
import asyncio
import aiohttp
import json
from pathlib import Path

PAPER_TEXT_PATH = "/home/juke/git/AI-CoScientist/paper_text.txt"
OUTPUT_JSON = "/home/juke/git/AI-CoScientist/data/reference_papers/review_npp_refs.json"
S2_API_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"

async def resolve_reference(session, ref_text):

    # Clean ref text to get a searchable title or citation
    # Remove leading number (e.g. "1 ")
    clean_ref = re.sub(r'^\d+\s+', '', ref_text).strip()
    
    # Remove newlines and extra spaces
    clean_ref = re.sub(r'\s+', ' ', clean_ref)
    
    # Heuristic: The title is usually the part until the journal name or year.
    # But finding the journal name is hard.
    # Let's take the first ~80 characters which likely includes Author + Title start.
    # Semantic Scholar is good at fuzzy match.
    # OR, try to extract just the title. 
    # References are: Authors. Title. Journal.
    # Let's try to split by '.' and use the second part if available, or first + second.
    

    # Refined Heuristic:
    # 1. Split by ". " to handle "et al." and initials better.
    # 2. The Title is often the longest segment or the second segment.
    
    parts = re.split(r'\. ', clean_ref)
    
    # Filter out short parts (initials, dates)
    long_parts = [p for p in parts if len(p) > 20]
    
    if len(long_parts) >= 2:
        # Usually: Authors (long), Title (long), Journal (long)
        # Try the second long part if available, otherwise the first.
        # But sometimes authors are short "Smith J."
        # Let's try finding the longest part, it's usually the title or authors+title.
        query = max(long_parts, key=len)[:200]
    elif long_parts:
        query = long_parts[0][:200]
    else:
        query = clean_ref[:200]
        
    print(f"DEBUG: Querying: '{query}'")

    
    params = {
        'query': query,
        'limit': 1,
        'fields': 'paperId,externalIds,title,venue,year,openAccessPdf'
    }

    
    try:
        async with session.get(S2_API_SEARCH, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get('data'):
                    paper = data['data'][0]
                    # Construct URL
                    ext_ids = paper.get('externalIds', {})
                    if 'DOI' in ext_ids:
                        url = f"https://doi.org/{ext_ids['DOI']}"
                    else:
                        url = f"https://www.semanticscholar.org/paper/{paper['paperId']}"
                        
                    return {
                        'title': paper['title'],
                        'original_ref': ref_text,
                        'url': url,
                        'paperId': paper['paperId'],
                        'venue': paper.get('venue'),
                        'year': paper.get('year')
                    }
    except Exception as e:
        print(f"Error resolving {query[:50]}...: {e}")
    return None

async def main():
    # 1. Extract References from Text
    with open(PAPER_TEXT_PATH, 'r') as f:
        text = f.read()
    
    # Locate references section
    # Based on file view, it starts around "1 Goodman WK..."
    # We can regex for lines starting with number
    
    refs = []
    # Find start of references
    lines = text.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "1 Goodman WK, Storch EA, Sheth SA. Harmonizing the Neurobiology and Treatment":
             start_idx = i
             break
        if "1 Goodman WK" in line:
            start_idx = i
            break
            
    if start_idx == 0:
        # Fallback search
        ref_pattern = re.compile(r'^\s*1\s+Goodman')
        for i, line in enumerate(lines):
            if ref_pattern.search(line):
                start_idx = i
                break

    print(f"References start at line {start_idx}")
    
    raw_refs = []
    current_ref = ""
    # Regex to detect start of new reference (number at start of line)
    new_ref_pattern = re.compile(r'^\s*(\d+)\s+')
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line: continue
        
        match = new_ref_pattern.match(line)
        if match:
            if current_ref:
                raw_refs.append(current_ref)
            current_ref = line
        else:
            current_ref += " " + line
            
    if current_ref:
        raw_refs.append(current_ref)
        
    print(f"Found {len(raw_refs)} references.")
    
    # 2. Resolve them
    resolved_papers = []
    async with aiohttp.ClientSession() as session:
        for i, ref in enumerate(raw_refs):
            print(f"Resolving {i+1}/{len(raw_refs)}...")
            res = await resolve_reference(session, ref)
            if res:
                resolved_papers.append(res)
                print(f"  Found: {res['title']}")
            else:
                print(f"  Failed to resolve: {ref[:50]}...")
            await asyncio.sleep(0.5) # Rate limit
            
    # 3. Save
    output_data = {'papers': resolved_papers}
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Saved {len(resolved_papers)} resolved papers to {OUTPUT_JSON}")

if __name__ == "__main__":
    asyncio.run(main())
