import chromadb
import sys

print('Starting script (Direct Dump Mode)...', flush=True)
try:
    db_path = '/home/juke/git/AI-CoScientist/chromadb_data_dd'
    client = chromadb.PersistentClient(path=db_path)
    
    cols = client.list_collections()
    if not cols:
        print('No collections found!', flush=True)
        sys.exit(0)
    
    col_name = cols[0].name
    print(f'Accessing collection: {col_name}', flush=True)
    collection = client.get_collection(name=col_name)
    
    print('Fetching documents...', flush=True)
    results = collection.get(limit=1000)
    
    count = len(results['ids'])
    print(f'Found {count} documents.', flush=True)
    
    keywords = ['mamba', 'liquid', 'autism', 'multimodal', 'foundation']
    
    print('\n--- MATCHING DOCUMENTS ---', flush=True)
    match_count = 0
    for i, doc in enumerate(results['documents']):
        if doc is None: continue
        doc_lower = doc.lower()
        
        if any(k in doc_lower for k in keywords):
            meta = results['metadatas'][i]
            title = meta.get('paper_title', 'Unknown')
            print(f'\n[Match {match_count+1}] Title: {title}', flush=True)
            print(f'Snippet: {doc[:300]}...', flush=True)
            match_count += 1
            if match_count >= 5: break

except Exception as e:
    print(f'CRITICAL ERROR: {e}', flush=True)
    import traceback
    traceback.print_exc()
