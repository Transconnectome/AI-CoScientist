
import asyncio
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import time

async def test_large_pdf():
    pdf_path = Path("data/reference_papers/pdfs/A_model_of_human_neural_networks_reveals_NPTX2_pathology_in_ALS_and_FTLD.pdf")
    
    print(f"Testing with file: {pdf_path}")
    print(f"File size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 1. Test PDF Extraction
    print("\n1. Testing PDF Extraction...")
    start_time = time.time()
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            print(f"  Pages: {len(reader.pages)}")
            text_parts = []
            for i, page in enumerate(reader.pages):
                if i % 10 == 0:
                    print(f"  Processing page {i}...", end='\r')
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            full_text = '\n\n'.join(text_parts)
            print(f"\n  ✓ Extracted {len(full_text):,} characters in {time.time() - start_time:.2f}s")
            
    except Exception as e:
        print(f"\n  ✗ PDF Extraction failed: {e}")
        return

    # 2. Test SciBERT
    print("\n2. Testing SciBERT Encoding...")
    try:
        model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        print("  ✓ Model loaded")
        
        # Create dummy chunks
        chunks = [full_text[i:i+512] for i in range(0, min(len(full_text), 50000), 512)]
        print(f"  Encoding {len(chunks)} chunks...")
        
        start_time = time.time()
        embeddings = model.encode(chunks, show_progress_bar=True)
        print(f"  ✓ Encoded in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"  ✗ SciBERT failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_large_pdf())
