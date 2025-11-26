#!/usr/bin/env python3
"""
Verify manually downloaded papers and create summary report.
"""

import json
from pathlib import Path
import PyPDF2

def verify_pdf(pdf_path: Path) -> dict:
    """Verify PDF is readable and extract metadata."""
    try:
        with open(pdf_path, 'rb') as f:
            # Check PDF magic bytes
            header = f.read(4)
            if header != b'%PDF':
                return {'readable': False, 'error': 'Not a valid PDF'}

            # Try to read with PyPDF2
            f.seek(0)
            try:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)

                # Try to extract text from first page
                try:
                    first_page_text = reader.pages[0].extract_text()
                    has_text = len(first_page_text) > 100
                except:
                    has_text = False

                return {
                    'readable': True,
                    'pages': num_pages,
                    'has_text': has_text,
                    'size_mb': pdf_path.stat().st_size / (1024 * 1024)
                }
            except Exception as e:
                return {'readable': False, 'error': str(e)}
    except Exception as e:
        return {'readable': False, 'error': str(e)}

def main():
    pdf_dir = Path("data/reference_papers/pdfs")
    pdfs = sorted(pdf_dir.glob("*.pdf"))

    print("=" * 70)
    print("MANUAL DOWNLOAD VERIFICATION REPORT")
    print("=" * 70)
    print(f"Total PDFs found: {len(pdfs)}")
    print("=" * 70)

    results = {
        'readable': 0,
        'unreadable': 0,
        'with_text': 0,
        'total_size_mb': 0,
        'papers': []
    }

    for pdf in pdfs:
        verification = verify_pdf(pdf)

        if verification['readable']:
            results['readable'] += 1
            results['total_size_mb'] += verification['size_mb']

            if verification.get('has_text'):
                results['with_text'] += 1

            results['papers'].append({
                'filename': pdf.name,
                'pages': verification['pages'],
                'size_mb': round(verification['size_mb'], 2),
                'has_text': verification.get('has_text', False)
            })

            status = "✓" if verification.get('has_text') else "⚠"
            print(f"{status} {pdf.name[:60]:60} | {verification['pages']:3}p | {verification['size_mb']:5.1f}MB")
        else:
            results['unreadable'] += 1
            print(f"✗ {pdf.name[:60]:60} | ERROR: {verification.get('error', 'Unknown')}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total papers: {len(pdfs)}")
    print(f"Readable PDFs: {results['readable']}/{len(pdfs)} ({results['readable']/len(pdfs)*100:.1f}%)")
    print(f"With extractable text: {results['with_text']}/{len(pdfs)} ({results['with_text']/len(pdfs)*100:.1f}%)")
    print(f"Unreadable: {results['unreadable']}")
    print(f"Total size: {results['total_size_mb']:.1f} MB")
    print(f"Average size: {results['total_size_mb']/len(pdfs):.1f} MB per paper")

    # Save report
    with open('data/reference_papers/verification_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Detailed report saved to: data/reference_papers/verification_report.json")

if __name__ == '__main__':
    main()
