import os
import tempfile
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image

class VisualAnalyzer:
    """
    Analyzes the visual components of a paper (Figures, Tables) using Multimodal LLMs.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-3-pro-preview"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def convert_pdf_to_images(self, pdf_path: str, max_pages: int = 10) -> List[Image.Image]:
        """Convert PDF pages to PIL Images."""
        print(f"Converting PDF to images: {pdf_path}")
        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
            print(f"  ✓ Converted {len(images)} pages.")
            return images
        except Exception as e:
            print(f"  ✗ PDF conversion failed: {e}")
            return []
            
    def analyze_visuals(self, pdf_path: str) -> str:
        """
        Perform a comprehensive visual analysis of the paper.
        Returns a markdown report.
        """
        images = self.convert_pdf_to_images(pdf_path)
        if not images:
            return "Visual Analysis Failed: Could not convert PDF to images."
            
        # We process in batches if too many, but usually 10 pages fits in Gemini 1.5 context easily.
        # We will send the images along with a specific prompt.
        
        prompt = """
        You are an expert Reviewer for top AI conferences (CVPR, NeurIPS).
        Your task is to CRITIQUE the VISUAL presentation of this paper based on the attached page images.
        
        Focus specifically on:
        1. **Figures**: Are they high quality? Do they clearly explain the method? Are the captions descriptive enough?
           - Identify the Key Architecture Figure (usually Fig 1 or 2). Does it make sense?
        2. **Tables**: Are the results bolded correctly? Is the comparison fair (same datasets/settings)?
           - Look for "SOTA" comparisons. 
        3. **Legibility**: Are fonts too small? Are plots cluttered?
        4. **Inconsistencies**: Do you see any obvious errors (e.g. caption says "lower is better" but bolded higher numbers)?
        
        Output a distinct section titled "## Visual Analysis & Evidence Check".
        Be critical. If a figure is confusing, say so.
        """
        
        print("Sending images to Gemini Vision for analysis...")
        try:
            # Inputs: prompt + list of images
            inputs = [prompt] + images
            
            response = self.model.generate_content(inputs)
            return response.text
        except Exception as e:
            return f"Visual Analysis Error: {e}"

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        analyzer = VisualAnalyzer()
        report = analyzer.analyze_visuals(sys.argv[1])
        print(report)
