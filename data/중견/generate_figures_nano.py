#!/usr/bin/env python3
"""
Generate Proposal Figures using Nano Banana (Gemini Image)
========================================================
Generates 4 key figures for the NRF Mid-Career Researcher Proposal.
"""

import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Setup
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "templates/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load API Key
# Try loading from multiple locations
env_paths = [
    SCRIPT_DIR / ".env",
    SCRIPT_DIR / "../../.env",
    Path.home() / "git/AI-CoScientist/.env"
]

for path in env_paths:
    if path.exists():
        load_dotenv(path)
        break

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found. Please set it in .env")
    exit(1)

client = genai.Client(api_key=api_key)

def generate_image(prompt, filename):
    """Generate image using Nano Banana"""
    print(f"\n🎨 Generating {filename}...")
    print(f"   Prompt: {prompt[:50]}...")
    
    try:
        start_time = time.time()
        
        # Use Nano Banana (gemini-2.5-flash-image)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE']
            )
        )
        
        elapsed = time.time() - start_time
        
        # Save image
        output_path = OUTPUT_DIR / filename
        
        # Handle response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    with open(output_path, 'wb') as f:
                        f.write(part.inline_data.data)
                    print(f"   ✅ Saved to {output_path} ({elapsed:.1f}s)")
                    return True
        
        print("   ⚠️ No image generated")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("🍌 Nano Banana Proposal Figure Generator")
    print("=" * 60)
    
    # 1. Fig 1: Problem & Gap
    prompt1 = """
    Create a professional scientific diagram comparing "Traditional Brain Age" vs "LifeSpan Trajectory Prediction".
    
    Left side (AS-IS): Shows a static brain scan with a single number "Age: 65". Label: "Static Snapshot".
    Right side (TO-BE): Shows a continuous curve graph over time (x-axis: Age, y-axis: Brain Health) with a specific trajectory line. Label: "Dynamic Trajectory".
    
    Style: Clean, minimal, academic publication quality. White background. Blue and orange color scheme.
    """
    generate_image(prompt1, "fig1_problem_gap_nano.png")
    
    # 2. Fig 2: Model Architecture
    prompt2 = """
    Create a high-tech neural network architecture diagram for "LifeSpan-FM".
    
    Structure:
    1. Bottom: Three parallel encoder blocks labeled "fMRI (100B)", "EEG (100B)", "Genomic (100B)".
    2. Middle: A fusion layer connecting them labeled "Q-Former Fusion".
    3. Top: A trajectory prediction curve labeled "Trajectory Output".
    
    Style: Modern deep learning diagram, 3D isometric view preferred but 2D is fine. Professional colors (teal, purple, white). White background.
    """
    generate_image(prompt2, "fig2_model_architecture_nano.png")
    
    # 3. Fig 3: Data Pipeline
    prompt3 = """
    Create a flowchart diagram for a medical AI data pipeline.
    
    Flow:
    1. Sources: Icons for UK Biobank, Hospitals.
    2. Arrow to "Preprocessing" box.
    3. Arrow to "Foundation Model Training" (large central box).
    4. Arrow to "Clinical Validation" box.
    
    Style: Flat design, clean lines, infographic style. White background. Professional blue tones.
    """
    generate_image(prompt3, "fig3_data_pipeline_nano.png")
    
    # 4. Fig 4: Roadmap
    prompt4 = """
    Create a 5-year project roadmap Gantt chart.
    
    Rows:
    - Year 1: Scale-up
    - Year 2: Foundation Model
    - Year 3: Fusion & Trajectory
    - Year 4: Validation
    - Year 5: Deployment
    
    Visuals: Horizontal bars spanning different lengths. 
    Style: Professional business presentation style. Clean fonts. White background.
    """
    generate_image(prompt4, "fig4_gantt_roadmap_nano.png")
    
    print("\n" + "=" * 60)
    print(f"✅ All figures generated in {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()


