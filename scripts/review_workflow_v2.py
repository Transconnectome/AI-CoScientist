import asyncio
import argparse
import os
import shutil
from pathlib import Path
from scripts.review_utils.visual_analyzer import VisualAnalyzer
from scripts.review_utils.reference_manager import ReferenceManager
from scripts.ingest_persona_knowledge import PersonaIngestor
from scripts.ingest_golden_references_advanced import PDFExtractor, MultiProviderLLM
from scripts.review_utils.ecmars_utils import get_ecmars_system_prompt_addendum
import chromadb

# Add src to path if needed
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class ReviewOrchestrator:
    def __init__(self, paper_path: str, output_dir: str):
        self.paper_path = Path(paper_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paper_id = self.paper_path.stem
        
        # Components
        self.visual_analyzer = VisualAnalyzer()
        self.ref_manager = ReferenceManager(download_dir=str(self.output_dir / "references"))
        self.pdf_extractor = PDFExtractor()
        self.llm = MultiProviderLLM()
        
        # Initialize Embedding Function (Must match Ingestion!)
        from chromadb.utils import embedding_functions
        # We need to wrap sentence_transformers to pass to Chroma
        # Or manually embed. Manual embedding is safer to ensure it matches exactly.
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
        
        # Persona Collection Name (Pre-built in Phase 1)
        self.persona_collection = "jiook_cha_expertise"
        
        # Session Collection Name (For this paper's references)
        self.session_collection = f"session_{self.paper_id}_references"
        
    def get_embedding_fn(self):
        # Create a closure compatible with Chroma
        # But actually, querying manually is often easier with complex setups.
        # Let's just use manual embedding in the query method.
        return self.embedding_model.encode
        

    async def run_visual_analysis(self) -> str:
        print("\n[Step 1] Visual Analysis...")
        report = self.visual_analyzer.analyze_visuals(str(self.paper_path))
        
        # Save report
        with open(self.output_dir / "visual_critique.md", "w") as f:
            f.write(report)
        print("  ✓ Visual critique generated.")
        return report

    async def build_reference_context(self, paper_text: str):
        print("\n[Step 2] Building SOTA Reference Context...")
        
        # 1. Extract Refs
        refs = self.ref_manager.extract_references_from_text(paper_text)
        print(f"  Extracted {len(refs)} raw references.")
        
        # 2. Filter & Download SOTA
        pdf_paths = await self.ref_manager.filter_and_download_sota(refs, max_download=5) # Limit to 5 for speed
        
        if not pdf_paths:
            print("  No SOTA references found or downloaded.")
            return

        # 3. Ingest into Session Collection
        # utilizing PersonaIngestor but with session name
        ingestor = PersonaIngestor(self.session_collection)
        
        # We need to manually iterate and process since ingest_directory expects a dir
        # But we have a list of specific paths.
        print(f"  Ingesting {len(pdf_paths)} papers into '{self.session_collection}'...")
        
        # Simpler: Just run ingest_directory on the reference folder
        # The ReferenceManager downloads to self.output_dir / "references"
        print(f"  Ingesting {len(pdf_paths)} papers into '{self.session_collection}'...")
        await ingestor.ingest_directory(self.output_dir / "references")
        print("  ✓ Reference Context Built.")

    async def generate_final_review(self, paper_text: str, visual_report: str):
        print("\n[Step 3] Generating Final Review with Persona & Context...")
        
        # Connect to ChromaDB to query our contexts
        client = chromadb.PersistentClient(path="chromadb_data")
        
        # 1. Query Persona Knowledge (Jiook Cha)
        # What would Jiook Cha say about this topic?
        # Manually embed the query text
        query_text = paper_text[:2000]
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        try:
            persona_coll = client.get_collection(f"{self.persona_collection}_L1") # Use L1 summaries
            persona_results = persona_coll.query(
                query_embeddings=query_embedding,
                n_results=3
            )
            persona_context = "\n".join(persona_results['documents'][0]) if persona_results['documents'] else ""
        except Exception as e:
            print(f"  Warning: Persona query failed ({e}). Proceeding without.")
            persona_context = ""

        # 2. Query SOTA Context (Session References)
        # Check for conflicts or similarities
        try:
            session_coll = client.get_collection(f"{self.session_collection}_L1")
            session_results = session_coll.query(
                query_embeddings=query_embedding,
                n_results=3
            )
            context_docs = session_results['documents'][0] if session_results['documents'] else []
            sota_context = "\n".join(context_docs)
            print(f"  ✓ Retrieved {len(context_docs)} SOTA context chunks.")
        except Exception as e:
            print(f"  Warning: Session query failed ({e}). Proceeding without.")
            sota_context = ""
            
        # 3. Construct Prompt
        base_system_prompt = """You are Dr. Jiook Cha, a world-class expert in AI and Neuroscience (Transformers, fMRI, Psychiatric AI).
        You are reviewing a paper for CVPR/NeurIPS.
        
        YOUR GOAL: Write a brutal but constructive, expert-level review.
        
        INPUTS:
        1. Target Paper Text
        2. Visual Analysis Report (Generated by Vision Model)
        3. Your Persona Knowledge (Your own past work/expertise)
        4. SOTA Reference Context (Related papers found in bibliography)
        
        INSTRUCTIONS:
        - TONE: Professional, rigorous, demanding perfection in experimental design and visual presentation.
        - STRUCTURE:
            0. **ECMARS Dashboard** (See below)
            1. Summary
            2. Strengths (Be specific)
            3. Weaknesses (Crucial part)
            4. **Visual Critique** (Synthesize the Visual Report here)
            5. **Relation to SOTA & Novelty** (Use SOTA Context)
            6. **Neuro-AI Perspective** (If applicable, critique biological plausibility)
            7. Detailed Feedback
            8. Final Recommendation (Strong Accept...Strong Reject)
        """
        
        # Inject ECMARS Logic
        ecmars_addendum = get_ecmars_system_prompt_addendum()
        full_system_prompt = base_system_prompt + "\n" + ecmars_addendum
        
        user_prompt = f"""
        ### VISUAL ANALYSIS REPORT:
        {visual_report}
        
        ### PERSONA KNOWLEDGE (Your Expertise):
        {persona_context}
        
        ### SOTA REFERENCE CONTEXT (Related Work):
        {sota_context}
        
        ### TARGET PAPER TEXT:
        {paper_text[:30000]}... [Truncated]
        """
        
        response, provider = await self.llm.generate(f"{full_system_prompt}\n\n{user_prompt}", max_tokens=4000)
        
        return response

    async def run(self):
        print(f"Starting Review Process for {self.paper_id}...")
        
        # 1. Read Text
        text = self.pdf_extractor.extract_text(self.paper_path)
        if not text:
            print("Failed to extract text.")
            return

        # 2. Visual Analysis (Parallelizable but sequential for now)
        visual_report = await self.run_visual_analysis()
        
        # 3. Build Context (Async)
        await self.build_reference_context(text)
        
        # 4. Generate Review
        review = await self.generate_final_review(text, visual_report)
        
        # 5. Save
        out_path = self.output_dir / f"Review_{self.paper_id}_JiookCha.md"
        with open(out_path, "w") as f:
            f.write(review)
            
        print(f"\nSUCCESS. Review saved to {out_path}")
        print("="*60)
        print(review[:1000] + "...")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, help="Path to PDF")
    parser.add_argument("--out", default="reviews_output", help="Output directory")
    args = parser.parse_args()
    
    orchestrator = ReviewOrchestrator(args.paper, args.out)
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())
