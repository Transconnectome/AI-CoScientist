import asyncio
import os
from pathlib import Path
from scripts.ingest_golden_references_advanced import AdvancedGoldenReferenceIngestor, PDFExtractor

# Subclass to override collection names
class PersonaIngestor(AdvancedGoldenReferenceIngestor):
    def __init__(self, persona_name: str, chromadb_path: str = "chromadb_data"):
        # We need to initialize components manually or hack super, 
        # but let's just re-init what we need to safely override collections.
        
        # Init base (standard collections)
        super().__init__(chromadb_path)
        
        # Now OVERRIDE the collections to be persona-specific
        print(f"Creating Persona Knowledge Base: {persona_name}")
        
        self.collection_l0 = self.chroma_client.get_or_create_collection(
            name=f"{persona_name}_L0",
            metadata={"description": "Level 0: Original chunks"}
        )
        self.collection_l1 = self.chroma_client.get_or_create_collection(
            name=f"{persona_name}_L1",
            metadata={"description": "Level 1: Section summaries"}
        )
        self.collection_l2 = self.chroma_client.get_or_create_collection(
            name=f"{persona_name}_L2",
            metadata={"description": "Level 2: Paper summaries"}
        )
        print(f"✓ Persona collections initialized: {persona_name}_L[0-2]")

    async def ingest_directory(self, dir_path: Path):
        print(f"Scanning {dir_path}...")
        pdf_files = list(dir_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs.")
        
        for pdf_path in pdf_files:
            try:
                # 1. Process Paper (Extract, Chunk, RAPTOR)
                paper_data = await self.process_paper(pdf_path)
                
                if not paper_data:
                    continue
                    
                # 2. Add to ChromaDB (L0, L1, L2)
                # Note: The base class 'process_paper' ends after extraction in the original file I viewed?
                # Wait, I need to check if 'process_paper' in original file actually DOES the adding.
                # Let's check the file content again or implement the adding here.
                # Based on previous view, 'process_paper' returned data but I didn't see the finish.
                # I will implement the addition logic here to be safe and explicit.
                
                print(f"  Adding {len(paper_data.level0_chunks)} chunks to Knowledge Base...")
                
                # Add Level 0
                if paper_data.level0_chunks:
                    ids = [c.chunk_id for c in paper_data.level0_chunks]
                    docs = [c.content for c in paper_data.level0_chunks]
                    metas = [c.metadata for c in paper_data.level0_chunks]
                    # Compute embeddings (batch)
                    embeddings = self.embedding_model.encode(docs).tolist()
                    
                    self.collection_l0.add(
                        ids=ids,
                        documents=docs,
                        embeddings=embeddings,
                        metadatas=metas
                    )
                
                # Add Level 1
                if paper_data.level1_summaries:
                    ids = [n.node_id for n in paper_data.level1_summaries]
                    docs = [n.content for n in paper_data.level1_summaries]
                    metas = [n.metadata for n in paper_data.level1_summaries]
                    embeddings = self.embedding_model.encode(docs).tolist()
                    
                    self.collection_l1.add(
                        ids=ids,
                        documents=docs,
                        embeddings=embeddings,
                        metadatas=metas
                    )
                    
                # Add Level 2
                if paper_data.level2_summary:
                   n = paper_data.level2_summary
                   self.collection_l2.add(
                       ids=[n.node_id],
                       documents=[n.content],
                       embeddings=self.embedding_model.encode([n.content]).tolist(),
                       metadatas=[n.metadata]
                   )
                   
                print(f"  ✓ {paper_data.title} ingested successfully.")
                
            except Exception as e:
                print(f"  ✗ Failed to ingest {pdf_path.name}: {e}")

async def main():
    ingestor = PersonaIngestor("jiook_cha_expertise")
    await ingestor.ingest_directory(Path("/home/juke/git/AI-CoScientist/data/jiook_cha_papers"))

if __name__ == "__main__":
    asyncio.run(main())
