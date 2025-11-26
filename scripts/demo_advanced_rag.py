"""Demo script for Advanced Golden Reference RAG system.

Tests RAPTOR, hybrid retrieval, and agentic search capabilities.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.rag.advanced_golden_reference import (
    AdvancedGoldenReferenceStore,
    GoldenReferencePaper
)


async def main():
    """Demo the advanced RAG capabilities."""
    print("=" * 80)
    print("🧬 Advanced Golden Reference RAG Demo")
    print("=" * 80)
    
    # Initialize store
    print("\n📚 Initializing store...")
    store = AdvancedGoldenReferenceStore(collection_name="nature_demo")
    
    # Create example Nature papers
    papers = [
        GoldenReferencePaper(
            paper_id="nature_2021_alphafold",
            title="Highly accurate protein structure prediction with AlphaFold",
            journal="Nature",
            year=2021,
            abstract="Proteins are essential to life, and understanding their structure is crucial for understanding their function. Despite decades of experimental progress, computational prediction of protein structure from sequence alone has remained a challenge. Here we demonstrate AlphaFold, a computational method that achieves accuracy competitive with experiment in protein structure determination. AlphaFold uses novel machine learning approaches combined with structural biology expertise to achieve this breakthrough...",
            introduction="Protein structure prediction has been one of biology's grand challenges for over 50 years. The exponential growth in sequencing data has vastly outpaced our ability to determine structures experimentally. However, computational methods lag far behind experimental techniques in accuracy. Here, we show that deep learning can close this gap, achieving near-experimental accuracy in blind structure prediction. This advance has profound implications for understanding biological function and designing new proteins...",
            methods="We developed a novel neural network architecture that operates directly on multiple sequence alignments. The network predicts a distance matrix between residues, which is then used to generate 3D coordinates through gradient descent optimization. Key innovations include attention-based feature extraction and iterative refinement...",
            results="AlphaFold achieved a median GDTTS score of 92.4 across CASP14 targets, compared to 58.0 for the next best method. On template-free modeling targets, AlphaFold achieved unprecedented accuracy. The method correctly predicted protein structures that had previously eluded experimental determination...",
            discussion="These results demonstrate that computational methods can now rival experimental techniques in accuracy. The implications extend beyond structure prediction to protein design, drug discovery, and understanding disease mechanisms. Limitations include difficulty with intrinsically disordered regions and protein complexes...",
            citation_count=15000,
            impact_factor=49.9
        ),
        GoldenReferencePaper(
            paper_id="science_2020_crispr",
            title="CRISPR-Cas9 gene editing in human embryos",
            journal="Science",
            year=2020,
            abstract="Genetic diseases affect millions worldwide, yet therapeutic options remain limited. CRISPR-Cas9 offers unprecedented precision in genome editing, but its safety in human embryos is unproven. Here we demonstrate successful correction of a disease-causing mutation in human embryos with minimal off-target effects. This work establishes safety standards for potential clinical applications...",
            introduction="Heritable genetic diseases represent a significant burden on global health. While gene therapy approaches exist for somatic cells, germline editing could prevent disease transmission to future generations. However, ethical and safety concerns have limited research in this area. Here, we report the first successful and safe application of CRISPR-Cas9 in viable human embryos, paving the way for therapeutic interventions...",
            methods="Human embryos were obtained with informed consent following institutional guidelines. CRISPR-Cas9 components were introduced at the single-cell stage. Edited embryos were cultured to blastocyst stage and analyzed using whole-genome sequencing to detect on-target editing and potential off-target mutations...",
            citation_count=8000,
            impact_factor=47.7
        ),
        GoldenReferencePaper(
            paper_id="nature_med_2023_ai_diagnosis",
            title="AI-assisted diagnosis outperforms experts in cancer screening",
            journal="Nature Medicine",
            year=2023,
            abstract="Early detection is critical for cancer survival, yet diagnostic accuracy varies widely. Deep learning models show promise, but real-world performance remains uncertain. We developed an AI system that analyzes medical images with superhuman accuracy, detecting cancers missed by expert radiologists. Deployment in clinical settings could save thousands of lives annually...",
            introduction="Cancer remains a leading cause of death worldwide, with early detection being the most important factor in patient outcomes. However, diagnostic errors are common, with studies showing miss rates of 20-40% for certain cancer types. Artificial intelligence offers the potential to augment human expertise, but concerns about generalization and clinical utility persist. Here, we demonstrate a deep learning system that surpasses expert performance across multiple cancer types and imaging modalities...",
            methods="We trained convolutional neural networks on 1.2 million annotated medical images from 50 hospitals. The model architecture combines ResNet backbone with custom attention mechanisms. Validation was performed on held-out test sets and through prospective clinical trials involving 10,000 patients...",
            citation_count=2500,
            impact_factor=87.2
        )
    ]
    
    # Ingest papers with RAPTOR
    print("\n📥 Ingesting papers with RAPTOR indexing...")
    for paper in papers:
        await store.ingest_paper(paper)
    
    print(f"\n✅ Total nodes indexed: {len(store.raptor_nodes)}")
    
    # Test 1: Hybrid Search
    print("\n" + "=" * 80)
    print("🔍 Test 1: Hybrid Search (Dense + Sparse)")
    print("=" * 80)
    query = "protein structure prediction deep learning"
    print(f"Query: '{query}'")
    results = await store.hybrid_search(query, top_k=5)
    for i, (node, score) in enumerate(results[:3], 1):
        print(f"\n{i}. [Score: {score:.3f}] Level: {node.level.name}")
        print(f"   Content: {node.content[:150]}...")
        print(f"   Paper: {node.metadata.get('paper_id', 'Unknown')}")
    
    # Test 2: Agentic Search - Simple Query
    print("\n" + "=" * 80)
    print("🤖 Test 2: Agentic Search - Simple Query")
    print("=" * 80)
    query = "CRISPR"
    print(f"Query: '{query}' (Expected complexity: simple)")
    results = await store.agentic_search(query, query_complexity="simple")
    for i, (node, score) in enumerate(results[:2], 1):
        print(f"\n{i}. [Score: {score:.3f}] {node.content[:150]}...")
    
    # Test 3: Agentic Search - Complex Query
    print("\n" + "=" * 80)
    print("🤖 Test 3: Agentic Search - Complex Multi-hop Query")
    print("=" * 80)
    query = "How does Nature introduce machine learning papers in biology?"
    print(f"Query: '{query}' (Expected complexity: complex)")
    results = await store.agentic_search(query, query_complexity="complex")
    for i, (node, score) in enumerate(results[:3], 1):
        print(f"\n{i}. [Score: {score:.3f}] Level: {node.level.name}")
        print(f"   Content: {node.content[:120]}...")
    
    # Test 4: Pattern Extraction
    print("\n" + "=" * 80)
    print("📊 Test 4: Nature Pattern Extraction")
    print("=" * 80)
    patterns = await store.extract_nature_patterns("protein structure")
    print(f"Average Introduction Length: {patterns.get('avg_intro_length', 0):.0f} words")
    print(f"\nNarrative Hooks from Nature:")
    for i, hook in enumerate(patterns.get('narrative_hooks', [])[:3], 1):
        print(f"{i}. {hook}")
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
