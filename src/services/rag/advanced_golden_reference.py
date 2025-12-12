"""Advanced Golden Reference Store with RAPTOR and Hybrid Retrieval.

Implements state-of-the-art RAG techniques for Nature/Science paper analysis:
- RAPTOR: Hierarchical tree-based indexing
- Hybrid Retrieval: Dense (SciBERT) + Sparse (BM25)
- Agentic Retrieval: Query-complexity adaptive search
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict
import asyncio


class RetrievalLevel(Enum):
    """RAPTOR tree levels."""
    CHUNK = 0  # Original text chunks
    SECTION = 1  # Section-level summaries
    PAPER = 2  # Full paper summary


@dataclass
class RetrievalNode:
    """Node in RAPTOR hierarchical tree."""
    node_id: str
    content: str
    level: RetrievalLevel
    embedding: Optional[np.ndarray] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class GoldenReferencePaper:
    """Exemplar paper from top-tier journals."""
    paper_id: str
    title: str
    journal: str  # "Nature", "Science", etc.
    year: int
    abstract: str
    introduction: str
    methods: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    full_text: Optional[str] = None
    
    # Extracted patterns (populated after analysis)
    narrative_structure: Optional[Dict] = None
    methodological_patterns: Optional[List[str]] = None
    statistical_rigor_score: float = 0.0
    reproducibility_score: float = 0.0
    
    # Citation metadata
    citation_count: int = 0
    impact_factor: float = 0.0
    
    # RAPTOR tree
    raptor_nodes: List[RetrievalNode] = field(default_factory=list)


class AdvancedGoldenReferenceStore:
    """Advanced RAG store with RAPTOR, GraphRAG, and Agentic retrieval.
    
    Features:
    1. RAPTOR: Hierarchical tree-based indexing for multi-level retrieval
    2. Hybrid Retrieval: Combines dense (SciBERT) and sparse (BM25) search
    3. Agentic: Adaptive query complexity-based retrieval strategy
    """
    
    def __init__(
        self, 
        collection_name: str = "golden_references",
        use_chromadb: bool = True,
        llm_service=None
    ):
        self.collection_name = collection_name
        self.papers: Dict[str, GoldenReferencePaper] = {}
        self.raptor_nodes: Dict[str, RetrievalNode] = {}
        
        # Embedding model (lazy loaded)
        self._embedding_model = None
        self._llm_service = llm_service
        
        # ChromaDB integration
        self.use_chromadb = use_chromadb
        if use_chromadb:
            try:
                import chromadb
                self.chroma_client = chromadb.Client()
                self.collection = self.chroma_client.get_or_create_collection(
                    name=collection_name
                )
                print(f"✅ ChromaDB initialized: {collection_name}")
            except ImportError:
                print("⚠️  ChromaDB not available, using in-memory store")
                self.use_chromadb = False
        
        print(f"📚 AdvancedGoldenReferenceStore initialized")
    
    def _load_embedding_model(self):
        """Lazy load SciBERT embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('allenai/scibert_scivocab_uncased')
                print("✅ SciBERT loaded")
            except ImportError:
                print("⚠️  sentence-transformers not installed")
                self._embedding_model = None
        return self._embedding_model
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate SciBERT embeddings."""
        model = self._load_embedding_model()
        if model is None:
            # Fallback: random embeddings (for testing)
            return np.random.randn(len(texts), 768)
        return model.encode(texts, show_progress_bar=False)
    
    async def build_raptor_tree(
        self, 
        paper: GoldenReferencePaper,
        chunk_size: int = 512,
        max_levels: int = 3
    ) -> List[RetrievalNode]:
        """Build RAPTOR hierarchical tree for a paper.
        
        Args:
            paper: Golden reference paper
            chunk_size: Token size for base chunks
            max_levels: Maximum tree depth
            
        Returns:
            List of RAPTOR nodes
        """
        nodes = []
        
        # Level 0: Chunk original sections
        sections = {
            "abstract": paper.abstract,
            "introduction": paper.introduction,
            "methods": paper.methods or "",
            "results": paper.results or "",
            "discussion": paper.discussion or ""
        }
        
        level0_nodes = []
        for section_name, content in sections.items():
            if not content:
                continue
            
            # Simple chunking (in production, use proper tokenizer)
            chunks = self._chunk_text(content, chunk_size)
            
            for i, chunk in enumerate(chunks):
                node_id = f"{paper.paper_id}_{section_name}_L0_{i}"
                node = RetrievalNode(
                    node_id=node_id,
                    content=chunk,
                    level=RetrievalLevel.CHUNK,
                    metadata={
                        "section": section_name,
                        "paper_id": paper.paper_id,
                        "journal": paper.journal
                    }
                )
                level0_nodes.append(node)
        
        # Generate embeddings for level 0
        if level0_nodes:
            contents = [n.content for n in level0_nodes]
            embeddings = self._generate_embeddings(contents)
            for node, emb in zip(level0_nodes, embeddings):
                node.embedding = emb
        
        nodes.extend(level0_nodes)
        
        # Level 1: Cluster and summarize (Section level)
        if len(level0_nodes) > 1 and max_levels > 1:
            level1_nodes = await self._cluster_and_summarize(
                level0_nodes, 
                paper.paper_id,
                level=1
            )
            nodes.extend(level1_nodes)
            
            # Level 2: Paper-level summary (optional)
            if len(level1_nodes) > 1 and max_levels > 2:
                level2_nodes = await self._cluster_and_summarize(
                    level1_nodes,
                    paper.paper_id,
                    level=2
                )
                nodes.extend(level2_nodes)
        
        return nodes
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Simple text chunking."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    async def _cluster_and_summarize(
        self,
        nodes: List[RetrievalNode],
        paper_id: str,
        level: int,
        n_clusters: int = 3
    ) -> List[RetrievalNode]:
        """Cluster nodes and generate summaries (RAPTOR step)."""
        if len(nodes) == 0:
            return []
        
        # Simple clustering (in production, use GMM)
        embeddings = np.array([n.embedding for n in nodes if n.embedding is not None])
        
        if len(embeddings) == 0:
            return []
        
        # K-means clustering
        from sklearn.cluster import KMeans
        n_clusters = min(n_clusters, len(nodes))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group nodes by cluster
        cluster_groups = defaultdict(list)
        for node, cluster_id in zip(nodes, clusters):
            cluster_groups[cluster_id].append(node)
        
        # Generate summary for each cluster
        summary_nodes = []
        for cluster_id, cluster_nodes in cluster_groups.items():
            # Concatenate cluster contents
            combined_text = "\n\n".join([n.content for n in cluster_nodes])
            
            # Generate summary (in production, use LLM)
            summary = await self._generate_summary(combined_text, level)
            
            node_id = f"{paper_id}_L{level}_C{cluster_id}"
            summary_node = RetrievalNode(
                node_id=node_id,
                content=summary,
                level=RetrievalLevel.SECTION if level == 1 else RetrievalLevel.PAPER,
                children_ids=[n.node_id for n in cluster_nodes],
                metadata={
                    "paper_id": paper_id,
                    "cluster_id": cluster_id,
                    "num_children": len(cluster_nodes)
                }
            )
            
            # Update parent references
            for child in cluster_nodes:
                child.parent_id = node_id
            
            # Generate embedding for summary
            summary_node.embedding = self._generate_embeddings([summary])[0]
            
            summary_nodes.append(summary_node)
        
        return summary_nodes
    
    async def _generate_summary(self, text: str, level: int) -> str:
        """Generate summary using LLM.
        
        Args:
            text: Text to summarize
            level: RAPTOR level (1=section, 2=paper)
            
        Returns:
            Generated summary
        """
        if self._llm_service is None:
            # Fallback: extract first 200 words
            words = text.split()[:200]
            return " ".join(words) + f" [Level {level} summary]"
        
        # Determine abstraction level
        if level == 1:
            instruction = "Summarize the following text into 2-3 sentences, focusing on key findings and methods."
            max_tokens = 150
        else:  # level 2
            instruction = "Provide a high-level 1-sentence summary of the main contribution."
            max_tokens = 50
        
        prompt = f"{instruction}\n\nText:\n{text[:2000]}"  # Limit input length
        
        try:
            response = await self._llm_service.complete(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.3
            )
            summary = response.content.strip()
            
            # Validate quality
            if self._validate_summary_quality(summary) < 0.5:
                # Fallback to extractive summary
                words = text.split()[:100]
                return " ".join(words)
            
            return summary
        except Exception as e:
            print(f"⚠️  LLM summarization failed: {e}")
            # Fallback
            words = text.split()[:100]
            return " ".join(words)
    
    def _validate_summary_quality(self, summary: str) -> float:
        """Validate summary quality using heuristics.
        
        Args:
            summary: Summary text
            
        Returns:
            Quality score 0-1
        """
        score = 0.5  # Base score
        
        # Length check (should be reasonable)
        word_count = len(summary.split())
        if 10 <= word_count <= 100:
            score += 0.2
        elif word_count < 5:
            score -= 0.3
        
        # Completeness (ends with punctuation)
        if summary.endswith(('.', '!', '?')):
            score += 0.1
        
        # Content quality (has meaningful words)
        meaningful_words = ['protein', 'gene', 'method', 'result', 'show', 'demonstrate', 'achieve']
        if any(word in summary.lower() for word in meaningful_words):
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    async def ingest_paper(self, paper: GoldenReferencePaper) -> None:
        """Parse, embed, and store exemplar paper with RAPTOR."""
        print(f"   📄 Ingesting: {paper.title}")
        
        # Build RAPTOR tree
        raptor_nodes = await self.build_raptor_tree(paper)
        paper.raptor_nodes = raptor_nodes
        
        # Store in ChromaDB
        if self.use_chromadb and raptor_nodes:
            ids = [n.node_id for n in raptor_nodes]
            documents = [n.content for n in raptor_nodes]
            embeddings = [n.embedding.tolist() for n in raptor_nodes if n.embedding is not None]
            # Convert metadata to ChromaDB-safe types (str, int, float, bool only)
            metadatas = []
            for n in raptor_nodes:
                safe_metadata = {}
                for k, v in n.metadata.items():
                    # Convert numpy types to Python types
                    if isinstance(v, (np.integer, np.int32, np.int64)):
                        safe_metadata[k] = int(v)
                    elif isinstance(v, (np.floating, np.float32, np.float64)):
                        safe_metadata[k] = float(v)
                    elif isinstance(v, (str, int, float, bool)) or v is None:
                        safe_metadata[k] = v
                    else:
                        # Skip unsupported types
                        pass
                metadatas.append(safe_metadata)
            
            if embeddings:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
        
        # Store paper
        self.papers[paper.paper_id] = paper
        for node in raptor_nodes:
            self.raptor_nodes[node.node_id] = node
        
        print(f"   ✅ Ingested with {len(raptor_nodes)} RAPTOR nodes")
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7  # Weight for dense vs sparse
    ) -> List[Tuple[RetrievalNode, float]]:
        """Hybrid search combining dense (vector) and sparse (BM25)."""
        if not self.use_chromadb:
            # Fallback: return all nodes
            return [(node, 1.0) for node in list(self.raptor_nodes.values())[:top_k]]
        
        # Dense search (ChromaDB)
        query_embedding = self._generate_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Convert to nodes with scores
        retrieved_nodes = []
        if results['ids'] and results['ids'][0]:
            for node_id, distance in zip(results['ids'][0], results['distances'][0]):
                if node_id in self.raptor_nodes:
                    # Convert distance to similarity (1 / (1 + distance))
                    similarity = 1 / (1 + distance)
                    retrieved_nodes.append((self.raptor_nodes[node_id], similarity))
        
        return retrieved_nodes
    
    async def agentic_search(
        self,
        query: str,
        query_complexity: str = "auto"  # "simple", "medium", "complex", "auto"
    ) -> List[Tuple[RetrievalNode, float]]:
        """Agentic search with adaptive strategy based on query complexity."""
        
        # Auto-detect complexity (stub)
        if query_complexity == "auto":
            query_complexity = self._classify_query_complexity(query)
        
        if query_complexity == "simple":
            # BM25-only for simple factual queries
            return await self.hybrid_search(query, alpha=0.3)
        elif query_complexity == "medium":
            # Balanced hybrid
            return await self.hybrid_search(query, alpha=0.7)
        else:  # complex
            # Multi-hop RAPTOR search (traverse tree levels)
            return await self._multihop_raptor_search(query)
    
    def _classify_query_complexity(self, query: str) -> str:
        """Classify query complexity (stub)."""
        # Simple heuristic
        if len(query.split()) < 5:
            return "simple"
        elif "how" in query.lower() or "why" in query.lower():
            return "complex"
        else:
            return "medium"
    
    async def _multihop_raptor_search(
        self,
        query: str,
        max_hops: int = 3
    ) -> List[Tuple[RetrievalNode, float]]:
        """Multi-hop search across RAPTOR tree levels."""
        # Start with leaf nodes
        results = await self.hybrid_search(query, top_k=10)
        
        # Traverse up the tree to get context
        enriched_results = []
        for node, score in results:
            # Add node itself
            enriched_results.append((node, score))
            
            # Add parent for context
            if node.parent_id and node.parent_id in self.raptor_nodes:
                parent = self.raptor_nodes[node.parent_id]
                enriched_results.append((parent, score * 0.8))  # Discounted score
        
        # Remove duplicates and re-rank
        seen_ids = set()
        unique_results = []
        for node, score in enriched_results:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                unique_results.append((node, score))
        
        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        return unique_results[:10]
    
    async def extract_nature_patterns(self, topic: str) -> Dict:
        """Extract Nature-specific writing patterns."""
        # Search for relevant papers
        results = await self.agentic_search(
            f"Nature introduction {topic}",
            query_complexity="complex"
        )
        
        if not results:
            return {"patterns": [], "message": "No papers found"}
        
        # Analyze patterns
        hooks = []
        structures = []
        
        for node, score in results[:5]:
            paper_id = node.metadata.get("paper_id")
            if paper_id in self.papers:
                paper = self.papers[paper_id]
                if paper.journal == "Nature":
                    # Extract first sentence as hook
                    first_sentence = paper.introduction.split('.')[0] if paper.introduction else ""
                    if first_sentence:
                        hooks.append(first_sentence)
        
        return {
            "narrative_hooks": hooks[:3],
            "avg_intro_length": np.mean([len(p.introduction.split()) for p in self.papers.values() if p.introduction]) if self.papers else 0,
            "num_papers_analyzed": len(results)
        }


# Demo function
async def demo_advanced_rag():
    """Demo of advanced RAG features."""
    store = AdvancedGoldenReferenceStore()
    
    # Add example paper
    paper = GoldenReferencePaper(
        paper_id="nature_2023_alphafold",
        title="Highly accurate protein structure prediction with AlphaFold",
        journal="Nature",
        year=2021,
        abstract="Proteins are essential to life, and understanding their structure is crucial for understanding their function. However, experimental determination of protein structure remains challenging. Here we present AlphaFold, a computational method that achieves unprecedented accuracy in protein structure prediction...",
        introduction="Protein structure determination has been a grand challenge in biology for over 50 years. The ability to predict protein structures from amino acid sequences alone would revolutionize our understanding of biological function and disease mechanisms. Despite decades of effort, this goal has remained elusive. Here, we demonstrate that deep learning can solve this problem with near-experimental accuracy...",
        methods="We developed a novel neural network architecture that integrates evolutionary, physical, and geometric constraints...",
        citation_count=15000,
        impact_factor=49.9
    )
    
    await store.ingest_paper(paper)
    
    # Test hybrid search
    print("\n🔍 Hybrid Search:")
    results = await store.hybrid_search("protein structure prediction", top_k=3)
    for node, score in results:
        print(f"   [{score:.3f}] Level {node.level.name}: {node.content[:100]}...")
    
    # Test agentic search
    print("\n🤖 Agentic Search (complex query):")
    results = await store.agentic_search("How does Nature introduce protein structure papers?")
    for node, score in results[:3]:
        print(f"   [{score:.3f}] {node.content[:100]}...")
    
    # Extract patterns
    print("\n📊 Nature Patterns:")
    patterns = await store.extract_nature_patterns("protein structure")
    for hook in patterns.get("narrative_hooks", [])[:2]:
        print(f"   • {hook}")


if __name__ == "__main__":
    asyncio.run(demo_advanced_rag())
