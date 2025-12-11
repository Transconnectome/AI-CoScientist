# FM-RL RAG Integration Guide

## 📋 Overview

This guide provides instructions for integrating FM-RL golden references with AI-CoScientist's RAG systems. After careful analysis, we've selected the **Adaptive Hybrid Retriever + Advanced Golden Reference** combination as the optimal solution for general brain foundation model research.

## 🎯 **UPDATED** RAG System Configuration

### Primary: Adaptive Hybrid Retriever (`src/services/rag/adaptive_hybrid_retriever.py`) ⭐️⭐️⭐️⭐️⭐️
- **Specialization**: General AI/ML research with dynamic optimization
- **Capabilities**:
  - Dynamic alpha tuning (vector/keyword balance)
  - Query-specific optimization
  - Performance-aware strategy switching
  - No domain restrictions (perfect for brain foundation models)
- **Why Better**: Unlike DD-RAPTOR (발달장애 전용), this supports general brain/AI research

### Secondary: Advanced Golden Reference (`src/services/rag/advanced_golden_reference.py`) ⭐️⭐️⭐️
- **Specialization**: High-quality paper management and retrieval
- **Capabilities**: RAPTOR hierarchical indexing, hybrid retrieval (dense + sparse)
- **Use Case**: Foundation Model and RL literature storage

### **Note**: Enhanced DD-RAPTOR was excluded because it's specialized only for developmental disorders (ASD, ADHD), not general brain foundation model research.

## 🔧 Integration Steps

### Step 1: Collection Setup in ChromaDB

```python
# Add to ChromaDB initialization
collections_config = {
    "fm_rl_golden_references": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "distance_metric": "cosine",
        "description": "Foundation Model + RL research papers",
        "metadata_schema": {
            "domain": "str",
            "relevance_score": "float",
            "journal": "str",
            "year": "int"
        }
    }
}
```

### Step 2: Adaptive Hybrid Retriever Configuration

Configure the Adaptive Hybrid Retriever for FM-RL research:

```python
# In adaptive_hybrid_retriever.py - configure for FM-RL
fm_rl_retrieval_config = {
    "strategy": RetrievalStrategy.ADAPTIVE_DYNAMIC,
    "alpha": 0.7,  # Favor semantic search for complex AI concepts
    "k_value": 15,  # Higher k for comprehensive research
    "rerank_enabled": True,
    "expand_query": True,  # Important for technical terminology
    "semantic_threshold": 0.75,
    "keyword_threshold": 0.6,
    "domain_keywords": [
        "foundation model", "reinforcement learning",
        "brain", "neural architecture", "adaptive inference",
        "multimodal", "neuroplasticity", "optimization",
        "transformer", "attention mechanism", "fine-tuning"
    ]
}
```

### Step 3: Golden Reference Integration

Configure the Advanced Golden Reference system to handle FM-RL papers:

```python
# Configuration for FM-RL golden references
fm_rl_config = {
    "collection_name": "fm_rl_golden_references",
    "raptor_levels": 3,  # Chunk, Section, Paper
    "embedding_models": {
        "dense": "sentence-transformers/all-MiniLM-L6-v2",
        "sparse": "bm25"
    },
    "retrieval_strategy": "hybrid",
    "alpha": 0.7  # Dense/sparse balance
}
```

### Step 4: Data Ingestion

Load the FM-RL golden references into the system:

```python
import json
from src.services.rag.advanced_golden_reference import AdvancedGoldenReferenceStore

# Load golden references
with open('data/FM-RL/golden_references.json') as f:
    references = json.load(f)

# Initialize store
store = AdvancedGoldenReferenceStore(
    collection_name="fm_rl_golden_references"
)

# Ingest papers by category
for category, papers in references['categories'].items():
    for paper in papers['papers']:
        await store.ingest_paper(paper)
```

## 📚 Golden References Overview

Our curated collection includes **12 high-impact papers** across 6 categories:

### 1. Brain Foundation Models (2 papers)
- Foundation models for brain activity prediction
- Large-scale pre-training for brain signal analysis

### 2. Reinforcement Learning for Neural Networks (2 papers)
- Neural architecture search via RL
- Adaptive neural networks with RL

### 3. Multimodal Brain Processing (2 papers)
- Multimodal brain data integration
- Cross-modal attention for brain signals

### 4. Adaptive Neural Architectures (2 papers)
- Dynamic neural networks
- Neuroplasticity-inspired deep learning

### 5. Neuroscience-AI Integration (2 papers)
- Brain-inspired AI principles
- Cognitive architectures from neuroscience

### 6. Optimization Algorithms (2 papers)
- RL for hyperparameter optimization
- Adaptive gradient methods with RL

## 🔍 Query Routing Strategy

Configure the Unified RAG Orchestrator to route FM-RL queries appropriately:

```python
# Add FM-RL routing rules
fm_rl_routing = {
    "domain": "FM_RL",
    "keywords": [
        "foundation model", "reinforcement learning",
        "brain", "neural architecture", "adaptive"
    ],
    "primary_strategy": "ENHANCED_DD_RAPTOR",
    "fallback_strategy": "GOLDEN_REFERENCE",
    "complexity_threshold": "MEDIUM"
}
```

## 🧪 Testing and Validation

### Sample Queries for Testing

1. **Foundation Model Query**:
   ```
   "What are the latest advances in foundation models for brain signal analysis?"
   ```

2. **RL Integration Query**:
   ```
   "How can reinforcement learning optimize neural architecture for real-time brain data processing?"
   ```

3. **Multimodal Query**:
   ```
   "What methods exist for integrating fMRI and EEG data in foundation models?"
   ```

4. **Adaptive Architecture Query**:
   ```
   "How do neuroplasticity principles inform adaptive neural network design?"
   ```

### Expected Performance Metrics

- **Retrieval Accuracy**: >90% for FM-RL domain queries
- **Response Latency**: <2 seconds for standard queries
- **Relevance Score**: >0.8 for top-3 results
- **Multimodal Support**: Full integration of brain imaging references

## 🔄 Continuous Improvement

### Feedback Loop Implementation

```python
# Add feedback mechanism
class FMRLFeedbackCollector:
    def collect_query_feedback(self, query, results, user_rating):
        """Collect feedback for continuous improvement"""
        feedback = {
            "query": query,
            "results": results,
            "rating": user_rating,
            "timestamp": datetime.now(),
            "domain": "FM_RL"
        }
        # Store feedback for model improvement
```

### Performance Monitoring

- Track query success rates by category
- Monitor retrieval latency and quality
- Identify knowledge gaps for future reference addition

## 📊 Integration Verification

Run this verification script to ensure proper integration:

```python
# Verification script
async def verify_fm_rl_integration():
    """Verify FM-RL RAG integration is working correctly"""

    # Test Enhanced DD-RAPTOR
    dd_raptor = EnhancedDDRAPTOR()
    result1 = await dd_raptor.search("foundation model brain analysis")

    # Test Golden Reference Store
    golden_store = AdvancedGoldenReferenceStore("fm_rl_golden_references")
    result2 = await golden_store.retrieve("reinforcement learning neural architecture")

    # Test Unified Orchestrator routing
    orchestrator = UnifiedRAGOrchestrator()
    result3 = await orchestrator.query("adaptive brain foundation model RL")

    print("✅ FM-RL RAG integration verified successfully")
    return result1, result2, result3
```

## 🎯 Next Steps

1. **Deploy Configuration**: Apply the configuration changes to the RAG systems
2. **Ingest References**: Load the golden references into ChromaDB
3. **Test Integration**: Run verification scripts
4. **Monitor Performance**: Set up metrics collection
5. **Iterative Improvement**: Add more references based on usage patterns

## 📞 Support

For integration assistance or issues:
- Review the logs in `src/services/rag/` modules
- Check ChromaDB collection status
- Verify embedding model availability
- Consult the main AI-CoScientist documentation

---

*This integration guide ensures optimal RAG performance for FM-RL research with seamless access to curated high-quality references.*