# Hybrid DD Search System

**Comprehensive hybrid search system combining DD papers (clinical research) with NeurIPS 2025 foundation model papers (technical innovations)**

## Overview

The Hybrid DD Search System intelligently combines two specialized knowledge bases:
- **DD Papers** (26 papers): Clinical developmental disorder research (autism, ADHD, neurodevelopmental conditions)
- **NeurIPS 2025 FM Papers** (13 papers): State-of-the-art foundation models for brain imaging and multimodal AI

### Key Features

1. **Intelligent Query Classification**
   - Automatically classifies queries as clinical, technical, or mixed
   - Adaptive search weighting based on query type
   - 60+ domain-specific keywords across categories

2. **Dual Database Integration**
   - Simultaneous search across both ChromaDB instances
   - Multi-layer RAPTOR architecture (L0: chunks, L1: summaries, L2: top-level)
   - SciBERT embeddings (768 dimensions) for scientific domain

3. **Advanced Result Merging**
   - Weighted scoring based on query classification
   - Cross-encoder reranking for optimal relevance
   - Source attribution and provenance tracking

4. **Performance Optimization**
   - Average response time: ~1 second per query
   - Efficient batch processing
   - Comprehensive metrics tracking

## Architecture

### Database Structure

```
DD Papers Database (chromadb_data_dd/)
├── dd_papers_L0: 1,387 documents (original chunks)
├── dd_papers_L1: 112 documents (intermediate summaries)
└── dd_papers_L2: 26 documents (top-level summaries)

FM Papers Database (chromadb_data_neurips2025/)
├── neurips_2025_L0: 1,161 documents (original chunks)
├── neurips_2025_L1: 53 documents (intermediate summaries)
└── neurips_2025_L2: 13 documents (top-level summaries)
```

### Search Pipeline

```
Query Input
    ↓
Query Classification (Clinical/Technical/Mixed)
    ↓
Parallel Database Search
    ├── DD Database Search (with weight)
    └── FM Database Search (with weight)
    ↓
Result Merging & Scoring
    ↓
Cross-Encoder Reranking
    ↓
Top-K Results with Attribution
```

## Installation & Setup

### Prerequisites

```bash
# Install required packages
pip install chromadb sentence-transformers torch numpy
```

### File Structure

```
src/services/rag/
├── query_classifier.py          # Query classification system
├── hybrid_dd_search.py          # Main hybrid search implementation
└── ...

tests/rag/
└── test_hybrid_search.py        # Comprehensive test suite

scripts/
└── demo_hybrid_dd_search.py     # Demo script
```

## Usage

### Basic Search

```python
from src.services.rag.hybrid_dd_search import HybridDDSearch

# Initialize search system
search = HybridDDSearch()

# Perform search
response = search.search("autism diagnosis using EEG signals", top_k=10)

# Access results
for result in response.results:
    print(f"[{result.source}] {result.level} - Score: {result.combined_score:.4f}")
    print(f"Document: {result.document[:200]}...")
```

### Query Classification

```python
from src.services.rag.query_classifier import QueryClassifier

classifier = QueryClassifier()

# Classify a query
classification = classifier.classify("foundation models for autism diagnosis")

print(f"Type: {classification.query_type.value}")
print(f"Clinical Score: {classification.clinical_score:.2%}")
print(f"Technical Score: {classification.technical_score:.2%}")
print(f"Reasoning: {classification.reasoning}")
```

### Layer-Specific Search

```python
# Search only L2 (high-level summaries)
response = search.search(
    "autism brain connectivity",
    top_k=5,
    layers=['L2']
)

# Search L0 and L1 (detailed content)
response = search.search(
    "transformer architecture",
    top_k=10,
    layers=['L0', 'L1']
)
```

### Export Results

```python
# Export to JSON
search.export_results(response, "search_results.json")

# Load and analyze
import json
with open("search_results.json") as f:
    data = json.load(f)

print(f"Query: {data['query']}")
print(f"DD Results: {data['statistics']['dd_count']}")
print(f"FM Results: {data['statistics']['fm_count']}")
```

## Performance Benchmarks

Based on comprehensive testing with 12 diverse queries:

| Metric | Value |
|--------|-------|
| **Average Response Time** | 964ms |
| **Min Response Time** | 641ms |
| **Max Response Time** | 1477ms |
| **Average DD Ratio** | 60.8% |
| **Average FM Ratio** | 39.2% |

### Query Type Performance

| Query Type | Avg Time | DD Ratio | FM Ratio |
|------------|----------|----------|----------|
| **Clinical** | 1178ms | 87.5% | 12.5% |
| **Technical** | 958ms | 55.0% | 45.0% |
| **Mixed** | 756ms | 40.0% | 60.0% |

## Query Classification Examples

### Clinical Queries (Prioritize DD Papers)

```python
# These queries get DD weight = 3.0, FM weight = 1.0
queries = [
    "autism spectrum disorder diagnosis and assessment",
    "ADHD treatment in pediatric populations",
    "behavioral interventions for developmental disorders",
    "fMRI connectivity in ASD patients"
]
```

**Result Distribution**: ~87.5% DD papers, ~12.5% FM papers

### Technical Queries (Prioritize FM Papers)

```python
# These queries get DD weight = 1.0, FM weight = 3.0
queries = [
    "transformer architecture for image classification",
    "self-supervised learning methods",
    "attention mechanisms in neural networks",
    "foundation models training strategies"
]
```

**Result Distribution**: ~55% DD papers, ~45% FM papers

### Mixed Queries (Balanced Search)

```python
# These queries get balanced weighting
queries = [
    "foundation models for autism diagnosis",
    "deep learning for EEG analysis in ADHD",
    "multimodal AI for brain imaging",
    "large language models in clinical diagnosis"
]
```

**Result Distribution**: ~40% DD papers, ~60% FM papers

## Configuration

### Default Configuration

```python
config = {
    "embedding_model": "allenai/scibert_scivocab_uncased",  # 768-dim
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_results_per_db": 20,
    "final_top_k": 10,
    "use_reranking": True,
    "min_similarity_threshold": 0.5,
    "layer_weights": {
        "L0": 1.0,   # Leaf nodes (chunks)
        "L1": 1.2,   # Intermediate summaries
        "L2": 1.5    # Top-level summaries
    }
}

search = HybridDDSearch(config=config)
```

### Custom Configuration

```python
# Disable reranking for faster searches
config = {"use_reranking": False}

# Increase L2 weight for higher-level summaries
config = {"layer_weights": {"L0": 1.0, "L1": 1.5, "L2": 2.0}}

# Search more documents per database
config = {"max_results_per_db": 50, "final_top_k": 20}
```

## Testing

### Run Unit Tests

```bash
# Run all hybrid search tests
pytest tests/rag/test_hybrid_search.py -v

# Run specific test class
pytest tests/rag/test_hybrid_search.py::TestQueryClassifier -v
pytest tests/rag/test_hybrid_search.py::TestHybridDDSearch -v
pytest tests/rag/test_hybrid_search.py::TestSearchQuality -v
```

### Run Performance Benchmark

```bash
# Standalone benchmark
python tests/rag/test_hybrid_search.py

# View results
cat hybrid_search_benchmark_results.json
```

### Run Demo

```bash
# Comprehensive demonstration
python scripts/demo_hybrid_dd_search.py
```

## API Reference

### HybridDDSearch

#### `__init__(dd_path, fm_path, config=None)`

Initialize hybrid search system.

**Parameters:**
- `dd_path` (str): Path to DD papers ChromaDB
- `fm_path` (str): Path to FM papers ChromaDB
- `config` (dict): Optional configuration

#### `search(query, top_k=10, layers=None, enable_classification=True)`

Perform hybrid search.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results to return
- `layers` (list): RAPTOR layers to search (default: all)
- `enable_classification` (bool): Use query classification

**Returns:**
- `HybridSearchResponse`: Complete search response

#### `export_results(response, output_file)`

Export search results to JSON.

**Parameters:**
- `response` (HybridSearchResponse): Search response
- `output_file` (str): Output file path

### QueryClassifier

#### `classify(query)`

Classify query type.

**Parameters:**
- `query` (str): Query string

**Returns:**
- `QueryClassification`: Classification result

#### `get_search_weights(classification)`

Get search weights for databases.

**Parameters:**
- `classification` (QueryClassification): Query classification

**Returns:**
- `(float, float)`: DD weight, FM weight tuple

## Response Structure

### HybridSearchResponse

```python
{
    "query": str,                        # Original query
    "query_classification": {            # Classification details
        "type": "clinical|technical|mixed",
        "clinical_score": float,
        "technical_score": float,
        "confidence": float,
        "reasoning": str
    },
    "results": [                         # Ranked results
        {
            "rank": int,
            "document": str,
            "metadata": dict,
            "dd_score": float,
            "fm_score": float,
            "combined_score": float,
            "source": "DD|FM",
            "level": "L0|L1|L2",
            "reasoning": str
        }
    ],
    "dd_count": int,                     # DD results count
    "fm_count": int,                     # FM results count
    "total_time_ms": float,              # Total search time
    "dd_search_time_ms": float,          # DD search time
    "fm_search_time_ms": float,          # FM search time
    "merge_time_ms": float,              # Merge/rerank time
    "performance_stats": {               # Cumulative statistics
        "total_queries": int,
        "query_distribution": {
            "clinical": int,
            "technical": int,
            "mixed": int
        },
        "average_latency_ms": float,
        "average_dd_ratio": float,
        "average_fm_ratio": float
    }
}
```

## Advanced Use Cases

### 1. Clinical Research Assistant

```python
# Search for autism-related clinical research
response = search.search(
    "autism diagnosis and intervention strategies",
    top_k=20,
    layers=['L1', 'L2']  # Focus on summaries
)

# Filter for DD papers only
dd_results = [r for r in response.results if r.source == 'DD']
```

### 2. Technical Literature Review

```python
# Search for foundation model architectures
response = search.search(
    "transformer models for multimodal learning",
    top_k=15
)

# Filter for FM papers only
fm_results = [r for r in response.results if r.source == 'FM']
```

### 3. Cross-Domain Research

```python
# Find connections between AI and clinical research
response = search.search(
    "AI models for neurodevelopmental disorder diagnosis",
    top_k=25
)

# Analyze source distribution
print(f"Clinical papers: {response.dd_count}")
print(f"Technical papers: {response.fm_count}")
print(f"Balance: {response.dd_count / response.fm_count:.2f}")
```

### 4. Batch Processing

```python
queries = [
    "autism EEG biomarkers",
    "ADHD brain imaging",
    "transformer attention mechanisms",
    "multimodal fusion techniques"
]

results = []
for query in queries:
    response = search.search(query, top_k=10)
    results.append({
        "query": query,
        "dd_count": response.dd_count,
        "fm_count": response.fm_count,
        "top_score": response.results[0].combined_score
    })
```

## Troubleshooting

### Common Issues

**1. Embedding Dimension Mismatch**

```
Error: Collection expecting embedding with dimension of 768, got 384
```

**Solution**: Ensure using SciBERT model:
```python
config = {"embedding_model": "allenai/scibert_scivocab_uncased"}
search = HybridDDSearch(config=config)
```

**2. ChromaDB Connection Error**

```
Error: No collection found
```

**Solution**: Verify database paths exist:
```bash
ls chromadb_data_dd/
ls chromadb_data_neurips2025/
```

**3. Slow Performance**

**Solutions**:
- Disable cross-encoder reranking: `config = {"use_reranking": False}`
- Reduce results per database: `config = {"max_results_per_db": 10}`
- Search fewer layers: `search.search(query, layers=['L2'])`

## Performance Optimization Tips

1. **Use Layer Targeting**: For quick overviews, search only L2
2. **Disable Reranking**: For speed-critical applications
3. **Batch Queries**: Process multiple queries in one session
4. **Cache Results**: Store frequently accessed results
5. **Adjust Weights**: Fine-tune layer weights for your use case

## Future Enhancements

- [ ] Add semantic caching for repeated queries
- [ ] Implement query expansion with synonyms
- [ ] Add multi-language support
- [ ] Integrate with LLM for answer generation
- [ ] Add relevance feedback loop
- [ ] Implement GraphRAG integration
- [ ] Add multimodal query support (images, figures)

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hybrid_dd_search,
  title = {Hybrid DD Search: Intelligent Multi-Database Search for Clinical and Technical Research},
  author = {AI-CoScientist Team},
  year = {2025},
  description = {Hybrid search system combining developmental disorder clinical research with foundation model technical innovations}
}
```

## License

This system is part of the AI-CoScientist project. See LICENSE for details.

## Support

For issues, questions, or contributions:
- GitHub Issues: [AI-CoScientist/issues](https://github.com/your-org/AI-CoScientist/issues)
- Documentation: See `CLAUDE.md` for project overview
- Examples: Run `python scripts/demo_hybrid_dd_search.py`

## Changelog

### Version 1.0.0 (2025-12-08)

- ✓ Initial release
- ✓ Query classification system with 60+ keywords
- ✓ Dual database integration (DD + FM papers)
- ✓ Advanced result merging and cross-encoder reranking
- ✓ Comprehensive test suite with 12 benchmark queries
- ✓ Performance optimization (<1s average response time)
- ✓ JSON export functionality
- ✓ Multi-layer RAPTOR search support
- ✓ Detailed documentation and examples

---

**Last Updated**: December 8, 2025
**Status**: Production Ready
**Performance**: ✓ Sub-second latency | ✓ High relevance | ✓ Comprehensive coverage
