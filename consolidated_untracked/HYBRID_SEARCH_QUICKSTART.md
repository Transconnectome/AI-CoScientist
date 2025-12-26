# Hybrid DD Search - Quick Start Guide

**5-Minute Quick Start for the Hybrid DD Search System**

---

## Installation

```bash
# Ensure you have the required packages
pip install chromadb sentence-transformers torch numpy
```

---

## Basic Usage

### 1. Simple Search

```python
from src.services.rag.hybrid_dd_search import HybridDDSearch

# Initialize (one-time setup)
search = HybridDDSearch()

# Search
response = search.search("autism diagnosis using EEG", top_k=5)

# View results
for i, result in enumerate(response.results, 1):
    print(f"{i}. [{result.source}] Score: {result.combined_score:.4f}")
    print(f"   {result.document[:200]}...\n")
```

**Output**:
```
1. [DD] Score: 4.3086
   The references section highlights various studies related to autism
   spectrum disorder (ASD), focusing on machine learning...

2. [FM] Score: 0.3975
   developed a lightweight, locally-executable LLM (EEG Emotion Copilot),
   optimized through model pruning and fine-tuning...
```

---

## Common Use Cases

### Clinical Research Query

```python
response = search.search("ADHD treatment effectiveness", top_k=10)
print(f"Found {response.dd_count} DD papers, {response.fm_count} FM papers")
```

### Technical Architecture Query

```python
response = search.search("transformer models for brain imaging", top_k=10)
print(f"Classification: {response.query_classification.query_type.value}")
```

### Layer-Specific Search

```python
# Search only high-level summaries (faster)
response = search.search("autism connectivity", top_k=5, layers=['L2'])
```

---

## Export Results

```python
# Save to JSON
search.export_results(response, "results.json")

# Load and analyze
import json
with open("results.json") as f:
    data = json.load(f)
    print(f"Total results: {data['statistics']['total_results']}")
```

---

## Performance Tips

1. **Fast Overview**: Use `layers=['L2']` for summaries only
2. **Detailed Search**: Use `layers=['L0']` for chunk-level search
3. **Disable Reranking**: Add `config={"use_reranking": False}` for speed

---

## Running Tests

```bash
# Run comprehensive benchmark
python tests/rag/test_hybrid_search.py

# Run demo
python scripts/demo_hybrid_dd_search.py
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Query Classification** | Automatic clinical/technical/mixed detection |
| **Dual Database** | 26 DD papers + 13 FM papers |
| **Multi-Layer** | L0 (chunks), L1 (summaries), L2 (top-level) |
| **Adaptive Weights** | Auto-adjusts DD/FM prioritization |
| **Fast** | <1 second average response time |

---

## Sample Queries

```python
# Clinical queries (prioritize DD papers)
search.search("autism spectrum disorder diagnosis")
search.search("ADHD treatment in children")
search.search("behavioral therapy for ASD")

# Technical queries (prioritize FM papers)
search.search("transformer architecture training")
search.search("self-supervised learning methods")
search.search("attention mechanisms")

# Mixed queries (balanced)
search.search("AI models for autism diagnosis")
search.search("deep learning for EEG analysis")
search.search("foundation models for neuroscience")
```

---

## Understanding Results

```python
response = search.search("your query", top_k=5)

# Query classification
print(response.query_classification.query_type.value)  # clinical/technical/mixed
print(response.query_classification.reasoning)

# Results
for result in response.results:
    print(f"Source: {result.source}")  # DD or FM
    print(f"Layer: {result.level}")    # L0, L1, or L2
    print(f"Score: {result.combined_score}")
    print(f"Reasoning: {result.reasoning}")

# Performance
print(f"Total time: {response.total_time_ms}ms")
print(f"DD results: {response.dd_count}")
print(f"FM results: {response.fm_count}")
```

---

## Troubleshooting

**Q: Slow performance?**
```python
# Use fewer results
response = search.search("query", top_k=5)

# Search fewer layers
response = search.search("query", layers=['L2'])

# Disable reranking
config = {"use_reranking": False}
search = HybridDDSearch(config=config)
```

**Q: Need more DD or FM results?**
```python
# Disable classification to use equal weights
response = search.search("query", enable_classification=False)
```

---

## Next Steps

- Read full documentation: `HYBRID_DD_SEARCH_README.md`
- View implementation details: `HYBRID_SEARCH_IMPLEMENTATION_REPORT.md`
- Run comprehensive demo: `python scripts/demo_hybrid_dd_search.py`
- Explore code: `src/services/rag/hybrid_dd_search.py`

---

**Quick Start Complete!** You're ready to use the Hybrid DD Search system.

For questions or issues, see the full documentation or run the demo script.
