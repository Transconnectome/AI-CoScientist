# RAG Integration Guide

AI-CoScientist now includes RAG (Retrieval-Augmented Generation) capabilities that learn from past successful improvements!

## 🎯 What's New?

### Learning from Experience
- **Stores successful patterns**: Every improvement is saved to ChromaDB
- **Retrieves similar cases**: Finds relevant past improvements
- **Context-aware suggestions**: Claude receives examples of what worked before
- **Continuous improvement**: Gets better with each use

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Paper Improvement with RAG                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Analyze Current Paper                                │
│     ├─ Overall: 7.94/10                                  │
│     └─ Problem: Low clarity (7.45)                       │
│                                                          │
│  2. Search ChromaDB  [NEW!]                              │
│     ├─ Query: "improve abstract clarity, score 7.45"    │
│     ├─ Found: 3 similar cases                            │
│     └─ Extract: Successful strategies                    │
│                                                          │
│  3. Enhance Claude Prompt  [NEW!]                        │
│     ├─ Base requirements                                 │
│     ├─ + RAG context (past successes)                    │
│     └─ Generate improvement                              │
│                                                          │
│  4. Store Success  [NEW!]                                │
│     ├─ Improved score: 8.32/10                           │
│     └─ Save pattern to ChromaDB                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install dependencies
pip install chromadb openai

# 2. Set API keys
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"  # For embeddings
```

### Option 1: Auto Mode (Recommended)

```bash
# RAG will automatically start ChromaDB if needed
python scripts/apply_improvements_with_rag.py
```

**What happens:**
1. ✅ Tries to connect to ChromaDB (localhost:8001)
2. 🐳 If not found, starts Docker container
3. 💾 If Docker fails, uses local persistent storage
4. ⚠️  If all fail, continues without RAG (graceful degradation)

### Option 2: Manual ChromaDB Setup

```bash
# Start ChromaDB manually
docker run -d --name ai-coscientist-chromadb -p 8001:8000 chromadb/chroma

# Run improvement
python scripts/apply_improvements_with_rag.py
```

### Option 3: Local Persistent (No Docker)

Set environment variable:
```bash
export CHROMADB_MODE=local
python scripts/apply_improvements_with_rag.py
```

## 📊 RAG Storage Schema

### Collection: improvement_patterns

Each stored improvement includes:

```yaml
Document: Improved text (Abstract, Introduction, etc.)

Metadata:
  # Identification
  improvement_id: "uuid-1234"
  paper_id: "paper_mbbn"
  section: "Abstract"
  timestamp: "2025-10-12T10:30:00"

  # Scores Before/After
  before_overall: 7.94
  after_overall: 8.32
  before_clarity: 7.45
  after_clarity: 7.43

  # Improvement Metrics
  overall_gain: 0.39
  clarity_gain: -0.02

  # Strategy
  strategy_applied: "4-sentence structure"
  field: "neuroscience"
  paper_type: "methodology"
```

### Search Example

When improving a new paper:

```python
# Query
"improve abstract clarity, current score 7.45"

# Returns (Top 3 similar cases)
[
  {
    "text": "[Improved abstract text]",
    "metadata": {
      "before_overall": 7.50,
      "after_overall": 8.30,
      "strategy_applied": "restructure with 4-sentence format",
      ...
    }
  },
  ...
]
```

## 📈 Expected Performance

### Cold Start (First Use)
- **Patterns stored**: 0
- **Performance**: Same as standard mode
- **Improvement**: ~0.3-0.5 points

### After 3-5 Papers
- **Patterns stored**: 9-15
- **Performance**: Starting to learn
- **Improvement**: ~0.4-0.6 points

### After 10+ Papers
- **Patterns stored**: 30+
- **Performance**: Domain expertise
- **Improvement**: ~0.5-0.8 points

### Long-term (50+ Papers)
- **Patterns stored**: 150+
- **Performance**: Highly optimized
- **Improvement**: ~0.6-1.0 points

## 🎨 Example Output

```
============================================================
🤖 AI-CoScientist Autonomous Improvement with RAG
============================================================

📄 Loading paper: paper_mbbn_original.txt
   Length: 174,241 characters

🔧 Initializing services...
   ✅ Ensemble Scorer ready
   🐳 Starting ChromaDB Docker container...
   ✅ ChromaDB Docker container started
   ✅ Connected to ChromaDB at localhost:8001
   ✅ RAG System ready (http mode)
   📚 Stored patterns: 15

📊 Baseline evaluation...
   Overall: 7.94/10
   Clarity: 7.45

🎯 Target sections: Abstract, Introduction, Methods
   Strategy: RAG-enhanced improvement (learning from past successes)

============================================================
🔄 ITERATION 1/3
============================================================

🔧 Improving Abstract with RAG...
  📏 Original length: 1835 chars
  🔍 Searching RAG for similar Abstract improvements...
  ✅ Found 3 similar improvement patterns

  📚 SIMILAR SUCCESSFUL IMPROVEMENTS:

  **Example 1:**
  Strategy: 4-sentence structure
  Score improvement: 7.50 → 8.25 (+0.75)
    - Clarity: 7.40 → 8.10

  **Example 2:**
  Strategy: quantify results in abstract
  Score improvement: 7.60 → 8.30 (+0.70)
    - Significance: 7.30 → 8.20

  **Example 3:**
  Strategy: concrete methodology description
  Score improvement: 7.55 → 8.15 (+0.60)
    - Clarity: 7.35 → 7.95

  ✅ Improved length: 1620 chars

💾 Storing successful patterns in RAG...
   ✅ Patterns stored for future use

📈 SCORES (Iteration 1):
   Overall: 8.35/10 (Δ +0.41)  ⬆️ Better than before!
   Clarity: 7.80 (Δ +0.35)

============================================================
✅ IMPROVEMENT COMPLETE
============================================================

📊 FINAL RESULTS:
   Iterations: 1
   Starting score: 7.94/10
   Final score: 8.35/10
   Total improvement: +0.41

📚 RAG STATISTICS:
   Total patterns stored: 18
   These patterns will help improve future papers!
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional
CHROMADB_MODE=auto     # auto, docker, local, disabled
CHROMADB_HOST=localhost
CHROMADB_PORT=8001
CHROMADB_PATH=./chromadb_data
```

### Programmatic Configuration

```python
from src.services.embeddings import EmbeddingService
from src.services.rag import RAGManager

# Initialize
embedding_service = EmbeddingService(provider="openai")
rag_manager = RAGManager(
    embedding_service=embedding_service,
    chromadb_mode="auto",  # Try Docker, fallback to local
    chromadb_host="localhost",
    chromadb_port=8001
)

# Check status
if rag_manager.is_enabled():
    stats = rag_manager.get_statistics()
    print(f"Patterns stored: {stats['total_patterns']}")
```

## 🎯 Use Cases

### 1. Consistent Paper Improvement
```bash
# Improve multiple papers in same domain
for paper in papers/*.pdf; do
    python scripts/apply_improvements_with_rag.py $paper
done
# Each iteration learns from previous successes!
```

### 2. Domain-Specific Learning
```python
# Store domain expertise
await rag_manager.store_improvement_pattern(
    paper_id="neuro_paper_01",
    section="Abstract",
    field="neuroscience",  # Domain tag
    ...
)

# Future neuroscience papers benefit automatically
```

### 3. Team Knowledge Sharing
```bash
# Shared ChromaDB server
export CHROMADB_HOST=team-chromadb.company.com
export CHROMADB_PORT=8001

# Entire team learns from each other's improvements
python scripts/apply_improvements_with_rag.py
```

## 🐛 Troubleshooting

### Issue: "ChromaDB not available"

**Solution 1**: Check Docker
```bash
docker ps | grep chroma
# If not running, start it
docker run -d --name ai-coscientist-chromadb -p 8001:8000 chromadb/chroma
```

**Solution 2**: Use local mode
```bash
export CHROMADB_MODE=local
python scripts/apply_improvements_with_rag.py
```

**Solution 3**: Disable RAG
```bash
export CHROMADB_MODE=disabled
# Falls back to standard improvement
python scripts/apply_improvements_autonomous.py
```

### Issue: "OPENAI_API_KEY not found"

RAG requires OpenAI for embeddings:
```bash
export OPENAI_API_KEY=sk-your-key
```

Alternative: Use standard mode without RAG:
```bash
python scripts/apply_improvements_autonomous.py
```

### Issue: Low improvement on first run

**Expected!** RAG needs data:
- **First run**: No patterns, standard performance
- **2-3 runs**: Starting to learn
- **5+ runs**: Noticeable improvement
- **10+ runs**: Significant benefit

## 📊 Comparison

| Feature | Standard Mode | RAG Mode |
|---------|--------------|----------|
| **First Run** | 0.3-0.5 improvement | 0.3-0.5 improvement (same) |
| **After 5 runs** | 0.3-0.5 improvement | 0.4-0.6 improvement ⬆️ |
| **After 10 runs** | 0.3-0.5 improvement | 0.5-0.8 improvement ⬆️⬆️ |
| **Learning** | None | Continuous |
| **Setup** | None | ChromaDB + OpenAI |
| **Cost** | Claude API only | Claude + OpenAI embeddings (~$0.0001/paper) |

## 🎓 Advanced Usage

### Custom Search Strategies

```python
# Search for specific score improvements
results = await rag_manager.search_similar_improvements(
    section="Abstract",
    problem_description="unclear methodology + low novelty",
    current_scores={"clarity": 7.2, "novelty": 7.1},
    n_results=5
)

# Filter by score gains
high_impact = [r for r in results
               if r['metadata']['overall_gain'] > 0.5]
```

### Bulk Pattern Import

```python
# Import patterns from successful papers
successful_papers = [
    {"before": 7.5, "after": 8.8, "text": "...", "strategy": "..."},
    {"before": 7.2, "after": 8.5, "text": "...", "strategy": "..."},
    ...
]

for paper in successful_papers:
    await rag_manager.store_improvement_pattern(...)
```

### Analytics

```python
# Get RAG statistics
stats = rag_manager.get_statistics()

print(f"Total patterns: {stats['total_patterns']}")
print(f"Client type: {stats['client_type']}")
print(f"Embedding model: {stats['embedding_model']}")

# Query collection directly
collection = rag_manager.client.get_collection("improvement_patterns")
print(f"Collection count: {collection.count()}")
```

## 📚 Architecture

```
┌──────────────────────────────────────────────────────────┐
│  RAG-Enhanced Paper Improvement Architecture             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐      ┌──────────────┐                  │
│  │   Paper     │──────▶│   Ensemble   │                  │
│  │   Input     │      │   Scorer     │                  │
│  └─────────────┘      └──────┬───────┘                  │
│                              │                           │
│                         Scores (7.94)                    │
│                              │                           │
│                              ▼                           │
│                    ┌─────────────────┐                  │
│                    │   RAG Manager   │                  │
│                    └────────┬────────┘                  │
│                             │                           │
│                    ┌────────▼────────┐                  │
│                    │   ChromaDB      │                  │
│                    │   (Patterns)    │                  │
│                    └────────┬────────┘                  │
│                             │                           │
│                     Similar Cases                       │
│                             │                           │
│                             ▼                           │
│  ┌────────────┐    ┌───────────────┐                   │
│  │  Claude    │◀───│   Enhanced    │                   │
│  │  Sonnet    │    │   Prompt      │                   │
│  └──────┬─────┘    └───────────────┘                   │
│         │                                               │
│    Improved Text                                        │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │   Evaluate   │──────▶ Store Pattern                 │
│  │   & Store    │       (if successful)                │
│  └──────────────┘                                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## 🚀 Next Steps

1. **First Run**: `python scripts/apply_improvements_with_rag.py`
2. **Verify Storage**: Check ChromaDB has patterns
3. **Second Run**: Notice improved suggestions
4. **Scale Up**: Use for all papers to build expertise

## 📞 Support

- **Issues**: Check troubleshooting section above
- **Docker Problems**: Ensure Docker is installed and running
- **API Keys**: Verify ANTHROPIC_API_KEY and OPENAI_API_KEY
- **Performance**: RAG improves after 5+ papers

---

**Made with ❤️ by AI-CoScientist Team**

*Learning from experience, one paper at a time.*
