# AI-CoScientist: AI-Powered Scientific Paper Enhancement System

Intelligent paper evaluation and enhancement system using ensemble machine learning to assess and improve scientific manuscripts across four key dimensions: Novelty, Methodology, Clarity, and Significance.

## 🎯 Primary Use Case: Paper Enhancement

The core functionality of AI-CoScientist is **automated paper evaluation and targeted improvement**. The system achieved a real-world improvement of **7.96 → 8.34 (+0.38 points, +4.8%)** with GPT-4 narrative score reaching the maximum 9.0/10.

### Key Capabilities

✅ **Multi-Dimensional Scoring**: Comprehensive evaluation across Novelty, Methodology, Clarity, Significance
✅ **Ensemble Evaluation**: Combines GPT-4 (40%), Hybrid (30%), and Multi-task (30%) models
✅ **Automated Enhancement**: Generates targeted improvement strategies based on score gaps
✅ **Incremental Improvement**: Iterative enhancement with validation at each step
✅ **No GPU Required**: Runs on standard CPU hardware

### Quick Start: Evaluate a Paper (30 seconds)

```bash
# Evaluate any scientific paper
python scripts/evaluate_docx.py /path/to/your/paper.docx
```

**Output**:
```
📊 Paper Evaluation Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score: 7.96/10 (confidence: 0.88)

Dimensional Scores:
  Novelty       : 7.46/10  ⚠️ Improve positioning
  Methodology   : 7.89/10  ✅ Strong
  Clarity       : 7.45/10  ⚠️ Enhance narrative
  Significance  : 7.40/10  ⚠️ Quantify impact

Model Contributions:
  GPT-4 (40%):        8.00/10  [Narrative quality]
  Hybrid (30%):       7.97/10  [Technical depth]
  Multi-task (30%):   7.88/10  [Novelty assessment]
```

📖 **For complete tutorial and advanced features**: See [PAPER_ENHANCEMENT_GUIDE.md](PAPER_ENHANCEMENT_GUIDE.md)

## 🏗️ System Architecture

AI-CoScientist provides **four ways** to use the paper enhancement system:

### 1. Interactive Chatbot (Recommended for Beginners) 🆕

```bash
# Start conversational review session
python scripts/chat_reviewer.py
```

**Natural language interface**:
```
💬 You: "Review my paper: paper.docx"
🤖 Bot: "Score: 7.96/10. Methodology is strong but novelty needs work.
        What would you like to improve?"

💬 You: "Get me to 8.5+"
🤖 Bot: "Here are 3 suggestions to reach 8.5:
        1. Transform title (30 min, +0.3 points)
        2. Add theoretical justification (2 hours, +0.3 points)
        3. Quantify impact (1 hour, +0.2 points)
        Which one first?"

💬 You: "Do number 2"
🤖 Bot: "Adding theoretical section... Done! New score: 8.34/10"
```

📖 **Chatbot Guide**: See [CHATBOT_GUIDE.md](CHATBOT_GUIDE.md) for detailed usage

### 2. Command-Line Scripts (Fastest)

```bash
# Evaluate paper
python scripts/evaluate_docx.py paper.docx

# Add theoretical justification (~1200 words)
python scripts/insert_theoretical_justification.py

# Add method comparison table
python scripts/add_comparison_table.py

# Re-evaluate enhanced paper
python scripts/evaluate_docx.py paper-revised.txt
```

### 3. Service Layer (Programmatic)

```python
from src.services.paper import PaperAnalyzer, PaperImprover
from src.services.llm.service import LLMService

# Initialize services
llm = LLMService(primary_provider="openai")
analyzer = PaperAnalyzer(llm, db)
improver = PaperImprover(llm, db)

# Analyze paper
scores = await analyzer.analyze_paper(paper_id)
suggestions = await analyzer.get_improvement_suggestions(paper_id)

# Apply improvements
improved = await improver.improve_section(paper_id, "introduction", suggestions)
```

### 4. REST API (Production)

```bash
# Start API server
poetry run uvicorn src.main:app --reload

# Access interactive docs at http://localhost:8000/docs
```

**API Endpoints**:
- `POST /api/v1/papers/analyze` - Evaluate paper quality
- `POST /api/v1/papers/improve` - Generate improvements
- `GET /api/v1/papers/{id}/scores` - Retrieve scores
- Full API documentation: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

## 🚀 Quick Comparison: Which Method to Use?

| Method | Best For | Speed | Ease of Use | Flexibility |
|--------|----------|-------|-------------|-------------|
| **Chatbot** | Beginners, exploratory review | Medium | ⭐⭐⭐⭐⭐ Easiest | High |
| **Scripts** | Quick evaluations, automation | ⚡ Fastest | ⭐⭐⭐⭐ Easy | Medium |
| **Services** | Custom workflows, integration | Fast | ⭐⭐⭐ Moderate | Very High |
| **API** | Production, web apps, teams | Fast | ⭐⭐ Advanced | Highest |

## 📦 Installation

### Prerequisites

- Python 3.8+
- (Optional) Docker and Docker Compose for full system
- (Optional) PostgreSQL for database persistence

### Quick Install (Scripts Only)

```bash
# Clone repository
git clone https://github.com/Transconnectome/AI-CoScientist.git
cd AI-CoScientist

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Verify installation
python scripts/evaluate_docx.py --help
```

### Full System Install (API + Services)

```bash
# Install with Poetry
poetry install

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
poetry run alembic upgrade head

# Start API server
poetry run uvicorn src.main:app --reload
```

### Docker Deployment

```bash
# Start all services (PostgreSQL, Redis, API)
docker-compose up -d

# Access API at http://localhost:8000
# Access docs at http://localhost:8000/docs
```

## 🎓 Usage Examples

### Example 1: Basic Evaluation

```bash
python scripts/evaluate_docx.py ~/Desktop/my-paper.docx
```

System generates:
- Overall score and confidence level
- Dimensional breakdown (Novelty, Methodology, Clarity, Significance)
- Model-specific assessments
- Improvement strategy document in `claudedocs/`

### Example 2: Enhancement Workflow

```bash
# Step 1: Baseline evaluation
python scripts/evaluate_docx.py paper.docx > baseline-score.txt

# Step 2: Review improvement strategy
cat claudedocs/paper_improvement_strategy_*.md

# Step 3: Apply highest-impact enhancement
python scripts/insert_theoretical_justification.py

# Step 4: Re-evaluate
python scripts/evaluate_docx.py paper-revised-v2.txt

# Result: Score improved from 7.96 to 8.34 (+0.38 points)
```

### Example 3: Programmatic Use

```python
#!/usr/bin/env python3
"""Automated paper improvement pipeline."""

import asyncio
from pathlib import Path
from src.services.paper import PaperAnalyzer, PaperImprover
from src.services.llm.service import LLMService

async def improve_paper(paper_path: str):
    """Analyze and improve a paper."""

    # Initialize services
    llm = LLMService(primary_provider="openai")
    analyzer = PaperAnalyzer(llm, None)

    # Read paper
    text = Path(paper_path).read_text()

    # Analyze
    scores = await analyzer.analyze_text(text)
    print(f"Overall Score: {scores['overall']:.2f}/10")

    # Get improvement suggestions
    suggestions = await analyzer.suggest_improvements(scores)

    # Display top 3 suggestions
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"{i}. {suggestion['title']}")
        print(f"   Impact: +{suggestion['expected_gain']:.2f} points")
        print(f"   Effort: {suggestion['estimated_hours']} hours")

if __name__ == "__main__":
    asyncio.run(improve_paper("paper.docx"))
```

## 📊 Evaluation Methodology

### Ensemble Architecture

```
Paper Input
    │
    ├─→ GPT-4 Scorer (40%)      → Narrative quality, communication
    ├─→ Hybrid Scorer (30%)     → Technical depth, methodology
    └─→ Multi-task Scorer (30%) → Novelty, contribution
    │
    └─→ Weighted Ensemble → Overall Score (0-10)
                         → 4 Dimensional Scores
                         → Confidence Level
```

### Scoring Dimensions

| Dimension | Weight | Evaluates |
|-----------|--------|-----------|
| **Novelty** | 25% | Originality, paradigm shift vs incremental |
| **Methodology** | 35% | Experimental rigor, validation, reproducibility |
| **Clarity** | 20% | Writing quality, organization, communication |
| **Significance** | 20% | Real-world impact, clinical/practical value |

### Score Interpretation

| Score | Quality | Publication Outlook |
|-------|---------|---------------------|
| 9.0-10.0 | **Exceptional** | Nature, Science, Cell |
| 8.5-8.9 | **Excellent** | Top specialty journals |
| 8.0-8.4 | **Very Good** | Strong specialty journals |
| 7.5-7.9 | **Good** | Respectable journals |
| 7.0-7.4 | **Acceptable** | Mid-tier journals |
| <7.0 | **Needs Work** | Major revisions required |

## 🚀 Enhancement Strategies

### Quick Wins (5-10 hours, +0.3-0.5 points)

1. **Transform Title** (30 min) → +0.3-0.5 GPT-4
   - From: "A more comprehensive analysis..."
   - To: "Solving the [Crisis] in [Field]: A [Framework]"

2. **Rewrite Abstract with Crisis Framing** (45 min) → +0.2-0.4 GPT-4
   - Lead with problem/crisis statement
   - Quantify gap ("50% of methods fail")
   - Position solution as paradigm shift

3. **Quantify All Impact Statements** (1-2 hours) → +0.1-0.3 Significance
   - Replace "improves outcomes" with "34% variance reduction"
   - Replace "reduces costs" with "$9.65B projected savings"

4. **Add Theoretical Justification** (2 hours with script) → +0.2-0.3 Methodology
   ```bash
   python scripts/insert_theoretical_justification.py
   ```

### Medium Effort (10-20 hours, +0.3-0.6 points)

- Multi-dataset validation
- Comparative benchmarking studies
- Comprehensive sensitivity analyses

### High Effort (20-40 hours, +0.4-0.7 points)

- Theoretical proofs and bounds
- Simulation studies (100+ scenarios)
- Clinical validation

**Detailed strategies**: See [PAPER_ENHANCEMENT_GUIDE.md](PAPER_ENHANCEMENT_GUIDE.md)

## 🗺️ System Roadmap

### ✅ Phase 1: Core Infrastructure (Complete)
- [x] FastAPI-based REST API
- [x] PostgreSQL database with SQLAlchemy ORM
- [x] Redis caching layer
- [x] Docker Compose deployment
- [x] Health checks and monitoring

### ✅ Phase 2: LLM Integration (Complete)
- [x] Multi-provider LLM service (OpenAI, Anthropic)
- [x] Prompt template system with Jinja2
- [x] Usage tracking and cost calculation
- [x] Streaming support

### ✅ Phase 3: Paper Enhancement Engine (Complete)
- [x] **Three-model ensemble evaluation**
- [x] **Dimensional scoring (Novelty, Methodology, Clarity, Significance)**
- [x] **Automated enhancement scripts**
- [x] **Improvement strategy generation**
- [x] **Service layer (PaperParser, PaperAnalyzer, PaperImprover)**
- [x] **Validation: Real paper improved 7.96 → 8.34 (+4.8%)**

### 🔄 Phase 4: Research Engine (In Progress)
- [ ] Hypothesis generation from literature
- [ ] Automated literature review
- [ ] Knowledge graph construction

### 📋 Phase 5: Experiment Engine (Planned)
- [ ] Experimental protocol design
- [ ] Statistical power analysis
- [ ] Data analysis automation

### 🎯 Phase 6: Full Paper Generation (Planned)
- [ ] Section-by-section generation
- [ ] Citation management
- [ ] Figure and table generation

### 🖥️ Phase 7: UI Development (Planned)
- [ ] Web interface for paper upload
- [ ] Interactive improvement dashboard
- [ ] Real-time scoring and suggestions

## 📚 Documentation

### Core Guides
- 📖 **[Paper Enhancement Guide](PAPER_ENHANCEMENT_GUIDE.md)** - Complete tutorial with examples
- 📘 **[API Reference](docs/API_REFERENCE.md)** - Full API documentation
- 📙 **[Documentation Index](docs/INDEX.md)** - Master documentation hub

### Status Reports
- ✅ **[System Ready](SYSTEM_READY.md)** - System configuration complete
- ✅ **[Implementation Status](IMPLEMENTATION_STATUS.md)** - What's built
- ✅ **[Enhancement Results](claudedocs/ENHANCEMENT_RESULTS_SUMMARY.md)** - Case study results

## 🏆 Real-World Results

**Case Study: Improving a Neuroscience Paper**

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| **Overall Score** | 7.96 | 8.34 | +0.38 (+4.8%) |
| **GPT-4 Score** | 8.00 | 9.00 | +1.00 🎯 Max |
| **Novelty** | 7.46 | 7.92 | +0.46 |
| **Methodology** | 7.89 | 8.15 | +0.26 |
| **Clarity** | 7.45 | 7.89 | +0.44 |
| **Significance** | 7.40 | 8.12 | +0.72 |

**Enhancements Applied**:
1. Title transformation (crisis framing)
2. Abstract rewrite with quantified impact
3. Theoretical justification section (~1200 words)
4. Impact quantification throughout

**Time Investment**: 5-8 hours
**Outcome**: Publication-ready for high-quality specialty journals

## 🛠️ Development

### Project Structure

```
AI-CoScientist/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Configuration, database, Redis
│   ├── models/           # SQLAlchemy models
│   ├── services/         # Business logic
│   │   ├── llm/         # LLM service with multi-provider support
│   │   ├── paper/       # Paper analysis and improvement services
│   │   └── scoring/     # Ensemble scoring models
│   └── utils/            # Utilities
├── scripts/              # Enhancement scripts
│   ├── evaluate_docx.py              # Main evaluation script
│   ├── insert_theoretical_justification.py
│   ├── add_impact_boxes.py
│   ├── add_comparison_table.py
│   └── add_literature_implications.py
├── tests/                # Test suite
├── docs/                 # Documentation
│   ├── INDEX.md         # Master documentation index
│   └── API_REFERENCE.md # Complete API documentation
├── claudedocs/           # Generated analysis and reports
├── docker/               # Docker configurations
└── design/               # Architecture design documents
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test
poetry run pytest tests/services/test_paper.py -v
```

### Code Quality

```bash
# Format code
poetry run black src tests

# Lint code
poetry run ruff check src tests

# Type checking
poetry run mypy src
```

## 🔐 Environment Variables

Create `.env` file with:

```bash
# Required for GPT-4 evaluation
ANTHROPIC_API_KEY=sk-ant-api03-your_key_here

# Optional for additional models
OPENAI_API_KEY=sk-your_openai_key

# Database (optional, defaults to SQLite)
DATABASE_URL=postgresql://user:pass@localhost:5432/ai_coscientist

# Redis (optional, for API caching)
REDIS_URL=redis://localhost:6379/0

# API configuration
SECRET_KEY=your_secret_key_min_32_characters
```

## 🤝 Contributing

Contributions welcome! Areas for development:

- Additional evaluation models and ensembles
- Domain-specific scoring rubrics
- Automated enhancement generation
- Multi-language support
- Web interface development

**Process**:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new features
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Anthropic** for Claude API (GPT-4 evaluation)
- **OpenAI** for GPT-4 API (alternative provider)
- **Transconnectome Lab** for validation datasets
- All contributors and researchers

## 📧 Contact

- **GitHub**: https://github.com/Transconnectome/AI-CoScientist
- **Issues**: https://github.com/Transconnectome/AI-CoScientist/issues
- **Lab Website**: [Transconnectome Lab]

## 📖 Citation

If you use AI-CoScientist in your research:

```bibtex
@software{ai_coscientist_2024,
  title = {AI-CoScientist: AI-Powered Scientific Paper Enhancement System},
  author = {Transconnectome Lab},
  year = {2024},
  url = {https://github.com/Transconnectome/AI-CoScientist},
  note = {Ensemble machine learning for automated paper evaluation and improvement}
}
```
