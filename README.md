# AI CoScientist

AI-powered autonomous scientific research system that automates the entire research workflow from hypothesis generation to paper writing.

## 🎯 Features

- **Hypothesis Generation**: AI-driven hypothesis generation from literature analysis
- **Experiment Design**: Automated experimental protocol design with statistical planning
- **Data Analysis**: Intelligent data analysis and visualization
- **Paper Writing**: Automated scientific paper generation
- **Peer Review**: AI-powered peer review simulation
- **Knowledge Base**: Semantic search across scientific literature

## 🏗️ Architecture

- **API Gateway**: FastAPI-based RESTful API
- **LLM Service**: Multi-provider LLM integration (OpenAI, Anthropic)
- **Knowledge Base**: Hybrid search with ChromaDB and PostgreSQL
- **Message Queue**: Celery with RabbitMQ for async tasks
- **Caching**: Redis for performance optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poetry (for dependency management)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/ai-coscientist.git
cd ai-coscientist
```

2. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Start with Docker Compose**:
```bash
docker-compose up -d
```

4. **Or run locally with Poetry**:
```bash
# Install dependencies
poetry install

# Run database migrations
poetry run alembic upgrade head

# Start the API server
poetry run uvicorn src.main:app --reload
```

### Access the API

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Prometheus Metrics: http://localhost:9090
- Grafana Dashboard: http://localhost:3001

## 📖 API Documentation

### Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'
```

### Create a Project

```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Drug Discovery for Alzheimers",
    "domain": "pharmacology",
    "research_question": "What novel compounds can inhibit amyloid-beta aggregation?"
  }'
```

### Generate Hypotheses

```bash
curl -X POST http://localhost:8000/api/v1/projects/{project_id}/hypotheses/generate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "research_question": "How to reduce CRISPR off-target effects?",
    "num_hypotheses": 5,
    "creativity_level": 0.8
  }'
```

## 🧪 Development

### Project Structure

```
ai-coscientist/
├── src/
│   ├── api/              # API endpoints
│   ├── core/             # Core configuration
│   ├── models/           # Database models
│   ├── services/         # Business logic
│   └── utils/            # Utilities
├── tests/                # Test suite
├── docker/               # Docker configurations
├── config/               # Configuration files
├── design/               # Design documents
└── scripts/              # Utility scripts
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_api.py
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

Key environment variables (see `.env.example` for complete list):

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key (min 32 characters)

## 📊 Monitoring

- **Prometheus**: Metrics collection at http://localhost:9090
- **Grafana**: Visualization at http://localhost:3001 (admin/admin)
- **Logs**: Structured JSON logs with correlation IDs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation
- Use type hints
- Add docstrings

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Anthropic for Claude API
- Semantic Scholar for literature data
- All contributors and researchers

## 📧 Contact

- Project Link: https://github.com/your-org/ai-coscientist
- Issues: https://github.com/your-org/ai-coscientist/issues
- Email: contact@ai-coscientist.com

## 🗺️ Roadmap

- [x] Phase 1: Core Infrastructure
- [x] Phase 2: LLM Integration
- [ ] Phase 3: Research Engine
- [ ] Phase 4: Experiment Engine
- [ ] Phase 5: Paper Engine
- [ ] Phase 6: UI Development
- [ ] Phase 7: Production Deployment

## 📚 Documentation

- [Architecture Design](design/architecture/system_architecture.md)
- [API Specification](design/api/rest_api_spec.md)
- [Design Summary](design/DESIGN_SUMMARY.md)
