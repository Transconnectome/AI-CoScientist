# RAG Enhancement System - API Reference & Usage Guide

## 📚 API Overview

The RAG Enhancement System provides a comprehensive REST API for scientific research automation, paper improvement, and intelligent content generation. This guide covers all endpoints, usage patterns, and integration examples.

## 🌐 Base URL & Authentication

```
Base URL: https://api.yourcompany.com/v1
Authentication: Bearer Token (JWT)
Content-Type: application/json
```

### Authentication
```bash
# Get access token
curl -X POST "${BASE_URL}/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## 🎯 Core RAG Endpoints

### 1. Unified RAG Search

Search across all RAG strategies with intelligent routing.

```http
POST /v1/rag/search
```

**Request Body:**
```json
{
  "query": "Explain neural mechanisms of autism spectrum disorder",
  "strategy_override": "graph_rag",  // Optional: force specific strategy
  "enable_fallback": true,           // Optional: enable fallback strategies
  "metadata": {
    "domain": "neuroscience",
    "complexity": "high",
    "intent": "analytical",
    "user_context": "researcher"
  },
  "filters": {
    "date_range": {
      "start": "2020-01-01",
      "end": "2025-01-01"
    },
    "sources": ["pubmed", "arxiv"],
    "quality_threshold": 0.8
  }
}
```

**Response:**
```json
{
  "id": "search_12345",
  "content": "Autism spectrum disorder (ASD) involves complex neural mechanisms...",
  "strategy_used": "graph_rag",
  "confidence": 0.92,
  "quality_metrics": {
    "faithfulness": 0.88,
    "answer_relevancy": 0.91,
    "context_precision": 0.85
  },
  "sources": [
    {
      "id": "doc_001",
      "title": "Neural Mechanisms in ASD: A Comprehensive Review",
      "authors": ["Smith, J.", "Doe, A."],
      "journal": "Nature Neuroscience",
      "year": 2023,
      "doi": "10.1038/nn.2023.001",
      "relevance_score": 0.94
    }
  ],
  "execution_time": 1.2,
  "tokens_used": 1250,
  "cached": false,
  "graph_path": [
    {"entity": "autism", "type": "condition"},
    {"entity": "neural_mechanisms", "type": "concept"},
    {"entity": "sensory_processing", "type": "function"}
  ]
}
```

**cURL Example:**
```bash
curl -X POST "${BASE_URL}/v1/rag/search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest treatments for autism?",
    "metadata": {
      "domain": "medical",
      "complexity": "moderate"
    }
  }'
```

**Python Example:**
```python
import requests

def search_rag(query: str, strategy: str = None):
    url = f"{BASE_URL}/v1/rag/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "metadata": {
            "domain": "neuroscience",
            "complexity": "high"
        }
    }

    if strategy:
        payload["strategy_override"] = strategy

    response = requests.post(url, headers=headers, json=payload)
    return response.json()

# Usage
result = search_rag("Explain autism neural mechanisms", "graph_rag")
print(f"Strategy used: {result['strategy_used']}")
print(f"Response: {result['content'][:200]}...")
```

### 2. Strategy-Specific Endpoints

#### Simple RAG Search
```http
POST /v1/rag/simple
```

**Request:**
```json
{
  "query": "What is machine learning?",
  "max_results": 5,
  "similarity_threshold": 0.7
}
```

#### Multimodal RAG Search
```http
POST /v1/rag/multimodal
```

**Request:**
```json
{
  "query": "Analyze brain scan abnormalities in autism",
  "modalities": ["text", "images", "tables"],
  "image_analysis": {
    "extract_text": true,
    "analyze_figures": true,
    "detect_charts": true
  }
}
```

#### GraphRAG Search
```http
POST /v1/rag/graph
```

**Request:**
```json
{
  "query": "How do genetic factors relate to autism symptoms?",
  "max_hops": 3,
  "entity_types": ["gene", "symptom", "condition"],
  "relationship_threshold": 0.6
}
```

### 3. Batch Processing

Process multiple queries efficiently.

```http
POST /v1/rag/batch
```

**Request:**
```json
{
  "queries": [
    {
      "id": "query_1",
      "query": "What causes autism?",
      "strategy": "simple"
    },
    {
      "id": "query_2",
      "query": "Latest autism treatments with brain imaging",
      "strategy": "multimodal"
    }
  ],
  "parallel_processing": true,
  "timeout": 30
}
```

**Response:**
```json
{
  "batch_id": "batch_12345",
  "status": "completed",
  "results": [
    {
      "query_id": "query_1",
      "status": "success",
      "result": { /* RAG response */ }
    },
    {
      "query_id": "query_2",
      "status": "success",
      "result": { /* RAG response */ }
    }
  ],
  "summary": {
    "total_queries": 2,
    "successful": 2,
    "failed": 0,
    "total_time": 3.5
  }
}
```

## 🧠 Query Classification & Optimization

### Get Query Classification
```http
POST /v1/rag/classify
```

**Request:**
```json
{
  "query": "How do genetic mutations affect neural development in autism?"
}
```

**Response:**
```json
{
  "classification": {
    "complexity": "complex",
    "domain": "medical",
    "intent": "analytical",
    "confidence": 0.89
  },
  "recommended_strategies": [
    {"strategy": "graph_rag", "score": 0.92},
    {"strategy": "enhanced_dd_raptor", "score": 0.87},
    {"strategy": "hybrid", "score": 0.73}
  ],
  "reasoning": "Complex medical query requiring relationship analysis between genetics and neurodevelopment"
}
```

### Strategy Performance Metrics
```http
GET /v1/rag/strategies/performance
```

**Response:**
```json
{
  "strategies": {
    "simple": {
      "usage_count": 1250,
      "success_rate": 0.94,
      "avg_response_time": 0.8,
      "avg_quality_score": 0.82
    },
    "graph_rag": {
      "usage_count": 890,
      "success_rate": 0.91,
      "avg_response_time": 2.1,
      "avg_quality_score": 0.89
    },
    "multimodal": {
      "usage_count": 450,
      "success_rate": 0.87,
      "avg_response_time": 3.2,
      "avg_quality_score": 0.91
    }
  },
  "overall_metrics": {
    "total_queries": 2590,
    "avg_response_time": 1.6,
    "avg_quality_score": 0.87
  }
}
```

## 📄 Paper Management & Improvement

### Upload Paper for Analysis
```http
POST /v1/papers/upload
```

**Form Data:**
```
file: paper.pdf
metadata: {
  "title": "Novel Autism Treatment Approaches",
  "authors": ["Dr. Smith", "Dr. Johnson"],
  "domain": "neuroscience",
  "target_quality": 0.9
}
```

**Response:**
```json
{
  "paper_id": "paper_12345",
  "status": "processing",
  "analysis_id": "analysis_67890",
  "estimated_completion": "2025-01-01T10:30:00Z",
  "extracted_content": {
    "text_pages": 15,
    "figures": 8,
    "tables": 3,
    "references": 45
  }
}
```

### Get Paper Analysis
```http
GET /v1/papers/{paper_id}/analysis
```

**Response:**
```json
{
  "paper_id": "paper_12345",
  "analysis": {
    "quality_score": 0.78,
    "strengths": [
      "Well-structured methodology",
      "Comprehensive literature review",
      "Statistical analysis is robust"
    ],
    "weaknesses": [
      "Limited sample size discussion",
      "Missing control group details",
      "Conclusions need strengthening"
    ],
    "suggestions": [
      {
        "type": "methodology",
        "priority": "high",
        "description": "Add detailed power analysis for sample size justification",
        "confidence": 0.91
      }
    ]
  },
  "rag_context": {
    "related_papers": 23,
    "relevant_methodologies": 8,
    "benchmark_comparisons": 5
  }
}
```

### Apply Improvements
```http
POST /v1/papers/{paper_id}/improve
```

**Request:**
```json
{
  "improvement_types": ["methodology", "writing_clarity", "citations"],
  "target_quality": 0.9,
  "preserve_author_voice": true,
  "custom_instructions": "Focus on strengthening statistical analysis section"
}
```

## 🔄 Self-Learning & Feedback

### Submit Feedback
```http
POST /v1/rag/feedback
```

**Request:**
```json
{
  "search_id": "search_12345",
  "rating": 4,
  "feedback_type": "quality",
  "comments": "Very helpful response, but could include more recent studies",
  "specific_issues": [
    {
      "type": "completeness",
      "severity": "minor",
      "description": "Missing 2024 research papers"
    }
  ]
}
```

**Response:**
```json
{
  "feedback_id": "feedback_67890",
  "status": "recorded",
  "impact": {
    "strategy_adjustment": "minor",
    "learning_applied": true,
    "estimated_improvement": 0.03
  }
}
```

### System Learning Status
```http
GET /v1/rag/learning/status
```

**Response:**
```json
{
  "learning_status": {
    "total_feedback_entries": 2847,
    "recent_improvements": [
      {
        "date": "2025-01-01",
        "improvement": "Enhanced graph traversal for medical queries",
        "impact_score": 0.12
      }
    ],
    "strategy_adaptations": {
      "graph_rag": {
        "parameter_adjustments": 5,
        "last_update": "2025-01-01T08:00:00Z"
      }
    }
  },
  "performance_trends": {
    "quality_improvement": 0.08,
    "response_time_improvement": -0.15,
    "user_satisfaction": 0.11
  }
}
```

## 📊 Monitoring & Analytics

### System Metrics
```http
GET /v1/monitoring/metrics
```

**Response:**
```json
{
  "system_health": {
    "status": "healthy",
    "uptime": "99.97%",
    "last_incident": "2024-12-15T03:00:00Z"
  },
  "performance_metrics": {
    "requests_per_minute": 145,
    "avg_response_time": 1.2,
    "p95_response_time": 2.8,
    "error_rate": 0.003
  },
  "resource_usage": {
    "cpu_utilization": 0.65,
    "memory_usage": 0.78,
    "storage_usage": 0.45
  },
  "strategy_distribution": {
    "simple": 0.25,
    "hybrid": 0.35,
    "graph_rag": 0.20,
    "multimodal": 0.15,
    "enhanced_dd_raptor": 0.05
  }
}
```

### Quality Analytics
```http
GET /v1/monitoring/quality
```

**Response:**
```json
{
  "quality_metrics": {
    "average_scores": {
      "faithfulness": 0.87,
      "answer_relevancy": 0.89,
      "context_precision": 0.84,
      "context_recall": 0.82
    },
    "score_distribution": {
      "excellent": 0.45,    // > 0.9
      "good": 0.35,         // 0.8-0.9
      "adequate": 0.15,     // 0.7-0.8
      "poor": 0.05          // < 0.7
    },
    "improvement_trends": {
      "daily_improvement": 0.002,
      "weekly_improvement": 0.015,
      "monthly_improvement": 0.067
    }
  }
}
```

## 🔧 Configuration & Administration

### Update Strategy Configuration
```http
PUT /v1/admin/config/strategies
```

**Request:**
```json
{
  "strategy_weights": {
    "simple": 0.1,
    "hybrid": 0.3,
    "graph_rag": 0.4,
    "multimodal": 0.2
  },
  "quality_thresholds": {
    "faithfulness": 0.8,
    "answer_relevancy": 0.75,
    "response_time": 2.0
  },
  "optimization_settings": {
    "enable_caching": true,
    "cache_ttl": 3600,
    "enable_batch_processing": true
  }
}
```

### System Health Check
```http
GET /v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T10:00:00Z",
  "checks": {
    "database": "healthy",
    "chromadb": "healthy",
    "neo4j": "healthy",
    "redis": "healthy",
    "ml_models": "healthy"
  },
  "version": "1.0.0",
  "uptime": 864000
}
```

## 📱 Client SDKs & Examples

### Python SDK
```python
# Installation: pip install ai-coscientist-client

from ai_coscientist import RAGClient

# Initialize client
client = RAGClient(
    base_url="https://api.yourcompany.com/v1",
    api_key="your_api_key"
)

# Simple search
result = await client.search(
    query="What are the neural mechanisms of autism?",
    strategy="graph_rag"
)

# Multimodal search
multimodal_result = await client.multimodal_search(
    query="Analyze brain imaging data for autism markers",
    include_images=True,
    include_tables=True
)

# Batch processing
queries = [
    {"query": "Autism genetic factors", "strategy": "graph_rag"},
    {"query": "Treatment effectiveness", "strategy": "simple"}
]
batch_results = await client.batch_search(queries)

# Paper improvement
paper_id = await client.upload_paper("research_paper.pdf")
analysis = await client.analyze_paper(paper_id)
improved_paper = await client.improve_paper(
    paper_id,
    target_quality=0.9
)
```

### JavaScript SDK
```javascript
// Installation: npm install @ai-coscientist/client

import { RAGClient } from '@ai-coscientist/client';

const client = new RAGClient({
  baseUrl: 'https://api.yourcompany.com/v1',
  apiKey: 'your_api_key'
});

// Search with async/await
async function searchRAG() {
  try {
    const result = await client.search({
      query: "Latest autism research findings",
      metadata: {
        domain: "neuroscience",
        complexity: "moderate"
      }
    });

    console.log('Strategy used:', result.strategy_used);
    console.log('Response:', result.content);
    console.log('Quality score:', result.quality_metrics.answer_relevancy);

  } catch (error) {
    console.error('Search failed:', error);
  }
}

// Stream responses
const stream = client.searchStream({
  query: "Comprehensive autism treatment review"
});

stream.on('data', (chunk) => {
  console.log('Received chunk:', chunk);
});

stream.on('end', (finalResult) => {
  console.log('Final result:', finalResult);
});
```

### curl Examples Collection
```bash
#!/bin/bash
# rag_api_examples.sh

BASE_URL="https://api.yourcompany.com/v1"
TOKEN="your_jwt_token"

# 1. Simple search
curl -X POST "${BASE_URL}/rag/search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What causes autism spectrum disorder?",
    "metadata": {"domain": "medical"}
  }'

# 2. Graph RAG search
curl -X POST "${BASE_URL}/rag/graph" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Relationship between genetics and autism symptoms",
    "max_hops": 2,
    "entity_types": ["gene", "symptom"]
  }'

# 3. Multimodal search
curl -X POST "${BASE_URL}/rag/multimodal" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Brain scan abnormalities in autism patients",
    "modalities": ["text", "images"],
    "image_analysis": {"extract_text": true}
  }'

# 4. Upload paper for analysis
curl -X POST "${BASE_URL}/papers/upload" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@research_paper.pdf" \
  -F 'metadata={"title":"My Research","domain":"neuroscience"}'

# 5. Get system metrics
curl -X GET "${BASE_URL}/monitoring/metrics" \
  -H "Authorization: Bearer ${TOKEN}"

# 6. Submit feedback
curl -X POST "${BASE_URL}/rag/feedback" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "search_id": "search_12345",
    "rating": 5,
    "comments": "Excellent response quality"
  }'
```

## ⚡ Performance Optimization

### Request Optimization
```python
# Use connection pooling
import aiohttp

async def optimized_requests():
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Reuse session for multiple requests
        tasks = []
        for query in batch_queries:
            task = session.post(f"{BASE_URL}/rag/search", json=query)
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]

# Enable compression
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip, deflate"
}

# Use caching headers
headers["Cache-Control"] = "max-age=300"
headers["If-None-Match"] = etag_from_previous_request
```

### Rate Limiting
```python
# Built-in rate limiting respects these limits:
# - 100 requests per minute per user
# - 1000 requests per hour per API key
# - Burst allowance: 20 requests

# Implement client-side rate limiting
import asyncio
from asyncio import Semaphore

class RateLimitedClient:
    def __init__(self, max_concurrent=10):
        self.semaphore = Semaphore(max_concurrent)
        self.last_request_time = 0
        self.min_interval = 0.1  # 100ms between requests

    async def request(self, *args, **kwargs):
        async with self.semaphore:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_request_time

            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)

            result = await self._make_request(*args, **kwargs)
            self.last_request_time = asyncio.get_event_loop().time()
            return result
```

## 🔍 Error Handling & Debugging

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_STRATEGY",
    "message": "Strategy 'invalid_strategy' is not supported",
    "details": {
      "supported_strategies": ["simple", "hybrid", "graph_rag", "multimodal"],
      "request_id": "req_12345",
      "timestamp": "2025-01-01T10:00:00Z"
    }
  }
}
```

### Common Error Codes
- `INVALID_STRATEGY`: Unsupported RAG strategy specified
- `QUERY_TOO_LONG`: Query exceeds maximum length (10,000 characters)
- `RATE_LIMIT_EXCEEDED`: Too many requests (see rate limiting)
- `INSUFFICIENT_CONTEXT`: No relevant documents found
- `QUALITY_THRESHOLD_NOT_MET`: Response quality below threshold
- `SERVICE_UNAVAILABLE`: Backend service temporarily unavailable
- `INVALID_TOKEN`: Authentication token invalid or expired

### Debugging Tools
```bash
# Enable debug mode
curl -X POST "${BASE_URL}/rag/search" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-Debug-Mode: true" \
  -d '{"query": "test query"}'

# Response includes debug information:
{
  "content": "...",
  "debug": {
    "strategy_selection_reasoning": "Graph RAG chosen due to entity-rich query",
    "retrieval_stats": {
      "documents_searched": 1500,
      "relevant_documents": 25,
      "search_time": 0.3
    },
    "generation_stats": {
      "tokens_generated": 450,
      "generation_time": 0.8
    }
  }
}
```

## 🎯 Best Practices

### Query Optimization
```python
# Good: Specific, well-structured queries
good_query = {
    "query": "What are the specific neural circuits affected in autism spectrum disorder, particularly in sensory processing?",
    "metadata": {
        "domain": "neuroscience",
        "complexity": "high",
        "intent": "analytical"
    }
}

# Avoid: Vague or overly broad queries
avoid_query = {
    "query": "Tell me about autism",  # Too broad
    "metadata": {}  # Missing helpful context
}
```

### Batch Processing
```python
# Efficient batch processing
def create_efficient_batches(queries, batch_size=10):
    """Group queries by strategy for optimal processing."""
    strategy_groups = {}

    for query in queries:
        strategy = classify_query_strategy(query)
        if strategy not in strategy_groups:
            strategy_groups[strategy] = []
        strategy_groups[strategy].append(query)

    # Process each strategy group separately
    batches = []
    for strategy, group_queries in strategy_groups.items():
        for i in range(0, len(group_queries), batch_size):
            batch = group_queries[i:i + batch_size]
            batches.append({
                "strategy": strategy,
                "queries": batch
            })

    return batches
```

### Monitoring Integration
```python
# Add request tracking
import uuid
import time

def track_api_request(func):
    async def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            # Log successful request
            logger.info(f"API request {request_id} completed in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"API request {request_id} failed after {duration:.2f}s: {e}")
            raise

    return wrapper
```

---

## 🎉 Conclusion

The RAG Enhancement System API provides comprehensive access to advanced retrieval and generation capabilities through a well-designed REST interface. With intelligent strategy routing, multimodal processing, self-learning capabilities, and extensive monitoring, it's designed for production scientific research applications.

**API Status**: 🚀 Production Ready
- Complete endpoint coverage for all system features
- Comprehensive error handling and debugging tools
- Multiple client SDK options available
- Production-grade monitoring and analytics
- Extensive usage examples and best practices included

**API Version**: v1.0.0
**Documentation Last Updated**: 2025-01-01