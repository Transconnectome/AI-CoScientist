# LLM Service Component Design

## 🎯 Purpose
중앙화된 LLM 통합 계층으로, 다양한 LLM 제공자에 대한 통합 인터페이스와 프롬프트 관리 기능을 제공합니다.

## 🏗️ Component Architecture

```
┌─────────────────────────────────────────────────┐
│           LLM Service Interface                 │
└─────────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ OpenAI  │   │Anthropic│   │  Local  │
│Adapter  │   │ Adapter │   │ Adapter │
└─────────┘   └─────────┘   └─────────┘
    │               │               │
    └───────────────┼───────────────┘
                    ▼
        ┌───────────────────────┐
        │   Prompt Manager      │
        │   - Template Engine   │
        │   - Validation        │
        │   - Optimization      │
        └───────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    ┌─────┐   ┌─────┐   ┌─────┐
    │Cache│   │Meter│   │ Log │
    └─────┘   └─────┘   └─────┘
```

## 📋 Interface Definition

### Core Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

class TaskType(Enum):
    """Scientific research task types"""
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    LITERATURE_ANALYSIS = "literature_analysis"
    EXPERIMENT_DESIGN = "experiment_design"
    DATA_ANALYSIS = "data_analysis"
    PAPER_WRITING = "paper_writing"
    PEER_REVIEW = "peer_review"

@dataclass
class LLMConfig:
    """Configuration for LLM requests"""
    provider: ModelProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    timeout: int = 60  # seconds

@dataclass
class LLMRequest:
    """LLM request payload"""
    prompt: str
    task_type: TaskType
    config: Optional[LLMConfig] = None
    context: Optional[Dict[str, Any]] = None
    system_message: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None

@dataclass
class LLMResponse:
    """LLM response payload"""
    content: str
    model: str
    provider: ModelProvider
    tokens_used: int
    cost: float
    latency_ms: float
    finish_reason: str
    metadata: Dict[str, Any]

class LLMServiceInterface(ABC):
    """Abstract interface for LLM service"""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from LLM"""
        pass

    @abstractmethod
    async def stream_complete(
        self,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream completion from LLM"""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings"""
        pass

    @abstractmethod
    def get_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for token usage"""
        pass
```

## 🔌 Provider Adapters

### OpenAI Adapter

```python
from openai import AsyncOpenAI
import tiktoken

class OpenAIAdapter(LLMServiceInterface):
    """OpenAI GPT adapter"""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.encoder = tiktoken.encoding_for_model("gpt-4")

        # Model pricing (per 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI"""
        config = request.config or self._get_default_config(request.task_type)

        messages = self._build_messages(request)

        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop_sequences,
            timeout=config.timeout
        )

        latency_ms = (time.time() - start_time) * 1000

        tokens_used = response.usage.total_tokens
        cost = self.get_cost(tokens_used, config.model)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=config.model,
            provider=ModelProvider.OPENAI,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
        )

    async def stream_complete(
        self,
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """Stream completion using OpenAI"""
        config = request.config or self._get_default_config(request.task_type)
        messages = self._build_messages(request)

        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI"""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def get_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for OpenAI API usage"""
        if model not in self.pricing:
            return 0.0

        # Simplified: assume 50/50 input/output split
        input_tokens = tokens // 2
        output_tokens = tokens - input_tokens

        pricing = self.pricing[model]
        cost = (
            (input_tokens / 1000) * pricing["input"] +
            (output_tokens / 1000) * pricing["output"]
        )
        return round(cost, 6)

    def _build_messages(self, request: LLMRequest) -> List[Dict]:
        """Build message array for OpenAI"""
        messages = []

        # System message
        if request.system_message:
            messages.append({
                "role": "system",
                "content": request.system_message
            })

        # Few-shot examples
        if request.examples:
            for example in request.examples:
                messages.append({
                    "role": "user",
                    "content": example["input"]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["output"]
                })

        # User prompt
        messages.append({
            "role": "user",
            "content": request.prompt
        })

        return messages

    def _get_default_config(self, task_type: TaskType) -> LLMConfig:
        """Get default config for task type"""
        configs = {
            TaskType.HYPOTHESIS_GENERATION: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4-turbo",
                temperature=0.8,
                max_tokens=1000
            ),
            TaskType.LITERATURE_ANALYSIS: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.3,
                max_tokens=2000
            ),
            TaskType.EXPERIMENT_DESIGN: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.5,
                max_tokens=1500
            ),
            TaskType.DATA_ANALYSIS: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.2,
                max_tokens=2000
            ),
            TaskType.PAPER_WRITING: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4-turbo",
                temperature=0.6,
                max_tokens=3000
            ),
            TaskType.PEER_REVIEW: LLMConfig(
                provider=ModelProvider.OPENAI,
                model="gpt-4",
                temperature=0.4,
                max_tokens=2000
            )
        }
        return configs.get(task_type, LLMConfig(
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        ))
```

### Anthropic Adapter

```python
from anthropic import AsyncAnthropic

class AnthropicAdapter(LLMServiceInterface):
    """Anthropic Claude adapter"""

    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

        # Model pricing (per 1M tokens)
        self.pricing = {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
        }

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Anthropic"""
        config = request.config or self._get_default_config(request.task_type)

        start_time = time.time()

        response = await self.client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=request.system_message or "",
            messages=[{"role": "user", "content": request.prompt}]
        )

        latency_ms = (time.time() - start_time) * 1000

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        cost = self.get_cost(tokens_used, config.model)

        return LLMResponse(
            content=response.content[0].text,
            model=config.model,
            provider=ModelProvider.ANTHROPIC,
            tokens_used=tokens_used,
            cost=cost,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        )

    def _get_default_config(self, task_type: TaskType) -> LLMConfig:
        """Get default config for task type"""
        return LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=2000
        )
```

## 🎨 Prompt Manager

```python
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any

class PromptManager:
    """Manage prompt templates and optimization"""

    def __init__(self, templates_dir: str):
        self.env = Environment(loader=FileSystemLoader(templates_dir))
        self.cache = {}

    def render_prompt(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Render prompt from template"""
        template = self.env.get_template(f"{template_name}.j2")
        return template.render(**context)

    def validate_prompt(self, prompt: str, max_tokens: int = 4000) -> bool:
        """Validate prompt doesn't exceed token limit"""
        # Rough estimation: 4 chars ≈ 1 token
        estimated_tokens = len(prompt) // 4
        return estimated_tokens <= max_tokens

    def optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for token efficiency"""
        # Remove extra whitespace
        optimized = " ".join(prompt.split())

        # Remove redundant phrases
        redundant_phrases = [
            "please note that",
            "it is important to",
            "you should be aware that"
        ]
        for phrase in redundant_phrases:
            optimized = optimized.replace(phrase, "")

        return optimized.strip()
```

### Prompt Templates

**hypothesis_generation.j2**
```jinja2
You are a scientific research assistant specializing in hypothesis generation.

Given the following research context:
- Domain: {{ domain }}
- Research Question: {{ research_question }}
- Literature Review: {{ literature_summary }}

{% if existing_hypotheses %}
Previous hypotheses to avoid duplication:
{% for hyp in existing_hypotheses %}
- {{ hyp }}
{% endfor %}
{% endif %}

Generate {{ num_hypotheses }} novel, testable scientific hypotheses that:
1. Address gaps in the current literature
2. Are falsifiable through experiments
3. Have clear independent and dependent variables
4. Are feasible with current technology

Format each hypothesis as:
- Hypothesis: [Clear statement]
- Rationale: [Why this is novel and important]
- Testability: [How it can be tested]
- Expected Outcome: [Predicted result]
```

**experiment_design.j2**
```jinja2
You are an expert in experimental design and methodology.

Hypothesis: {{ hypothesis }}

Design a rigorous experimental protocol that includes:

1. **Objective**: Clear experimental goal
2. **Variables**:
   - Independent variable(s)
   - Dependent variable(s)
   - Control variables
3. **Methodology**:
   - Sample size (with power analysis)
   - Experimental groups
   - Control groups
   - Randomization strategy
4. **Procedure**: Step-by-step protocol
5. **Data Collection**: Measurement methods and instruments
6. **Statistical Analysis**: Appropriate tests and significance levels
7. **Ethical Considerations**: Safety and compliance
8. **Expected Timeline**: Realistic schedule

Ensure the design is:
- Reproducible
- Statistically powered
- Ethically sound
- Practically feasible
```

**paper_writing_introduction.j2**
```jinja2
You are a scientific writer crafting an academic paper introduction.

Paper Context:
- Title: {{ title }}
- Research Question: {{ research_question }}
- Key Findings: {{ key_findings }}
- Target Journal: {{ journal }}

Write a compelling Introduction section that:

1. **Background** (2-3 paragraphs):
   - Establish the broader context
   - Review relevant literature
   - Highlight knowledge gaps

2. **Significance** (1 paragraph):
   - Explain why this research matters
   - Potential impact on the field

3. **Objectives** (1 paragraph):
   - Clearly state research objectives
   - Preview the approach

4. **Structure Preview** (optional):
   - Brief outline of the paper

Style Requirements:
- Academic tone
- Present tense for general facts, past tense for previous research
- Clear, concise language
- Appropriate citations ({{ citation_style }})
- Word count: {{ word_count }} words
```

## 💾 Caching Strategy

```python
from functools import lru_cache
import hashlib
import redis.asyncio as redis

class LLMCache:
    """Cache layer for LLM responses"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour default

    async def get(self, prompt: str, config: LLMConfig) -> Optional[str]:
        """Get cached response"""
        cache_key = self._generate_key(prompt, config)
        cached = await self.redis.get(cache_key)
        return cached.decode() if cached else None

    async def set(
        self,
        prompt: str,
        config: LLMConfig,
        response: str,
        ttl: Optional[int] = None
    ):
        """Cache response"""
        cache_key = self._generate_key(prompt, config)
        await self.redis.setex(
            cache_key,
            ttl or self.ttl,
            response
        )

    def _generate_key(self, prompt: str, config: LLMConfig) -> str:
        """Generate cache key"""
        key_data = f"{prompt}:{config.model}:{config.temperature}"
        return f"llm_cache:{hashlib.sha256(key_data.encode()).hexdigest()}"
```

## 📊 Usage Tracking

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class UsageMetrics:
    """Track LLM usage metrics"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    requests_by_task: Dict[TaskType, int] = field(default_factory=dict)
    tokens_by_provider: Dict[ModelProvider, int] = field(default_factory=dict)
    errors: int = 0

class UsageTracker:
    """Track and report LLM usage"""

    def __init__(self, metrics_store: Any):
        self.store = metrics_store
        self.metrics = UsageMetrics()

    async def record_request(
        self,
        request: LLMRequest,
        response: LLMResponse
    ):
        """Record request metrics"""
        self.metrics.total_requests += 1
        self.metrics.total_tokens += response.tokens_used
        self.metrics.total_cost += response.cost

        # By task type
        task_count = self.metrics.requests_by_task.get(request.task_type, 0)
        self.metrics.requests_by_task[request.task_type] = task_count + 1

        # By provider
        provider_tokens = self.metrics.tokens_by_provider.get(
            response.provider, 0
        )
        self.metrics.tokens_by_provider[response.provider] = (
            provider_tokens + response.tokens_used
        )

        # Persist to database
        await self.store.save_metrics({
            "timestamp": datetime.utcnow(),
            "task_type": request.task_type.value,
            "provider": response.provider.value,
            "model": response.model,
            "tokens": response.tokens_used,
            "cost": response.cost,
            "latency_ms": response.latency_ms
        })

    async def record_error(self, error: Exception):
        """Record error"""
        self.metrics.errors += 1
        await self.store.save_error({
            "timestamp": datetime.utcnow(),
            "error_type": type(error).__name__,
            "message": str(error)
        })

    def get_metrics(self) -> UsageMetrics:
        """Get current metrics"""
        return self.metrics
```

## 🔄 Complete Service Implementation

```python
class LLMService:
    """Main LLM service orchestrator"""

    def __init__(
        self,
        openai_key: str,
        anthropic_key: str,
        redis_client: redis.Redis,
        metrics_store: Any
    ):
        self.adapters = {
            ModelProvider.OPENAI: OpenAIAdapter(openai_key),
            ModelProvider.ANTHROPIC: AnthropicAdapter(anthropic_key)
        }
        self.prompt_manager = PromptManager("./prompts")
        self.cache = LLMCache(redis_client)
        self.usage_tracker = UsageTracker(metrics_store)

    async def complete(
        self,
        request: LLMRequest,
        use_cache: bool = True
    ) -> LLMResponse:
        """Generate completion with caching and fallback"""

        # Check cache
        if use_cache:
            cached_response = await self.cache.get(
                request.prompt,
                request.config
            )
            if cached_response:
                return LLMResponse(
                    content=cached_response,
                    model="cached",
                    provider=request.config.provider,
                    tokens_used=0,
                    cost=0.0,
                    latency_ms=0.0,
                    finish_reason="cached",
                    metadata={}
                )

        # Get adapter
        config = request.config or self._get_default_config(request.task_type)
        adapter = self.adapters.get(config.provider)

        if not adapter:
            raise ValueError(f"Unknown provider: {config.provider}")

        try:
            # Generate completion
            response = await adapter.complete(request)

            # Cache response
            if use_cache:
                await self.cache.set(
                    request.prompt,
                    config,
                    response.content
                )

            # Track usage
            await self.usage_tracker.record_request(request, response)

            return response

        except Exception as e:
            # Record error
            await self.usage_tracker.record_error(e)

            # Attempt fallback
            if config.provider == ModelProvider.OPENAI:
                fallback_config = LLMConfig(
                    provider=ModelProvider.ANTHROPIC,
                    model="claude-3-sonnet-20240229",
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
                request.config = fallback_config
                return await self.complete(request, use_cache=False)

            raise

    async def complete_with_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        task_type: TaskType,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion using template"""

        # Render prompt
        prompt = self.prompt_manager.render_prompt(template_name, context)

        # Validate
        if not self.prompt_manager.validate_prompt(prompt):
            raise ValueError("Prompt exceeds token limit")

        # Optimize
        prompt = self.prompt_manager.optimize_prompt(prompt)

        # Create request
        request = LLMRequest(
            prompt=prompt,
            task_type=task_type,
            config=config,
            context=context
        )

        return await self.complete(request)

    def _get_default_config(self, task_type: TaskType) -> LLMConfig:
        """Get default config for task type"""
        return self.adapters[ModelProvider.OPENAI]._get_default_config(
            task_type
        )
```

## 🧪 Testing Strategy

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_llm_service_completion():
    """Test basic completion"""
    service = LLMService(
        openai_key="test",
        anthropic_key="test",
        redis_client=AsyncMock(),
        metrics_store=AsyncMock()
    )

    request = LLMRequest(
        prompt="What is photosynthesis?",
        task_type=TaskType.LITERATURE_ANALYSIS
    )

    response = await service.complete(request)

    assert response.content
    assert response.tokens_used > 0
    assert response.cost >= 0

@pytest.mark.asyncio
async def test_llm_service_caching():
    """Test response caching"""
    service = LLMService(...)

    request = LLMRequest(...)

    # First call - should hit API
    response1 = await service.complete(request)

    # Second call - should use cache
    response2 = await service.complete(request)

    assert response2.finish_reason == "cached"
    assert response2.cost == 0.0

@pytest.mark.asyncio
async def test_llm_service_fallback():
    """Test provider fallback"""
    service = LLMService(...)

    # Mock OpenAI to fail
    service.adapters[ModelProvider.OPENAI].complete = AsyncMock(
        side_effect=Exception("API Error")
    )

    request = LLMRequest(
        prompt="Test",
        task_type=TaskType.HYPOTHESIS_GENERATION,
        config=LLMConfig(provider=ModelProvider.OPENAI, model="gpt-4")
    )

    # Should fallback to Anthropic
    response = await service.complete(request)

    assert response.provider == ModelProvider.ANTHROPIC
```

## 📈 Performance Considerations

### Token Optimization
- Prompt compression techniques
- Template caching
- Response truncation for summaries

### Cost Management
- Model selection by complexity
- Batch processing
- Intelligent caching
- Rate limiting

### Latency Optimization
- Connection pooling
- Async/await everywhere
- Streaming for long responses
- Regional API endpoints

## 🔐 Security

### API Key Management
- Environment variables
- Secrets manager integration
- Key rotation policies

### Rate Limiting
- Per-user limits
- Per-task-type limits
- Global rate limits

### Audit Logging
- All requests logged
- Cost tracking
- Error monitoring
