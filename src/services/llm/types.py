"""LLM service types and enums."""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class TaskType(str, Enum):
    """Scientific research task types."""
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    LITERATURE_ANALYSIS = "literature_analysis"
    EXPERIMENT_DESIGN = "experiment_design"
    DATA_ANALYSIS = "data_analysis"
    PAPER_WRITING = "paper_writing"
    PEER_REVIEW = "peer_review"


@dataclass
class LLMConfig:
    """Configuration for LLM requests."""
    provider: ModelProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    timeout: int = 60


@dataclass
class LLMRequest:
    """LLM request payload."""
    prompt: str
    task_type: TaskType
    config: Optional[LLMConfig] = None
    context: Optional[Dict[str, Any]] = None
    system_message: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None


@dataclass
class LLMResponse:
    """LLM response payload."""
    content: str
    model: str
    provider: ModelProvider
    tokens_used: int
    cost: float
    latency_ms: float
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
