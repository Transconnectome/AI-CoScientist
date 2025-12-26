from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class TaskType(str, Enum):
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    LITERATURE_ANALYSIS = "literature_analysis"
    EXPERIMENT_DESIGN = "experiment_design"
    DATA_ANALYSIS = "data_analysis"
    PAPER_WRITING = "paper_writing"
    PEER_REVIEW = "peer_review"
    # Psychology-specific task types
    PSYCHOLOGY_RESEARCH = "psychology_research"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    COGNITIVE_EVALUATION = "cognitive_evaluation"
    DEVELOPMENTAL_ASSESSMENT = "developmental_assessment"
    NEUROPSYCHOLOGY_ANALYSIS = "neuropsychology_analysis"

class LLMConfig(BaseModel):
    provider: ModelProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    timeout: int = 60

class LLMRequest(BaseModel):
    prompt: str
    task_type: TaskType = TaskType.HYPOTHESIS_GENERATION
    system_message: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None # List of {"input": "...", "output": "..."}
    config: Optional[LLMConfig] = None

class LLMResponse(BaseModel):
    content: str
    model: str
    provider: ModelProvider
    tokens_used: int
    cost: float
    latency_ms: float
    finish_reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
