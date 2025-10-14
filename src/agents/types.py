from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    LITERATURE_SEARCH = "literature_search"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    PAPER_IMPROVEMENT = "paper_improvement"
    DOMAIN_VALIDATION = "domain_validation"
    QUALITY_ASSESSMENT = "quality_assessment"

@dataclass
class AgentTask:
    task_id: str
    task_type: TaskType
    description: str
    context_needed: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.context_needed is None:
            self.context_needed = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentResult:
    agent_id: str
    task_id: str
    output: Any
    confidence: float
    execution_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    supporting_evidence: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.supporting_evidence is None:
            self.supporting_evidence = []
        if self.metadata is None:
            self.metadata = {}
