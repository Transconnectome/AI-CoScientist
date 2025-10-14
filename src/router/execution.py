# src/router/execution.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.agents.types import AgentResult

@dataclass
class ExecutionResult:
    """Result of task execution"""
    status: str  # success, partial_success, failed
    agent_results: List[AgentResult]
    quality_score: float
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
