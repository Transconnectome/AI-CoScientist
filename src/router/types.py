from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from src.agents.base import ResearchAgent

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class ResearchTask:
    description: str
    task_type: str
    prior_work: Optional[str] = None
    quality_target: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskProfile:
    """Analysis of a research task"""
    domains: List[str]
    complexity: ComplexityLevel
    task_type: str
    sub_tasks: List[Dict[str, Any]]
    required_expertise: List[str]
    quality_gates: List[str]
    context_dependencies: List[str]
    keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_llm_response(cls, response: str) -> 'TaskProfile':
        """Parse LLM response into TaskProfile"""
        import json
        try:
            data = json.loads(response)
            return cls(
                domains=data.get("domains", []),
                complexity=ComplexityLevel(data.get("complexity", "medium")),
                task_type=data.get("task_type", "unknown"),
                sub_tasks=data.get("sub_tasks", []),
                required_expertise=data.get("required_expertise", []),
                quality_gates=data.get("quality_gates", []),
                context_dependencies=data.get("context_dependencies", []),
                keywords=data.get("keywords", [])
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback if parsing fails
            return cls(
                domains=["general"],
                complexity=ComplexityLevel.MEDIUM,
                task_type="unknown",
                sub_tasks=[],
                required_expertise=[],
                quality_gates=[],
                context_dependencies=[]
            )


@dataclass
class AgentConfig:
    """Configuration for selected agent"""
    agent_id: str
    agent: 'ResearchAgent'  # Type hint
    match_score: float
    assigned_tasks: List[str] = field(default_factory=list)
    priority: int = 1
