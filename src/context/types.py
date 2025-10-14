# src/context/types.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

@dataclass
class Insight:
    """A piece of research insight"""
    content: str
    type: str  # finding, constraint, hypothesis, validation
    domains: List[str]
    score: float
    source_papers: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid4()))

@dataclass
class ResearchSession:
    """Tracks a research session"""
    id: str = field(default_factory=lambda: str(uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    task_description: str = ""
    insights: List[Insight] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
