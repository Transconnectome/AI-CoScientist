from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class QualityScores:
    """Multi-dimensional quality scores"""
    overall: float
    novelty: Optional[float] = None
    rigor: Optional[float] = None
    clarity: Optional[float] = None
    significance: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of quality validation"""
    status: str  # APPROVED, APPROVED_WITH_CONDITIONS, REJECTED
    final_output: Any
    quality_scores: QualityScores
    iterations: int
    history: List[Dict] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityThresholds:
    """Quality gate thresholds"""
    minimums: Dict[str, float] = field(default_factory=lambda: {
        "overall": 0.7,
        "novelty": 0.6,
        "rigor": 0.7
    })
    targets: Dict[str, float] = field(default_factory=lambda: {
        "overall": 0.85,
        "novelty": 0.8,
        "rigor": 0.9
    })
