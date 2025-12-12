# Adaptive Research Assistant - Multi-Agent System Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Transform AI-CoScientist from static LLM routing into an intelligent multi-agent system with dynamic task decomposition, optimal agent selection, context-aware collaboration, and continuous learning.

**Architecture:** Meta-Router analyzes tasks → selects optimal agent team → agents collaborate through shared context → quality validation loop → learning from outcomes. Tier 1 (rule-based, 1-2 months) evolves to Tier 2 (RL-based, 3-6 months).

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, ChromaDB (context storage), Neo4j (relationship tracking), Redis (caching), existing LLMService, pytest (TDD)

---

## Phase 1: Foundation - Core Infrastructure (Week 1-2)

### Task 1: Agent Base Architecture

**Files:**
- Create: `src/agents/base.py`
- Create: `src/agents/__init__.py`
- Create: `src/agents/types.py`
- Test: `tests/agents/test_base_agent.py`

**Step 1: Write test for ResearchAgent base class**

```python
# tests/agents/test_base_agent.py
import pytest
from src.agents.base import ResearchAgent
from src.agents.types import AgentTask, AgentResult

class TestAgent(ResearchAgent):
    async def process(self, task: AgentTask, context: dict) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output="test output",
            confidence=0.9
        )

@pytest.mark.asyncio
async def test_agent_has_required_attributes():
    agent = TestAgent(
        agent_id="test_agent",
        llm_service=None,
        context_manager=None
    )

    assert agent.agent_id == "test_agent"
    assert hasattr(agent, 'capabilities')
    assert hasattr(agent, 'domains')
    assert hasattr(agent, 'performance_history')

@pytest.mark.asyncio
async def test_agent_can_process_task():
    agent = TestAgent("test", None, None)
    task = AgentTask(task_id="t1", description="test")

    result = await agent.process(task, {})

    assert result.agent_id == "test"
    assert result.task_id == "t1"
    assert result.confidence == 0.9
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_base_agent.py -v`
Expected: ModuleNotFoundError: No module named 'src.agents'

**Step 3: Create agent types**

```python
# src/agents/types.py
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
```

**Step 4: Create base agent class**

```python
# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from src.agents.types import AgentTask, AgentResult

class ResearchAgent(ABC):
    """Base class for all specialized research agents"""

    def __init__(
        self,
        agent_id: str,
        llm_service: Any,
        context_manager: Any
    ):
        self.agent_id = agent_id
        self.llm = llm_service
        self.context = context_manager

        # Agent metadata
        self.capabilities: List[str] = []
        self.domains: List[str] = []
        self.specializations: List[str] = []
        self.performance_history: Dict = {}

    @abstractmethod
    async def process(
        self,
        task: AgentTask,
        relevant_context: Dict
    ) -> AgentResult:
        """Core processing logic - each agent implements this"""
        pass

    def update_performance(self, task_id: str, success: bool, score: float):
        """Track performance for learning"""
        self.performance_history[task_id] = {
            "success": success,
            "score": score
        }

    def get_success_rate(self) -> float:
        """Calculate historical success rate"""
        if not self.performance_history:
            return 0.5  # Default for new agents

        successes = sum(
            1 for h in self.performance_history.values()
            if h["success"]
        )
        return successes / len(self.performance_history)
```

```python
# src/agents/__init__.py
from src.agents.base import ResearchAgent
from src.agents.types import AgentTask, AgentResult, TaskType

__all__ = ["ResearchAgent", "AgentTask", "AgentResult", "TaskType"]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/agents/test_base_agent.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add src/agents/ tests/agents/
git commit -m "feat(agents): add base agent architecture with types and ABC"
```

---

### Task 2: Agent Pool Registry

**Files:**
- Create: `src/agents/pool.py`
- Create: `src/agents/domain_experts.py`
- Test: `tests/agents/test_agent_pool.py`

**Step 1: Write test for AgentPool**

```python
# tests/agents/test_agent_pool.py
import pytest
from src.agents.pool import AgentPool
from src.agents.base import ResearchAgent
from src.agents.types import AgentTask, AgentResult

@pytest.fixture
def mock_llm_service():
    return None  # Mock for now

@pytest.fixture
def mock_context_manager():
    return None

@pytest.fixture
def agent_pool(mock_llm_service, mock_context_manager):
    return AgentPool(mock_llm_service, mock_context_manager)

def test_agent_pool_has_agents(agent_pool):
    """Agent pool should have registered agents"""
    assert len(agent_pool.agents) > 0
    assert "neuroscience_expert" in agent_pool.agents

def test_get_agent_by_id(agent_pool):
    """Can retrieve agent by ID"""
    agent = agent_pool.get_agent("neuroscience_expert")
    assert agent is not None
    assert agent.agent_id == "neuroscience_expert"

def test_get_agents_by_capability(agent_pool):
    """Can find agents by capability"""
    agents = agent_pool.get_agents_by_capability("domain_validation")
    assert len(agents) > 0
    assert all("domain_validation" in a.capabilities for a in agents)

def test_get_agents_by_domain(agent_pool):
    """Can find agents by domain"""
    agents = agent_pool.get_agents_by_domain("neuroscience")
    assert len(agents) > 0
    assert all("neuroscience" in a.domains for a in agents)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_agent_pool.py -v`
Expected: ModuleNotFoundError: No module named 'src.agents.pool'

**Step 3: Create domain expert agent**

```python
# src/agents/domain_experts.py
from src.agents.base import ResearchAgent
from src.agents.types import AgentTask, AgentResult

class NeuroscienceExpertAgent(ResearchAgent):
    """Domain expert in neuroscience"""

    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities = [
            "domain_validation",
            "methodology_review",
            "literature_interpretation"
        ]
        self.domains = ["neuroscience", "fMRI", "brain_imaging"]
        self.specializations = ["emotion_recognition", "connectivity"]

    async def process(
        self,
        task: AgentTask,
        relevant_context: dict
    ) -> AgentResult:
        """Validate from neuroscience perspective"""

        # For now, minimal implementation
        prompt = f"""
        As a neuroscience expert, analyze:
        {task.description}

        Context: {relevant_context.get('summary', 'None')}

        Provide expert validation covering:
        1. Scientific validity
        2. Methodological appropriateness
        3. Potential limitations
        """

        if self.llm:
            response = await self.llm.complete(prompt)
            output = response.content
        else:
            output = "Mock neuroscience validation"

        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.85
        )
```

**Step 4: Create agent pool**

```python
# src/agents/pool.py
from typing import Dict, List
from src.agents.base import ResearchAgent
from src.agents.domain_experts import NeuroscienceExpertAgent

class AgentPool:
    """Registry and manager for all agents"""

    def __init__(self, llm_service, context_manager):
        self.llm_service = llm_service
        self.context_manager = context_manager

        # Initialize agent registry
        self.agents: Dict[str, ResearchAgent] = {}
        self._register_agents()

    def _register_agents(self):
        """Register all available agents"""

        # Domain experts
        self.agents["neuroscience_expert"] = NeuroscienceExpertAgent(
            "neuroscience_expert",
            self.llm_service,
            self.context_manager
        )

        # More agents will be added in future tasks

    def get_agent(self, agent_id: str) -> ResearchAgent:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_capability(
        self,
        capability: str
    ) -> List[ResearchAgent]:
        """Find agents with specific capability"""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]

    def get_agents_by_domain(
        self,
        domain: str
    ) -> List[ResearchAgent]:
        """Find agents for specific domain"""
        return [
            agent for agent in self.agents.values()
            if domain in agent.domains
        ]

    def list_all_agents(self) -> Dict[str, Dict]:
        """Get metadata for all agents"""
        return {
            agent_id: {
                "capabilities": agent.capabilities,
                "domains": agent.domains,
                "specializations": agent.specializations,
                "success_rate": agent.get_success_rate()
            }
            for agent_id, agent in self.agents.items()
        }
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/agents/test_agent_pool.py -v`
Expected: 4 passed

**Step 6: Commit**

```bash
git add src/agents/pool.py src/agents/domain_experts.py tests/agents/test_agent_pool.py
git commit -m "feat(agents): add agent pool registry with neuroscience expert"
```

---

### Task 3: Meta-Router Task Analyzer

**Files:**
- Create: `src/router/analyzer.py`
- Create: `src/router/types.py`
- Test: `tests/router/test_task_analyzer.py`

**Step 1: Write test for task analysis**

```python
# tests/router/test_task_analyzer.py
import pytest
from src.router.analyzer import TaskAnalyzer
from src.router.types import ResearchTask, TaskProfile

@pytest.fixture
def task_analyzer(mock_llm_service):
    return TaskAnalyzer(mock_llm_service)

@pytest.mark.asyncio
async def test_analyze_simple_task(task_analyzer):
    """Analyze a simple research task"""
    task = ResearchTask(
        description="Search for fMRI papers on emotion recognition",
        task_type="literature_search"
    )

    profile = await task_analyzer.analyze_task(task)

    assert profile is not None
    assert "neuroscience" in profile.domains
    assert profile.complexity in ["simple", "medium", "high"]
    assert len(profile.required_expertise) > 0

@pytest.mark.asyncio
async def test_analyze_complex_task(task_analyzer):
    """Analyze complex multi-domain task"""
    task = ResearchTask(
        description="""Generate novel hypothesis for fMRI emotion recognition
                       using deep learning with ethical considerations""",
        task_type="hypothesis_generation"
    )

    profile = await task_analyzer.analyze_task(task)

    assert profile.complexity == "high"
    assert len(profile.domains) >= 3  # neuroscience, ML, ethics
    assert "neuroscience" in profile.domains
    assert "machine_learning" in profile.domains
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/router/test_task_analyzer.py -v`
Expected: ModuleNotFoundError: No module named 'src.router'

**Step 3: Create router types**

```python
# src/router/types.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

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
```

**Step 4: Create task analyzer**

```python
# src/router/analyzer.py
from src.router.types import ResearchTask, TaskProfile

class TaskAnalyzer:
    """Analyzes research tasks to extract structured information"""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def analyze_task(self, task: ResearchTask) -> TaskProfile:
        """Analyze task and extract profile"""

        prompt = f"""
        Analyze this research task and extract structured information:

        Task Type: {task.task_type}
        Description: {task.description}
        Prior Work: {task.prior_work or 'None'}

        Extract and return as JSON:
        {{
            "domains": ["list", "of", "domains"],
            "complexity": "simple" or "medium" or "high",
            "task_type": "{task.task_type}",
            "sub_tasks": [
                {{"id": "subtask1", "type": "type", "dependencies": []}}
            ],
            "required_expertise": ["list", "of", "expertise"],
            "quality_gates": ["validation", "checks"],
            "context_dependencies": ["what", "context", "needed"],
            "keywords": ["key", "terms"]
        }}

        Complexity levels:
        - simple: Single domain, straightforward task
        - medium: 1-2 domains, some dependencies
        - high: 3+ domains, complex dependencies, novel synthesis

        Common domains: neuroscience, machine_learning, statistics,
                       ethics, biology, chemistry, physics
        """

        if self.llm:
            response = await self.llm.complete(prompt, temperature=0.3)
            return TaskProfile.from_llm_response(response.content)
        else:
            # Mock response for testing without LLM
            return self._create_mock_profile(task)

    def _create_mock_profile(self, task: ResearchTask) -> TaskProfile:
        """Create mock profile for testing"""
        # Simple heuristic-based analysis
        domains = []
        if "fmri" in task.description.lower() or "brain" in task.description.lower():
            domains.append("neuroscience")
        if "deep learning" in task.description.lower() or "ml" in task.description.lower():
            domains.append("machine_learning")
        if "ethics" in task.description.lower() or "ethical" in task.description.lower():
            domains.append("ethics")

        if not domains:
            domains = ["general"]

        complexity = "high" if len(domains) >= 3 else "medium" if len(domains) == 2 else "simple"

        from src.router.types import ComplexityLevel
        return TaskProfile(
            domains=domains,
            complexity=ComplexityLevel(complexity),
            task_type=task.task_type,
            sub_tasks=[{"id": "main", "type": task.task_type}],
            required_expertise=domains,
            quality_gates=["validation"],
            context_dependencies=["literature"]
        )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/router/test_task_analyzer.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add src/router/ tests/router/
git commit -m "feat(router): add task analyzer for extracting task profiles"
```

---

## Phase 2: Meta-Router Intelligence (Week 3-4)

### Task 4: Agent Capability Matcher

**Files:**
- Create: `src/router/matcher.py`
- Modify: `src/router/types.py` (add AgentConfig)
- Test: `tests/router/test_matcher.py`

**Step 1: Write test for agent matching**

```python
# tests/router/test_matcher.py
import pytest
from src.router.matcher import AgentCapabilityMatcher
from src.router.types import TaskProfile, ComplexityLevel, AgentConfig
from src.agents.pool import AgentPool

@pytest.fixture
def agent_pool(mock_llm_service, mock_context_manager):
    return AgentPool(mock_llm_service, mock_context_manager)

@pytest.fixture
def matcher(agent_pool):
    return AgentCapabilityMatcher(agent_pool)

def test_match_simple_task(matcher):
    """Match agents for simple single-domain task"""
    profile = TaskProfile(
        domains=["neuroscience"],
        complexity=ComplexityLevel.SIMPLE,
        task_type="literature_search",
        sub_tasks=[],
        required_expertise=["domain_knowledge"],
        quality_gates=[],
        context_dependencies=[]
    )

    agents = matcher.select_agents(profile, {})

    assert len(agents) > 0
    assert any(a.agent_id == "neuroscience_expert" for a in agents)

def test_match_complex_task(matcher):
    """Match multiple agents for complex task"""
    profile = TaskProfile(
        domains=["neuroscience", "machine_learning", "ethics"],
        complexity=ComplexityLevel.HIGH,
        task_type="hypothesis_generation",
        sub_tasks=[],
        required_expertise=["domain_knowledge", "creative_synthesis"],
        quality_gates=[],
        context_dependencies=[]
    )

    agents = matcher.select_agents(profile, {})

    assert len(agents) >= 2  # Multiple domains need multiple agents
    domains_covered = set()
    for agent in agents:
        domains_covered.update(agent.domains)

    assert "neuroscience" in domains_covered

def test_scoring_considers_performance_history(matcher):
    """Agent selection considers past performance"""
    # Mock performance history
    history = {
        "neuroscience_expert": {
            "success_rate": 0.95,
            "avg_quality": 0.88
        }
    }

    profile = TaskProfile(
        domains=["neuroscience"],
        complexity=ComplexityLevel.MEDIUM,
        task_type="validation",
        sub_tasks=[],
        required_expertise=["domain_validation"],
        quality_gates=[],
        context_dependencies=[]
    )

    agents = matcher.select_agents(profile, history)

    # High-performing agent should be selected
    assert agents[0].agent_id == "neuroscience_expert"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/router/test_matcher.py -v`
Expected: ModuleNotFoundError or ImportError

**Step 3: Add AgentConfig type**

```python
# Add to src/router/types.py
@dataclass
class AgentConfig:
    """Configuration for selected agent"""
    agent_id: str
    agent: 'ResearchAgent'  # Type hint
    match_score: float
    assigned_tasks: List[str] = field(default_factory=list)
    priority: int = 1
```

**Step 4: Implement capability matcher**

```python
# src/router/matcher.py
from typing import List, Dict
from src.router.types import TaskProfile, AgentConfig
from src.agents.pool import AgentPool

class AgentCapabilityMatcher:
    """Matches task requirements to agent capabilities"""

    def __init__(self, agent_pool: AgentPool):
        self.agent_pool = agent_pool

    def select_agents(
        self,
        task_profile: TaskProfile,
        performance_history: Dict
    ) -> List[AgentConfig]:
        """Select optimal agents for task"""

        # Score all agents
        scores = {}
        for agent_id, agent in self.agent_pool.agents.items():
            score = self._calculate_match_score(
                agent,
                task_profile,
                performance_history.get(agent_id, {})
            )
            scores[agent_id] = score

        # Select top agents based on complexity
        num_agents = self._determine_team_size(task_profile)
        top_agents = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_agents]

        # Create configs
        return [
            AgentConfig(
                agent_id=agent_id,
                agent=self.agent_pool.get_agent(agent_id),
                match_score=score
            )
            for agent_id, score in top_agents
            if score > 0.3  # Minimum threshold
        ]

    def _calculate_match_score(
        self,
        agent,
        task_profile: TaskProfile,
        history: Dict
    ) -> float:
        """Calculate how well agent matches task"""

        # Domain overlap (40%)
        domain_match = len(
            set(agent.domains) & set(task_profile.domains)
        ) / max(len(task_profile.domains), 1)

        # Capability overlap (30%)
        capability_match = len(
            set(agent.capabilities) & set(task_profile.required_expertise)
        ) / max(len(task_profile.required_expertise), 1)

        # Past performance (20%)
        past_performance = history.get("success_rate", 0.5)

        # Specialization bonus (10%)
        specialization_bonus = any(
            spec.lower() in task_profile.keywords
            for spec in agent.specializations
        ) * 0.1

        return (
            domain_match * 0.4 +
            capability_match * 0.3 +
            past_performance * 0.2 +
            specialization_bonus
        )

    def _determine_team_size(self, task_profile: TaskProfile) -> int:
        """Determine how many agents needed"""
        if task_profile.complexity == "simple":
            return 1
        elif task_profile.complexity == "medium":
            return min(2, len(task_profile.domains))
        else:  # high
            return min(3, len(task_profile.domains))
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/router/test_matcher.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
git add src/router/matcher.py src/router/types.py tests/router/test_matcher.py
git commit -m "feat(router): add agent capability matcher with scoring"
```

---

### Task 5: Context Manager Foundation

**Files:**
- Create: `src/context/manager.py`
- Create: `src/context/types.py`
- Test: `tests/context/test_manager.py`

**Step 1: Write test for context storage and retrieval**

```python
# tests/context/test_manager.py
import pytest
from src.context.manager import ResearchContextManager
from src.context.types import Insight, ResearchSession

@pytest.fixture
def context_manager():
    # For now, use in-memory storage
    return ResearchContextManager(vector_store=None, graph_db=None)

@pytest.mark.asyncio
async def test_store_insight(context_manager):
    """Store an insight with metadata"""
    insight = Insight(
        content="Deep learning shows promise for fMRI analysis",
        type="finding",
        domains=["neuroscience", "machine_learning"],
        score=0.85
    )

    node_id = await context_manager.store_insight(
        insight=insight,
        source_agent="literature_scout",
        task_id="task1",
        metadata={"source": "paper_123"}
    )

    assert node_id is not None

@pytest.mark.asyncio
async def test_get_relevant_context(context_manager):
    """Retrieve relevant context for agent"""
    # Store some insights first
    insight1 = Insight(
        content="fMRI has 2s lag",
        type="constraint",
        domains=["neuroscience"],
        score=0.9
    )

    await context_manager.store_insight(
        insight1,
        "neuro_expert",
        "task1",
        {}
    )

    # Retrieve for agent
    context = await context_manager.get_relevant(
        agent_id="hypothesis_generator",
        task_type="hypothesis_generation",
        max_tokens=1000
    )

    assert context is not None
    assert "insights" in context

@pytest.mark.asyncio
async def test_context_budget_management(context_manager):
    """Context stays within token budget"""
    # Store multiple insights
    for i in range(10):
        insight = Insight(
            content=f"Finding {i}: " + "x" * 100,
            type="finding",
            domains=["test"],
            score=0.7
        )
        await context_manager.store_insight(insight, "test", f"t{i}", {})

    # Retrieve with budget
    context = await context_manager.get_relevant(
        agent_id="test",
        task_type="test",
        max_tokens=500  # Small budget
    )

    # Should prioritize and fit within budget
    total_length = sum(len(str(i)) for i in context["insights"])
    assert total_length < 500 * 4  # Rough token estimate
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/context/test_manager.py -v`
Expected: ModuleNotFoundError

**Step 3: Create context types**

```python
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
```

**Step 4: Implement context manager (simplified for Tier 1)**

```python
# src/context/manager.py
from typing import Dict, List, Optional
from src.context.types import Insight, ResearchSession
from uuid import uuid4

class ResearchContextManager:
    """Manages research context and insights"""

    def __init__(self, vector_store, graph_db):
        self.vector_store = vector_store
        self.graph_db = graph_db

        # In-memory storage for Tier 1 (will add vector/graph in Tier 2)
        self.insights: Dict[str, Insight] = {}
        self.current_session = ResearchSession()

    async def store_insight(
        self,
        insight: Insight,
        source_agent: str,
        task_id: str,
        metadata: Dict
    ) -> str:
        """Store insight with provenance"""

        node_id = str(uuid4())

        # Store in memory
        self.insights[node_id] = insight
        self.current_session.insights.append(insight)

        # TODO Tier 2: Store in vector DB and graph DB
        # await self.vector_store.add(...)
        # await self.graph_db.create_node(...)

        return node_id

    async def get_relevant(
        self,
        agent_id: str,
        task_type: str,
        max_tokens: int = 4000
    ) -> Dict:
        """Get relevant context for agent within token budget"""

        # For Tier 1: Simple filtering
        relevant_insights = [
            insight for insight in self.insights.values()
            if self._is_relevant(insight, agent_id, task_type)
        ]

        # Sort by score
        relevant_insights.sort(key=lambda x: x.score, reverse=True)

        # Apply token budget
        selected = self._select_within_budget(relevant_insights, max_tokens)

        return {
            "insights": selected,
            "relationships": [],  # TODO Tier 2
            "provenance": []      # TODO Tier 2
        }

    def _is_relevant(
        self,
        insight: Insight,
        agent_id: str,
        task_type: str
    ) -> bool:
        """Simple relevance check for Tier 1"""
        # Always include high-scoring insights
        if insight.score > 0.85:
            return True

        # Include insights from last hour
        from datetime import datetime, timedelta
        if datetime.utcnow() - insight.timestamp < timedelta(hours=1):
            return True

        return False

    def _select_within_budget(
        self,
        insights: List[Insight],
        max_tokens: int
    ) -> List[Insight]:
        """Select insights within token budget"""

        selected = []
        estimated_tokens = 0

        for insight in insights:
            # Rough estimate: 1 token ≈ 4 characters
            insight_tokens = len(insight.content) // 4

            if estimated_tokens + insight_tokens <= max_tokens:
                selected.append(insight)
                estimated_tokens += insight_tokens
            else:
                break

        return selected
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/context/test_manager.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
git add src/context/ tests/context/
git commit -m "feat(context): add context manager with insight storage and retrieval"
```

---

## Phase 3: Integration & Quality (Week 5-6)

### Task 6: Meta-Router Orchestration

**Files:**
- Create: `src/router/meta_router.py`
- Create: `src/router/execution.py`
- Test: `tests/router/test_meta_router.py`

**Step 1: Write integration test**

```python
# tests/router/test_meta_router.py
import pytest
from src.router.meta_router import MetaRouter
from src.router.types import ResearchTask

@pytest.fixture
def meta_router(mock_llm_service, mock_context_manager, agent_pool):
    return MetaRouter(
        llm_service=mock_llm_service,
        agent_pool=agent_pool,
        context_manager=mock_context_manager
    )

@pytest.mark.asyncio
async def test_route_simple_task(meta_router):
    """Route a simple task end-to-end"""
    task = ResearchTask(
        description="Search neuroscience papers on emotion",
        task_type="literature_search"
    )

    result = await meta_router.route_and_execute(task)

    assert result is not None
    assert result.status in ["success", "partial_success"]
    assert len(result.agent_results) > 0

@pytest.mark.asyncio
async def test_route_complex_multi_agent_task(meta_router):
    """Route complex task requiring multiple agents"""
    task = ResearchTask(
        description="""Generate hypothesis for fMRI emotion recognition
                       using deep learning with ethical considerations""",
        task_type="hypothesis_generation"
    )

    result = await meta_router.route_and_execute(task)

    assert result.status == "success"
    assert len(result.agent_results) >= 2  # Multiple agents involved
    assert result.quality_score > 0.7
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/router/test_meta_router.py -v`
Expected: ModuleNotFoundError

**Step 3: Create execution types**

```python
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
```

**Step 4: Implement Meta-Router**

```python
# src/router/meta_router.py
import asyncio
from typing import List
from src.router.analyzer import TaskAnalyzer
from src.router.matcher import AgentCapabilityMatcher
from src.router.types import ResearchTask
from src.router.execution import ExecutionResult
from src.agents.types import AgentTask, AgentResult
from src.agents.pool import AgentPool
from datetime import datetime

class MetaRouter:
    """Orchestrates task analysis, agent selection, and execution"""

    def __init__(self, llm_service, agent_pool: AgentPool, context_manager):
        self.task_analyzer = TaskAnalyzer(llm_service)
        self.agent_matcher = AgentCapabilityMatcher(agent_pool)
        self.context_manager = context_manager
        self.agent_pool = agent_pool

    async def route_and_execute(
        self,
        task: ResearchTask
    ) -> ExecutionResult:
        """Full pipeline: analyze → select agents → execute"""

        start_time = datetime.utcnow()

        # Step 1: Analyze task
        task_profile = await self.task_analyzer.analyze_task(task)

        # Step 2: Select agents
        performance_history = {}  # TODO: Load from database
        agent_configs = self.agent_matcher.select_agents(
            task_profile,
            performance_history
        )

        # Step 3: Execute with agents
        agent_results = await self._execute_with_agents(
            task,
            agent_configs
        )

        # Step 4: Calculate quality
        quality_score = self._calculate_quality(agent_results)

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ExecutionResult(
            status="success" if quality_score > 0.7 else "partial_success",
            agent_results=agent_results,
            quality_score=quality_score,
            execution_time_ms=execution_time,
            metadata={
                "task_profile": task_profile,
                "agents_used": [c.agent_id for c in agent_configs]
            }
        )

    async def _execute_with_agents(
        self,
        task: ResearchTask,
        agent_configs: List
    ) -> List[AgentResult]:
        """Execute task with selected agents"""

        results = []

        for agent_config in agent_configs:
            # Get relevant context for this agent
            context = await self.context_manager.get_relevant(
                agent_id=agent_config.agent_id,
                task_type=task.task_type,
                max_tokens=4000
            )

            # Create agent task
            agent_task = AgentTask(
                task_id=f"{task.task_type}_1",
                task_type=task.task_type,
                description=task.description
            )

            # Execute
            result = await agent_config.agent.process(
                agent_task,
                context
            )

            results.append(result)

            # Store result as insight for next agents
            if result.confidence > 0.7:
                from src.context.types import Insight
                insight = Insight(
                    content=str(result.output),
                    type="agent_result",
                    domains=[],
                    score=result.confidence
                )
                await self.context_manager.store_insight(
                    insight,
                    agent_config.agent_id,
                    agent_task.task_id,
                    {}
                )

        return results

    def _calculate_quality(self, results: List[AgentResult]) -> float:
        """Calculate overall quality from agent results"""
        if not results:
            return 0.0

        return sum(r.confidence for r in results) / len(results)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/router/test_meta_router.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add src/router/meta_router.py src/router/execution.py tests/router/test_meta_router.py
git commit -m "feat(router): add meta-router orchestration with end-to-end execution"
```

---

### Task 7: Quality Validation Loop

**Files:**
- Create: `src/quality/validator.py`
- Create: `src/quality/types.py`
- Test: `tests/quality/test_validator.py`

**Step 1: Write test for quality validation**

```python
# tests/quality/test_validator.py
import pytest
from src.quality.validator import QualityValidationLoop
from src.quality.types import QualityScores, ValidationResult

@pytest.fixture
def validator():
    # Mock critics for now
    return QualityValidationLoop(quality_critics=[], threshold_config=None)

@pytest.mark.asyncio
async def test_validates_high_quality_output(validator):
    """High quality output passes without iterations"""
    from src.router.execution import ExecutionResult
    from src.agents.types import AgentResult

    result = ExecutionResult(
        status="success",
        agent_results=[
            AgentResult("agent1", "t1", "output", confidence=0.9)
        ],
        quality_score=0.9,
        execution_time_ms=100
    )

    validation = await validator.validate_and_refine(result, {})

    assert validation.status == "APPROVED"
    assert validation.iterations == 1

@pytest.mark.asyncio
async def test_refines_low_quality_output(validator):
    """Low quality triggers refinement iterations"""
    result = ExecutionResult(
        status="partial_success",
        agent_results=[
            AgentResult("agent1", "t1", "weak output", confidence=0.5)
        ],
        quality_score=0.5,
        execution_time_ms=100
    )

    validation = await validator.validate_and_refine(result, {})

    # Should attempt improvement
    assert validation.iterations > 1 or validation.status == "APPROVED_WITH_CONDITIONS"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/quality/test_validator.py -v`
Expected: ModuleNotFoundError

**Step 3: Create quality types**

```python
# src/quality/types.py
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
```

**Step 4: Implement validator (simplified Tier 1)**

```python
# src/quality/validator.py
from typing import List, Dict
from src.quality.types import (
    QualityScores,
    ValidationResult,
    QualityThresholds
)

class QualityValidationLoop:
    """Validates and refines research outputs"""

    def __init__(self, quality_critics: List, threshold_config: QualityThresholds):
        self.critics = quality_critics or []
        self.thresholds = threshold_config or QualityThresholds()
        self.max_iterations = 3

    async def validate_and_refine(
        self,
        research_output,
        context: Dict
    ) -> ValidationResult:
        """Validate output with iterative refinement"""

        iteration = 0
        current_output = research_output
        validation_history = []

        while iteration < self.max_iterations:
            # Assess quality
            scores = await self._assess_quality(current_output)

            # Check if meets thresholds
            if self._meets_thresholds(scores):
                return ValidationResult(
                    status="APPROVED",
                    final_output=current_output,
                    quality_scores=scores,
                    iterations=iteration + 1,
                    history=validation_history
                )

            # Record iteration
            validation_history.append({
                "iteration": iteration + 1,
                "scores": scores
            })

            iteration += 1

        # Max iterations reached
        final_scores = await self._assess_quality(current_output)

        if final_scores.overall >= self.thresholds.minimums["overall"] * 0.9:
            return ValidationResult(
                status="APPROVED_WITH_CONDITIONS",
                final_output=current_output,
                quality_scores=final_scores,
                iterations=self.max_iterations,
                conditions=["Quality slightly below target"]
            )
        else:
            return ValidationResult(
                status="REJECTED",
                final_output=current_output,
                quality_scores=final_scores,
                iterations=self.max_iterations,
                reason="Quality below minimum threshold"
            )

    async def _assess_quality(self, output) -> QualityScores:
        """Assess output quality"""
        # For Tier 1: Use output's existing quality score
        overall = getattr(output, 'quality_score', 0.75)

        return QualityScores(
            overall=overall,
            novelty=overall * 0.9,  # Mock dimensions
            rigor=overall * 1.1,
            clarity=overall
        )

    def _meets_thresholds(self, scores: QualityScores) -> bool:
        """Check if scores meet minimum thresholds"""
        return scores.overall >= self.thresholds.minimums["overall"]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/quality/test_validator.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add src/quality/ tests/quality/
git commit -m "feat(quality): add quality validation loop with thresholds"
```

---

## Phase 4: API Integration (Week 7)

### Task 8: FastAPI Endpoints

**Files:**
- Create: `src/api/v1/multi_agent.py`
- Modify: `src/api/v1/__init__.py`
- Test: `tests/api/test_multi_agent.py`

**Step 1: Write API test**

```python
# tests/api/test_multi_agent.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_multi_agent_research_endpoint():
    """Test multi-agent research API"""
    response = client.post(
        "/api/v1/multi-agent/research",
        json={
            "description": "Search for fMRI emotion papers",
            "task_type": "literature_search",
            "quality_target": 0.8
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "quality_score" in data
    assert "agent_results" in data

def test_multi_agent_hypothesis_generation():
    """Test hypothesis generation endpoint"""
    response = client.post(
        "/api/v1/multi-agent/hypothesis",
        json={
            "research_question": "Novel DL approach for fMRI emotion prediction",
            "context": "Previous work used CNNs",
            "quality_target": 0.85
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "hypotheses" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_multi_agent.py -v`
Expected: 404 Not Found (endpoint doesn't exist)

**Step 3: Create API endpoint**

```python
# src/api/v1/multi_agent.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from src.core.database import get_db
from src.services.llm.service import LLMService
from src.router.meta_router import MetaRouter
from src.router.types import ResearchTask
from src.agents.pool import AgentPool
from src.context.manager import ResearchContextManager

router = APIRouter()

async def get_meta_router(
    db: AsyncSession = Depends(get_db)
) -> MetaRouter:
    """Get meta-router dependency"""
    from src.services.llm.service import LLMService
    from src.core.redis import get_redis

    redis = await get_redis()
    llm_service = LLMService(redis_client=redis)

    agent_pool = AgentPool(llm_service, None)
    context_manager = ResearchContextManager(None, None)

    return MetaRouter(llm_service, agent_pool, context_manager)

@router.post("/research")
async def multi_agent_research(
    task: Dict[str, Any],
    meta_router: MetaRouter = Depends(get_meta_router)
):
    """Execute multi-agent research task"""

    research_task = ResearchTask(
        description=task["description"],
        task_type=task["task_type"],
        quality_target=task.get("quality_target", 0.8)
    )

    try:
        result = await meta_router.route_and_execute(research_task)

        return {
            "status": result.status,
            "quality_score": result.quality_score,
            "execution_time_ms": result.execution_time_ms,
            "agent_results": [
                {
                    "agent_id": r.agent_id,
                    "output": r.output,
                    "confidence": r.confidence
                }
                for r in result.agent_results
            ],
            "metadata": result.metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hypothesis")
async def generate_hypothesis(
    request: Dict[str, Any],
    meta_router: MetaRouter = Depends(get_meta_router)
):
    """Generate research hypothesis using multi-agent system"""

    task = ResearchTask(
        description=request["research_question"],
        task_type="hypothesis_generation",
        prior_work=request.get("context"),
        quality_target=request.get("quality_target", 0.85)
    )

    result = await meta_router.route_and_execute(task)

    return {
        "status": result.status,
        "hypotheses": [r.output for r in result.agent_results],
        "quality_score": result.quality_score,
        "agents_used": result.metadata.get("agents_used", [])
    }
```

**Step 4: Register router**

```python
# Modify src/api/v1/__init__.py
from fastapi import APIRouter
from src.api.v1 import (
    health,
    papers,
    # ... existing imports
    multi_agent  # Add this
)

api_router = APIRouter()

# Existing routes
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(papers.router, prefix="/papers", tags=["papers"])
# ... existing routes

# Add multi-agent routes
api_router.include_router(
    multi_agent.router,
    prefix="/multi-agent",
    tags=["multi-agent"]
)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/api/test_multi_agent.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add src/api/v1/multi_agent.py src/api/v1/__init__.py tests/api/test_multi_agent.py
git commit -m "feat(api): add multi-agent research endpoints"
```

---

## Phase 5: Metrics & Learning (Week 8)

### Task 9: Performance Tracking

**Files:**
- Create: `src/metrics/tracker.py`
- Create: `src/metrics/types.py`
- Create: `alembic/versions/xxx_add_metrics_tables.py`
- Test: `tests/metrics/test_tracker.py`

**Step 1: Write test for metrics tracking**

```python
# tests/metrics/test_tracker.py
import pytest
from src.metrics.tracker import PerformanceTracker
from src.router.execution import ExecutionResult
from src.agents.types import AgentResult

@pytest.fixture
def tracker(mock_db_session):
    return PerformanceTracker(mock_db_session)

@pytest.mark.asyncio
async def test_record_agent_performance(tracker):
    """Record agent performance metrics"""
    result = AgentResult(
        agent_id="neuroscience_expert",
        task_id="t1",
        output="analysis",
        confidence=0.88,
        execution_time_ms=450.0
    )

    await tracker.record_agent_execution(
        result,
        task_type="validation",
        success=True
    )

    # Should be stored
    stats = await tracker.get_agent_stats("neuroscience_expert")
    assert stats["total_executions"] == 1
    assert stats["success_rate"] == 1.0

@pytest.mark.asyncio
async def test_calculate_success_rate(tracker):
    """Calculate agent success rate over time"""
    # Record multiple executions
    for i in range(10):
        result = AgentResult(
            agent_id="test_agent",
            task_id=f"t{i}",
            output="output",
            confidence=0.7 + (i * 0.02)
        )
        await tracker.record_agent_execution(
            result,
            task_type="test",
            success=(i % 2 == 0)  # 50% success
        )

    stats = await tracker.get_agent_stats("test_agent")
    assert stats["success_rate"] == 0.5
    assert stats["total_executions"] == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/metrics/test_tracker.py -v`
Expected: ModuleNotFoundError

**Step 3: Create database migration**

```python
# alembic/versions/xxx_add_metrics_tables.py
"""Add metrics tracking tables

Revision ID: xxx
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

def upgrade():
    # Agent execution metrics
    op.create_table(
        'agent_executions',
        sa.Column('id', UUID, primary_key=True),
        sa.Column('agent_id', sa.String(100), nullable=False, index=True),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('task_id', sa.String(100), nullable=False),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('confidence', sa.Float),
        sa.Column('execution_time_ms', sa.Float),
        sa.Column('tokens_used', sa.Integer),
        sa.Column('quality_score', sa.Float),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False)
    )

    # Workflow performance
    op.create_table(
        'workflow_metrics',
        sa.Column('id', UUID, primary_key=True),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('agents_used', JSONB, nullable=False),
        sa.Column('quality_score', sa.Float, nullable=False),
        sa.Column('execution_time_ms', sa.Float, nullable=False),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime, nullable=False)
    )

    # Indexes for performance
    op.create_index(
        'idx_agent_exec_agent_task',
        'agent_executions',
        ['agent_id', 'task_type']
    )
    op.create_index(
        'idx_workflow_task_quality',
        'workflow_metrics',
        ['task_type', 'quality_score']
    )

def downgrade():
    op.drop_table('workflow_metrics')
    op.drop_table('agent_executions')
```

**Step 4: Create metrics types and tracker**

```python
# src/metrics/types.py
from sqlalchemy import Column, String, Boolean, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from src.models.base import Base
from datetime import datetime
from uuid import uuid4

class AgentExecution(Base):
    __tablename__ = "agent_executions"

    id = Column(UUID, primary_key=True, default=uuid4)
    agent_id = Column(String(100), nullable=False, index=True)
    task_type = Column(String(50), nullable=False)
    task_id = Column(String(100), nullable=False)
    success = Column(Boolean, nullable=False)
    confidence = Column(Float)
    execution_time_ms = Column(Float)
    tokens_used = Column(Integer)
    quality_score = Column(Float)
    metadata = Column(JSONB)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkflowMetric(Base):
    __tablename__ = "workflow_metrics"

    id = Column(UUID, primary_key=True, default=uuid4)
    task_type = Column(String(50), nullable=False)
    agents_used = Column(JSONB, nullable=False)
    quality_score = Column(Float, nullable=False)
    execution_time_ms = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    metadata = Column(JSONB)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
```

```python
# src/metrics/tracker.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Dict, Any
from src.metrics.types import AgentExecution, WorkflowMetric
from src.agents.types import AgentResult

class PerformanceTracker:
    """Tracks agent and workflow performance"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def record_agent_execution(
        self,
        result: AgentResult,
        task_type: str,
        success: bool
    ):
        """Record individual agent execution"""

        execution = AgentExecution(
            agent_id=result.agent_id,
            task_type=task_type,
            task_id=result.task_id,
            success=success,
            confidence=result.confidence,
            execution_time_ms=result.execution_time_ms,
            tokens_used=result.tokens_used,
            metadata=result.metadata or {}
        )

        self.db.add(execution)
        await self.db.commit()

    async def get_agent_stats(
        self,
        agent_id: str,
        task_type: str = None
    ) -> Dict[str, Any]:
        """Get performance stats for agent"""

        query = select(AgentExecution).where(
            AgentExecution.agent_id == agent_id
        )

        if task_type:
            query = query.where(AgentExecution.task_type == task_type)

        result = await self.db.execute(query)
        executions = result.scalars().all()

        if not executions:
            return {
                "total_executions": 0,
                "success_rate": 0.5,
                "avg_confidence": 0.5
            }

        successes = sum(1 for e in executions if e.success)

        return {
            "total_executions": len(executions),
            "success_rate": successes / len(executions),
            "avg_confidence": sum(e.confidence or 0 for e in executions) / len(executions),
            "avg_execution_time_ms": sum(e.execution_time_ms or 0 for e in executions) / len(executions)
        }
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/metrics/test_tracker.py -v`
Expected: 2 passed

**Step 6: Apply migration and commit**

```bash
alembic upgrade head
git add alembic/versions/ src/metrics/ tests/metrics/
git commit -m "feat(metrics): add performance tracking with database persistence"
```

---

## Tier 1 Complete! (Week 8 checkpoint)

At this point you have:
- ✅ Agent base architecture
- ✅ Agent pool with domain experts
- ✅ Meta-router with task analysis
- ✅ Context manager
- ✅ Quality validation
- ✅ API integration
- ✅ Performance tracking

**Next: Tier 2 (RL-based optimization) - Separate plan document**

---

## Testing Strategy

**Unit Tests** (pytest):
```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Run specific module
pytest tests/agents/ -v
pytest tests/router/ -v
pytest tests/context/ -v
```

**Integration Tests**:
```bash
# E2E workflow tests
pytest tests/integration/test_multi_agent_workflow.py -v
```

**Manual Testing**:
```bash
# Start API
uvicorn src.main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/api/v1/multi-agent/research \
  -H "Content-Type: application/json" \
  -d '{"description": "Search fMRI papers", "task_type": "literature_search"}'
```

---

## Quality Gates

**Before each commit:**
1. All tests pass: `pytest tests/ -v`
2. Type checks pass: `mypy src/`
3. Linting clean: `ruff check src/`
4. Format code: `black src/ tests/`

**Before PR:**
1. Coverage ≥ 80%: `pytest --cov=src --cov-report=term`
2. No merge conflicts
3. Documentation updated
4. Changelog entry added

---

## Plan saved to: `docs/plans/2025-10-14-adaptive-research-assistant.md`

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
