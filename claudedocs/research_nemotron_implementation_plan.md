# Nemotron 기반 구현 계획 및 워크플로우 구성

**연구 일자**: 2025-10-24
**프로젝트**: AI-CoScientist
**목적**: Nemotron 기반 RAG 시스템 구현을 위한 아키텍처, 워크플로우, 배포 전략 수립

---

## 📋 Executive Summary

본 연구는 NVIDIA Nemotron 생태계를 활용한 AI-CoScientist 시스템 구현 방안을 제시합니다. 현재 시스템의 강점(GPT-4/Claude 기반 논문 평가)을 유지하면서 Nemotron의 장점(reranking, 효율적인 retrieval, on-premise 배포)을 통합하는 **하이브리드 접근법**을 권장합니다.

### 핵심 권장사항
- ✅ **하이브리드 아키텍처**: GPT-4/Claude (평가) + Nemotron (검색/처리)
- ✅ **LangGraph 워크플로우**: 상태 관리형 agentic RAG 파이프라인
- ✅ **NeMo Retriever 통합**: EmbedQA + RerankQA로 검색 품질 25-40% 향상
- ✅ **단계적 배포**: Docker → Kubernetes → 프로덕션 확장
- ✅ **성능 최적화**: TensorRT-LLM 기반 2-4x throughput 개선

---

## 🏗️ Part 1: 아키텍처 설계

### 1.1 하이브리드 아키텍처 (권장)

```
┌─────────────────────────────────────────────────────────────┐
│                   AI-CoScientist Platform                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │  Paper Upload   │────────▶│ LangGraph Agent │           │
│  │  & Processing   │         │   Orchestrator  │           │
│  └─────────────────┘         └────────┬────────┘           │
│                                        │                     │
│                    ┌───────────────────┴───────────────┐    │
│                    │                                     │    │
│           ┌────────▼────────┐              ┌───────────▼────┐
│           │  Evaluation     │              │  Retrieval     │
│           │  Pipeline       │              │  Pipeline      │
│           │                 │              │                │
│           │ GPT-4/Claude    │              │ Nemotron Nano  │
│           │ (Primary)       │              │ 9B V2          │
│           │                 │              │                │
│           │ • Novelty       │              │ • Embedding    │
│           │ • Methodology   │              │   (EmbedQA 1B) │
│           │ • Clarity       │              │ • Reranking    │
│           │ • Significance  │              │   (RerankQA 1B)│
│           └─────────────────┘              └────────────────┘
│                    │                                │         │
│                    └────────────┬──────────────────┘         │
│                                 │                             │
│                        ┌────────▼────────┐                   │
│                        │  Vector Store   │                   │
│                        │                 │                   │
│                        │  ChromaDB       │                   │
│                        │  (Persistent)   │                   │
│                        └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 컴포넌트 역할 분담

| 컴포넌트 | 모델 | 역할 | 이유 |
|---------|------|------|------|
| **Paper Evaluation** | GPT-4 / Claude Opus | 논문 품질 평가 (Novelty, Methodology, Clarity, Significance) | 복잡한 과학적 추론 능력 필수 (175B+ params) |
| **Embedding** | Llama 3.2 EmbedQA 1B | 쿼리/문서 벡터화 | Q&A 특화, 경량화 |
| **Reranking** | Llama 3.2 RerankQA 1B | 검색 결과 재순위화 | 현재 시스템의 주요 gap 해결 |
| **Summarization** | Nemotron Nano 9B | 논문 요약, 정보 추출 | Cost 절감, 온프레미스 가능 |
| **Orchestration** | LangGraph | 워크플로우 관리 | 상태 관리, 도구 호출, 복잡한 흐름 처리 |
| **Vector DB** | ChromaDB | 개선 패턴 영구 저장 | 세션 간 학습 누적 필요 |

### 1.3 아키텍처 패턴: Agentic RAG with LangGraph

```python
from langgraph.graph import StateGraph, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

# State 정의
class PaperAnalysisState(TypedDict):
    paper_content: str
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    evaluation_scores: Dict[str, float]
    improvement_suggestions: List[str]
    current_step: str

# Agent 노드 정의
def retrieve_node(state: PaperAnalysisState):
    """NeMo Retriever로 관련 문서 검색"""
    embeddings = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2")
    query_embedding = embeddings.embed_query(state["query"])
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=10)
    return {"retrieved_docs": docs, "current_step": "rerank"}

def rerank_node(state: PaperAnalysisState):
    """RerankQA로 문서 재순위화"""
    reranker = NVIDIARerank(model="nvidia/llama-3.2-nv-rerankqa-1b-v2")
    reranked = reranker.compress_documents(
        documents=state["retrieved_docs"],
        query=state["query"]
    )
    return {"reranked_docs": reranked[:5], "current_step": "evaluate"}

def evaluate_node(state: PaperAnalysisState):
    """GPT-4로 논문 평가 (핵심 작업)"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    context = "\n\n".join([doc.page_content for doc in state["reranked_docs"]])

    prompt = f"""
    Evaluate the following scientific paper using retrieved context.

    Paper: {state["paper_content"][:2000]}

    Relevant Context: {context}

    Provide scores (0-10) for:
    1. Novelty
    2. Methodology
    3. Clarity
    4. Significance
    """

    response = llm.invoke(prompt)
    scores = parse_scores(response.content)
    return {"evaluation_scores": scores, "current_step": "suggest"}

def suggest_improvements_node(state: PaperAnalysisState):
    """개선 제안 생성 (Nemotron 사용 가능)"""
    # 경량 작업은 Nemotron 사용
    llm = ChatNVIDIA(model="nvidia/nvidia-nemotron-nano-9b-v2")

    prompt = f"""
    Based on scores: {state["evaluation_scores"]}
    Suggest 3-5 specific improvements for the paper.
    """

    suggestions = llm.invoke(prompt).content
    return {"improvement_suggestions": parse_suggestions(suggestions), "current_step": "end"}

# Graph 구성
workflow = StateGraph(PaperAnalysisState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("suggest", suggest_improvements_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "evaluate")
workflow.add_edge("evaluate", "suggest")
workflow.add_edge("suggest", END)

app = workflow.compile()
```

---

## 🔄 Part 2: 워크플로우 구성

### 2.1 논문 평가 워크플로우

```
[Paper Upload] → [Preprocessing] → [LangGraph Orchestration]
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
            [Retrieval Step]        [Evaluation Step]      [Enhancement Step]
                    │                       │                       │
        ┌───────────┴───────────┐  ┌───────┴───────┐      ┌────────┴────────┐
        │                       │  │               │      │                 │
        ▼                       ▼  ▼               ▼      ▼                 ▼
    [Embed]                [Rerank] [GPT-4]    [Claude]  [Nemotron]    [Templates]
    EmbedQA 1B             RerankQA  Scoring   Scoring   Suggestions   Retrieval
        │                       │  │               │      │                 │
        └───────────┬───────────┘  └───────┬───────┘      └────────┬────────┘
                    ▼                      ▼                       ▼
            [ChromaDB Store]        [Score Ensemble]        [Action Plan]
                                            │
                                            ▼
                                    [Final Report]
```

### 2.2 단계별 워크플로우 상세

#### Step 1: 문서 전처리 및 분석
```python
class DocumentProcessor:
    def __init__(self):
        self.nemotron_llm = ChatNVIDIA(model="nvidia/nvidia-nemotron-nano-9b-v2")

    async def preprocess_paper(self, paper_path: str) -> ProcessedPaper:
        """논문 전처리 및 구조 분석"""
        # 1. 문서 로드
        raw_text = self.load_document(paper_path)

        # 2. 섹션 추출 (Nemotron 사용 - 경량 작업)
        sections = await self.extract_sections(raw_text)

        # 3. 메타데이터 추출
        metadata = await self.extract_metadata(raw_text)

        # 4. 청킹 (RecursiveCharacterTextSplitter)
        chunks = self.chunk_document(raw_text, chunk_size=800, overlap=120)

        return ProcessedPaper(
            sections=sections,
            metadata=metadata,
            chunks=chunks,
            raw_text=raw_text
        )

    async def extract_sections(self, text: str) -> Dict[str, str]:
        """섹션 구조 분석 (Abstract, Intro, Methods, Results, Discussion)"""
        prompt = """
        Identify and extract the following sections from this paper:
        - Abstract
        - Introduction
        - Methods/Methodology
        - Results
        - Discussion
        - Conclusion

        Return as JSON: {"section_name": "content"}
        """

        response = await self.nemotron_llm.ainvoke(prompt + f"\n\nPaper:\n{text[:4000]}")
        return json.loads(response.content)
```

#### Step 2: Retrieval Pipeline (NeMo Retriever)
```python
class NeMoRetrievalPipeline:
    def __init__(self, chromadb_client):
        # Embedding 모델
        self.embedder = NVIDIAEmbeddings(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            truncate="END"
        )

        # Reranking 모델
        self.reranker = NVIDIARerank(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2"
        )

        # Vector store
        self.vectorstore = chromadb_client

    async def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 5
    ) -> List[Document]:
        """검색 + 재순위화 파이프라인"""

        # 1. Embedding
        query_vector = await self.embedder.aembed_query(query)

        # 2. Similarity Search
        candidates = await self.vectorstore.asimilarity_search_by_vector(
            query_vector,
            k=top_k
        )

        # 3. Reranking (핵심 개선점!)
        reranked_docs = self.reranker.compress_documents(
            documents=candidates,
            query=query
        )

        return reranked_docs[:rerank_top_k]
```

#### Step 3: Evaluation Pipeline (GPT-4/Claude)
```python
class HybridEvaluationPipeline:
    def __init__(self):
        self.gpt4 = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.claude = Anthropic(model="claude-opus-4")
        self.nemotron = ChatNVIDIA(model="nvidia/nvidia-nemotron-nano-9b-v2")

    async def evaluate_paper(
        self,
        paper: ProcessedPaper,
        retrieved_context: List[Document]
    ) -> EvaluationResult:
        """하이브리드 평가: GPT-4 (메인) + Claude (검증)"""

        # Context 준비
        context_text = "\n\n".join([doc.page_content for doc in retrieved_context])

        # GPT-4 평가 (40% 가중치)
        gpt4_scores = await self._evaluate_with_gpt4(paper, context_text)

        # Claude 평가 (30% 가중치)
        claude_scores = await self._evaluate_with_claude(paper, context_text)

        # Nemotron 보조 분석 (30% 가중치 - 기술적 측면)
        nemotron_scores = await self._technical_analysis_with_nemotron(paper)

        # Ensemble
        final_scores = self._ensemble_scores(
            gpt4_scores, claude_scores, nemotron_scores,
            weights=[0.4, 0.3, 0.3]
        )

        return EvaluationResult(
            overall_score=final_scores["overall"],
            dimension_scores=final_scores["dimensions"],
            confidence=final_scores["confidence"],
            model_contributions={
                "gpt4": gpt4_scores,
                "claude": claude_scores,
                "nemotron": nemotron_scores
            }
        )

    async def _evaluate_with_gpt4(self, paper, context):
        """GPT-4로 심층 평가"""
        prompt = f"""
        You are an expert scientific paper reviewer. Evaluate this paper across four dimensions.

        Paper Sections:
        - Abstract: {paper.sections.get('abstract', '')[:500]}
        - Introduction: {paper.sections.get('introduction', '')[:1000]}
        - Methods: {paper.sections.get('methods', '')[:1000]}
        - Results: {paper.sections.get('results', '')[:1000]}

        Retrieved Context (similar successful papers):
        {context[:2000]}

        Evaluate on a scale of 0-10:
        1. **Novelty**: Originality and innovation
        2. **Methodology**: Rigor and validity
        3. **Clarity**: Writing quality and organization
        4. **Significance**: Impact and importance

        Provide detailed reasoning for each score.
        Return as JSON.
        """

        response = await self.gpt4.ainvoke(prompt)
        return self._parse_evaluation_response(response.content)
```

#### Step 4: Enhancement Workflow
```python
class EnhancementWorkflow:
    def __init__(self, llm_service, rag_pipeline):
        self.llm_service = llm_service
        self.rag_pipeline = rag_pipeline

    async def generate_improvement_plan(
        self,
        evaluation: EvaluationResult,
        paper: ProcessedPaper
    ) -> ImprovementPlan:
        """개선 계획 생성 (Agentic 접근)"""

        # LangGraph agent 정의
        workflow = StateGraph(ImprovementState)

        # 1. 약점 식별 노드
        workflow.add_node("identify_gaps", self._identify_gaps_node)

        # 2. 개선 전략 검색 노드
        workflow.add_node("retrieve_strategies", self._retrieve_strategies_node)

        # 3. 개선안 생성 노드
        workflow.add_node("generate_suggestions", self._generate_suggestions_node)

        # 4. 우선순위 결정 노드
        workflow.add_node("prioritize", self._prioritize_suggestions_node)

        # 5. 실행 계획 생성 노드
        workflow.add_node("create_action_plan", self._create_action_plan_node)

        # Edge 연결
        workflow.set_entry_point("identify_gaps")
        workflow.add_edge("identify_gaps", "retrieve_strategies")
        workflow.add_edge("retrieve_strategies", "generate_suggestions")
        workflow.add_edge("generate_suggestions", "prioritize")
        workflow.add_edge("prioritize", "create_action_plan")
        workflow.add_edge("create_action_plan", END)

        # 실행
        app = workflow.compile()
        result = await app.ainvoke({
            "evaluation": evaluation,
            "paper": paper,
            "threshold_score": 8.5
        })

        return ImprovementPlan(**result)

    async def _retrieve_strategies_node(self, state: ImprovementState):
        """성공 패턴 검색 (NeMo Retriever 활용)"""
        gaps = state["identified_gaps"]

        strategies = []
        for gap in gaps:
            # 각 gap에 대해 유사한 개선 사례 검색
            query = f"How to improve {gap['dimension']} in scientific papers: {gap['issue']}"

            docs = await self.rag_pipeline.retrieve_and_rerank(query, top_k=5)
            strategies.extend(docs)

        return {"retrieved_strategies": strategies}
```

### 2.3 LangGraph 워크플로우 패턴

#### Pattern 1: Query Router (쿼리 라우팅)
```python
class QueryRouter:
    """쿼리 유형에 따라 적절한 처리 경로 선택"""

    def route_query(self, state: AgentState) -> str:
        """쿼리 분석 및 라우팅"""
        query = state["query"]

        # 분류 로직
        if "evaluate" in query.lower() or "score" in query.lower():
            return "evaluation_path"
        elif "improve" in query.lower() or "enhance" in query.lower():
            return "enhancement_path"
        elif "compare" in query.lower():
            return "comparison_path"
        else:
            return "general_path"

# Graph에 조건부 엣지 추가
workflow.add_conditional_edges(
    "router",
    QueryRouter().route_query,
    {
        "evaluation_path": "evaluate",
        "enhancement_path": "improve",
        "comparison_path": "compare",
        "general_path": "answer"
    }
)
```

#### Pattern 2: Self-Correction (자가 검증)
```python
def self_correction_workflow():
    """생성 → 검증 → 수정 반복"""

    workflow = StateGraph(CorrectionState)

    # 초기 생성
    workflow.add_node("generate", generate_improvement)

    # 품질 검증
    workflow.add_node("validate", validate_quality)

    # 수정
    workflow.add_node("revise", revise_content)

    # 조건부 루프
    def should_continue(state):
        if state["quality_score"] >= 0.8:
            return "end"
        elif state["iteration"] < 3:
            return "revise"
        else:
            return "end"

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "revise": "revise",
            "end": END
        }
    )
    workflow.add_edge("revise", "generate")

    return workflow.compile()
```

#### Pattern 3: Multi-Step Reasoning (다단계 추론)
```python
class MultiStepReasoning:
    """복잡한 분석을 단계적으로 수행"""

    def create_workflow(self):
        workflow = StateGraph(ReasoningState)

        # Step 1: 문제 분해
        workflow.add_node("decompose", self.decompose_problem)

        # Step 2: 각 하위 문제 해결
        workflow.add_node("solve_subproblems", self.solve_subproblems)

        # Step 3: 결과 통합
        workflow.add_node("synthesize", self.synthesize_results)

        # Step 4: 일관성 검증
        workflow.add_node("verify_consistency", self.verify_consistency)

        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "solve_subproblems")
        workflow.add_edge("solve_subproblems", "synthesize")
        workflow.add_edge("synthesize", "verify_consistency")
        workflow.add_edge("verify_consistency", END)

        return workflow.compile()
```

---

## 🚀 Part 3: 배포 전략

### 3.1 단계별 배포 로드맵

#### Phase 1: 로컬 Docker 개발 (1-2주)
```bash
# NIM 컨테이너 설정
export NGC_API_KEY=your_ngc_key
docker volume create nim-cache

# Nemotron LLM
docker run -d --name nemotron-llm \
  --runtime=nvidia --gpus 1 \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v nim-cache:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest

# NeMo Retriever Embedding
docker run -d --name nemo-embedder \
  --runtime=nvidia --gpus 1 \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v nim-cache:/opt/nim/.cache \
  -p 8001:8000 \
  nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest

# NeMo Retriever Reranker
docker run -d --name nemo-reranker \
  --runtime=nvidia --gpus 1 \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v nim-cache:/opt/nim/.cache \
  -p 8002:8000 \
  nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:latest
```

**Docker Compose 구성**:
```yaml
# docker-compose.nemotron.yml
version: '3.8'

services:
  nemotron-llm:
    image: nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest
    runtime: nvidia
    environment:
      - NGC_API_KEY=${NGC_API_KEY}
    volumes:
      - nim-cache:/opt/nim/.cache
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: 16gb
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nemo-embedder:
    image: nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
    runtime: nvidia
    environment:
      - NGC_API_KEY=${NGC_API_KEY}
    volumes:
      - nim-cache:/opt/nim/.cache
    ports:
      - "8001:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: 16gb

  nemo-reranker:
    image: nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:latest
    runtime: nvidia
    environment:
      - NGC_API_KEY=${NGC_API_KEY}
    volumes:
      - nim-cache:/opt/nim/.cache
    ports:
      - "8002:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: 16gb

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8003:8000"
    volumes:
      - chromadb-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

  ai-coscientist-api:
    build: .
    depends_on:
      - nemotron-llm
      - nemo-embedder
      - nemo-reranker
      - chromadb
    environment:
      - NEMOTRON_BASE_URL=http://nemotron-llm:8000/v1
      - EMBEDDER_BASE_URL=http://nemo-embedder:8000/v1
      - RERANKER_BASE_URL=http://nemo-reranker:8000/v1
      - CHROMADB_URL=http://chromadb:8000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    ports:
      - "8080:8000"

volumes:
  nim-cache:
  chromadb-data:
```

#### Phase 2: Kubernetes 배포 (2-3주)

**NIM Operator 설치**:
```bash
# NIM Operator 설치
kubectl create namespace nim-operator
helm repo add nvidia https://nvidia.github.io/nim-operator
helm install nim-operator nvidia/nim-operator \
  --namespace nim-operator \
  --set nimOperator.enabled=true
```

**NIM 리소스 정의**:
```yaml
# nim-nemotron-deployment.yaml
apiVersion: apps.nvidia.com/v1alpha1
kind: NIMService
metadata:
  name: nemotron-llm
  namespace: ai-coscientist
spec:
  image:
    repository: nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2
    tag: latest
    pullSecrets:
      - name: ngc-secret

  resources:
    limits:
      nvidia.com/gpu: 1
      memory: 32Gi
    requests:
      nvidia.com/gpu: 1
      memory: 16Gi

  # 성능 프로파일 선택
  profile: throughput  # or 'latency'

  env:
    - name: NGC_API_KEY
      valueFrom:
        secretKeyRef:
          name: ngc-api-key
          key: api-key

  storage:
    modelCache:
      storageClassName: fast-ssd
      size: 100Gi

  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 5
    targetCPUUtilizationPercentage: 70
    targetGPUUtilizationPercentage: 80

  service:
    type: ClusterIP
    port: 8000
```

**NeMo Retriever 배포**:
```yaml
# nemo-retriever-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-embedder
  namespace: ai-coscientist
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nemo-embedder
  template:
    metadata:
      labels:
        app: nemo-embedder
    spec:
      containers:
      - name: embedder
        image: nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        env:
        - name: NGC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ngc-api-key
              key: api-key
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: model-cache
          mountPath: /opt/nim/.cache
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: nim-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nemo-embedder-svc
  namespace: ai-coscientist
spec:
  selector:
    app: nemo-embedder
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

**HPA (Horizontal Pod Autoscaler) 설정**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nemotron-hpa
  namespace: ai-coscientist
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nemotron-llm
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "10"
```

#### Phase 3: 프로덕션 최적화 (3-4주)

**모니터링 (Prometheus + Grafana)**:
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s

    scrape_configs:
      - job_name: 'nim-metrics'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - ai-coscientist
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: (nemotron|nemo-embedder|nemo-reranker)
        metrics_path: /metrics
```

**성능 메트릭 수집**:
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# 메트릭 정의
inference_requests = Counter(
    'nim_inference_requests_total',
    'Total number of inference requests',
    ['model', 'status']
)

inference_latency = Histogram(
    'nim_inference_latency_seconds',
    'Inference latency in seconds',
    ['model', 'profile'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

active_requests = Gauge(
    'nim_active_requests',
    'Number of active inference requests',
    ['model']
)

tokens_generated = Counter(
    'nim_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()
        model_name = scope["path"].split("/")[2]  # Extract model from path

        active_requests.labels(model=model_name).inc()

        try:
            await self.app(scope, receive, send)
            inference_requests.labels(model=model_name, status="success").inc()
        except Exception as e:
            inference_requests.labels(model=model_name, status="error").inc()
            raise
        finally:
            duration = time.time() - start_time
            inference_latency.labels(
                model=model_name,
                profile="throughput"
            ).observe(duration)
            active_requests.labels(model=model_name).dec()
```

### 3.2 성능 최적화 전략

#### 3.2.1 TensorRT-LLM 최적화

```python
# NIM 컨테이너는 자동으로 TensorRT-LLM 최적화 적용
# 프로파일 선택으로 제어

# Latency 우선 (낮은 지연시간)
# - 더 많은 GPU 사용
# - 작은 배치 사이즈
# - TTFT (Time to First Token) 최소화
PROFILE_LATENCY = {
    "max_batch_size": 8,
    "gpu_count": 2,
    "optimization_level": 3,
    "precision": "fp16"
}

# Throughput 우선 (높은 처리량)
# - 최소 GPU로 최대 처리량
# - 큰 배치 사이즈
# - 전체 throughput 최대화
PROFILE_THROUGHPUT = {
    "max_batch_size": 64,
    "gpu_count": 1,
    "optimization_level": 3,
    "precision": "fp16"
}

# 환경변수로 프로파일 선택
# docker run -e NIM_OPTIMIZATION_PROFILE=throughput ...
```

#### 3.2.2 배치 처리 최적화

```python
class BatchedInferenceService:
    """여러 요청을 배치로 묶어 처리"""

    def __init__(self, model_url: str, max_batch_size: int = 32):
        self.model_url = model_url
        self.max_batch_size = max_batch_size
        self.pending_requests = []
        self.batch_timeout = 0.1  # 100ms

    async def infer(self, prompt: str) -> str:
        """단일 요청을 배치 큐에 추가"""
        future = asyncio.Future()
        self.pending_requests.append((prompt, future))

        # 배치 크기 또는 타임아웃 도달 시 처리
        if len(self.pending_requests) >= self.max_batch_size:
            await self._process_batch()
        else:
            asyncio.create_task(self._auto_flush())

        return await future

    async def _process_batch(self):
        """배치 처리"""
        if not self.pending_requests:
            return

        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]

        prompts = [req[0] for req in batch]
        futures = [req[1] for req in batch]

        # 배치 추론
        responses = await self._batch_inference(prompts)

        # 결과 분배
        for future, response in zip(futures, responses):
            future.set_result(response)

    async def _batch_inference(self, prompts: List[str]) -> List[str]:
        """실제 배치 추론"""
        response = await httpx.post(
            f"{self.model_url}/v1/completions",
            json={
                "prompts": prompts,
                "max_tokens": 512,
                "temperature": 0.7
            }
        )
        return [choice["text"] for choice in response.json()["choices"]]

    async def _auto_flush(self):
        """타임아웃 후 자동 플러시"""
        await asyncio.sleep(self.batch_timeout)
        await self._process_batch()
```

#### 3.2.3 캐싱 전략

```python
from functools import lru_cache
import hashlib
import pickle
import redis

class MultiLevelCache:
    """3단계 캐싱: 메모리 → Redis → Vector DB"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.memory_cache_size = 1000

    @lru_cache(maxsize=1000)
    def memory_cache_get(self, key: str) -> Optional[str]:
        """L1: 메모리 캐시 (가장 빠름)"""
        return None  # LRU cache가 자동 관리

    async def redis_cache_get(self, key: str) -> Optional[str]:
        """L2: Redis 캐시 (빠름, 공유됨)"""
        cached = await self.redis.get(f"cache:{key}")
        if cached:
            return pickle.loads(cached)
        return None

    async def redis_cache_set(self, key: str, value: str, ttl: int = 3600):
        """Redis에 캐시 저장"""
        await self.redis.setex(
            f"cache:{key}",
            ttl,
            pickle.dumps(value)
        )

    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """임베딩 캐시 조회"""
        # 텍스트 해시
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # L1: 메모리 캐시
        cached = self.memory_cache_get(text_hash)
        if cached:
            return cached

        # L2: Redis 캐시
        cached = await self.redis_cache_get(f"embed:{text_hash}")
        if cached:
            # L1 캐시에도 저장
            self.memory_cache_get.cache_info()  # Warm up
            return cached

        return None

    async def cache_embedding(self, text: str, embedding: List[float]):
        """임베딩 캐시 저장"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Redis에 저장 (1시간 TTL)
        await self.redis_cache_set(f"embed:{text_hash}", embedding, ttl=3600)

        # 메모리 캐시는 LRU가 자동 관리
        self.memory_cache_get(text_hash)  # Touch to cache
```

#### 3.2.4 병렬 처리

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelPipeline:
    """여러 모델을 병렬로 호출"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def parallel_evaluation(
        self,
        paper: ProcessedPaper
    ) -> Dict[str, Any]:
        """GPT-4, Claude, Nemotron 동시 호출"""

        # 3개 모델 병렬 실행
        tasks = [
            self.evaluate_with_gpt4(paper),
            self.evaluate_with_claude(paper),
            self.evaluate_with_nemotron(paper)
        ]

        # 모든 결과 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 에러 처리
        gpt4_result = results[0] if not isinstance(results[0], Exception) else None
        claude_result = results[1] if not isinstance(results[1], Exception) else None
        nemotron_result = results[2] if not isinstance(results[2], Exception) else None

        # Ensemble (에러 발생 시 사용 가능한 것만)
        return self.ensemble_results(gpt4_result, claude_result, nemotron_result)

    async def parallel_retrieval(
        self,
        queries: List[str]
    ) -> List[List[Document]]:
        """여러 쿼리 병렬 검색"""

        tasks = [
            self.retrieve_and_rerank(query)
            for query in queries
        ]

        return await asyncio.gather(*tasks)
```

---

## 📊 Part 4: 성능 벤치마크 및 모니터링

### 4.1 핵심 성능 지표

| 메트릭 | 목표 | 측정 방법 |
|--------|------|----------|
| **TTFT** (Time to First Token) | < 500ms | Latency profile 사용 |
| **ITL** (Inter-Token Latency) | < 50ms | Streaming 응답 모니터링 |
| **Throughput** | > 100 tokens/sec | Throughput profile + batch |
| **Paper Evaluation Time** | < 30 sec | End-to-end 측정 |
| **Retrieval Accuracy** | > 0.85 MRR | Reranking 후 평가 |
| **GPU Utilization** | 70-85% | NVIDIA DCGM |
| **Cost per Paper** | < $0.50 | API 비용 추적 |

### 4.2 A/B 테스트 프레임워크

```python
class ABTestingFramework:
    """기존 시스템 vs Nemotron 비교"""

    def __init__(self):
        self.metrics_tracker = MetricsTracker()

    async def comparative_evaluation(
        self,
        paper: ProcessedPaper,
        test_group: str = "A"  # A: 기존, B: Nemotron
    ) -> ComparisonResult:
        """A/B 테스트 실행"""

        if test_group == "A":
            # 기존 시스템 (GPT-4 only)
            result = await self.evaluate_with_legacy_system(paper)
        else:
            # Nemotron 하이브리드 시스템
            result = await self.evaluate_with_hybrid_system(paper)

        # 메트릭 기록
        await self.metrics_tracker.record(
            test_group=test_group,
            latency=result.latency,
            cost=result.cost,
            quality_score=result.quality_score,
            user_satisfaction=result.user_satisfaction
        )

        return result

    async def analyze_results(self) -> ABTestReport:
        """테스트 결과 분석"""
        metrics_a = await self.metrics_tracker.get_group_metrics("A")
        metrics_b = await self.metrics_tracker.get_group_metrics("B")

        return ABTestReport(
            latency_improvement=self._calculate_improvement(
                metrics_a.avg_latency,
                metrics_b.avg_latency
            ),
            cost_reduction=self._calculate_improvement(
                metrics_a.avg_cost,
                metrics_b.avg_cost
            ),
            quality_delta=metrics_b.avg_quality - metrics_a.avg_quality,
            statistical_significance=self._ttest(metrics_a, metrics_b),
            recommendation=self._generate_recommendation(metrics_a, metrics_b)
        )
```

### 4.3 품질 보증 체크리스트

```python
class QualityAssurance:
    """배포 전 품질 검증"""

    async def pre_deployment_checklist(self) -> ChecklistResult:
        """배포 전 필수 검증 항목"""

        checks = []

        # 1. 모델 로딩 확인
        checks.append(await self._verify_model_loading())

        # 2. API 응답성 확인
        checks.append(await self._verify_api_responsiveness())

        # 3. 품질 회귀 테스트
        checks.append(await self._verify_quality_regression())

        # 4. 성능 벤치마크
        checks.append(await self._verify_performance_benchmarks())

        # 5. 리소스 사용량 확인
        checks.append(await self._verify_resource_usage())

        # 6. 보안 검사
        checks.append(await self._verify_security())

        all_passed = all(check.passed for check in checks)

        return ChecklistResult(
            all_checks_passed=all_passed,
            individual_checks=checks,
            deployment_approved=all_passed,
            failed_checks=[c for c in checks if not c.passed]
        )

    async def _verify_quality_regression(self) -> CheckResult:
        """품질 회귀 없음을 확인"""
        # 테스트 논문 세트로 평가
        test_papers = await self.load_test_dataset()

        baseline_scores = []
        new_scores = []

        for paper in test_papers:
            # 기존 시스템 점수
            baseline = await self.evaluate_with_baseline(paper)
            baseline_scores.append(baseline.overall_score)

            # 새 시스템 점수
            new = await self.evaluate_with_new_system(paper)
            new_scores.append(new.overall_score)

        # 통계적 유의성 검증
        avg_baseline = np.mean(baseline_scores)
        avg_new = np.mean(new_scores)

        # 허용 오차: -0.1 이내
        passed = (avg_new >= avg_baseline - 0.1)

        return CheckResult(
            name="Quality Regression Test",
            passed=passed,
            details={
                "baseline_avg": avg_baseline,
                "new_avg": avg_new,
                "delta": avg_new - avg_baseline,
                "threshold": -0.1
            }
        )
```

---

## 🔧 Part 5: 통합 코드 예제

### 5.1 완전한 하이브리드 서비스 구현

```python
# src/services/hybrid_rag_service.py

from typing import List, Dict, Optional, Any
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import asyncio

class HybridRAGService:
    """Nemotron + GPT-4/Claude 하이브리드 RAG 서비스"""

    def __init__(
        self,
        nemotron_url: str = "http://localhost:8000/v1",
        embedder_url: str = "http://localhost:8001/v1",
        reranker_url: str = "http://localhost:8002/v1",
        chromadb_client = None,
        openai_api_key: str = None,
        anthropic_api_key: str = None
    ):
        # Nemotron 모델들
        self.nemotron_llm = ChatNVIDIA(
            base_url=nemotron_url,
            model="nvidia/nvidia-nemotron-nano-9b-v2",
            temperature=0.7,
            max_tokens=2048
        )

        self.embedder = NVIDIAEmbeddings(
            base_url=embedder_url,
            model="nvidia/llama-3.2-nv-embedqa-1b-v2"
        )

        self.reranker = NVIDIARerank(
            base_url=reranker_url,
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2"
        )

        # 평가용 LLM들
        self.gpt4 = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=openai_api_key
        ) if openai_api_key else None

        self.claude = ChatAnthropic(
            model="claude-opus-4",
            temperature=0.3,
            api_key=anthropic_api_key
        ) if anthropic_api_key else None

        # Vector store
        self.vectorstore = chromadb_client

        # LangGraph workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""

        class PaperState(TypedDict):
            paper: ProcessedPaper
            query: str
            retrieved_docs: List[Document]
            reranked_docs: List[Document]
            gpt4_scores: Optional[Dict]
            claude_scores: Optional[Dict]
            nemotron_analysis: Optional[Dict]
            final_evaluation: Optional[EvaluationResult]
            improvement_plan: Optional[ImprovementPlan]

        workflow = StateGraph(PaperState)

        # 노드 정의
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("evaluate_gpt4", self._evaluate_gpt4_node)
        workflow.add_node("evaluate_claude", self._evaluate_claude_node)
        workflow.add_node("analyze_nemotron", self._analyze_nemotron_node)
        workflow.add_node("ensemble", self._ensemble_node)
        workflow.add_node("improve", self._improve_node)

        # 워크플로우 구성
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")

        # 병렬 평가
        workflow.add_edge("rerank", "evaluate_gpt4")
        workflow.add_edge("rerank", "evaluate_claude")
        workflow.add_edge("rerank", "analyze_nemotron")

        # 결과 통합
        workflow.add_edge("evaluate_gpt4", "ensemble")
        workflow.add_edge("evaluate_claude", "ensemble")
        workflow.add_edge("analyze_nemotron", "ensemble")

        workflow.add_edge("ensemble", "improve")
        workflow.add_edge("improve", END)

        return workflow.compile()

    async def _retrieve_node(self, state: PaperState) -> Dict:
        """문서 검색"""
        query = state["query"]

        # Embedding
        query_vector = await self.embedder.aembed_query(query)

        # Vector search
        docs = await self.vectorstore.asimilarity_search_by_vector(
            query_vector,
            k=10
        )

        return {"retrieved_docs": docs}

    async def _rerank_node(self, state: PaperState) -> Dict:
        """재순위화"""
        reranked = self.reranker.compress_documents(
            documents=state["retrieved_docs"],
            query=state["query"]
        )

        return {"reranked_docs": reranked[:5]}

    async def _evaluate_gpt4_node(self, state: PaperState) -> Dict:
        """GPT-4 평가"""
        if not self.gpt4:
            return {"gpt4_scores": None}

        paper = state["paper"]
        context = "\n\n".join([doc.page_content for doc in state["reranked_docs"]])

        prompt = f"""
        Evaluate this scientific paper on a scale of 0-10:

        Paper Abstract: {paper.sections.get('abstract', '')[:500]}

        Context from similar papers: {context[:1500]}

        Provide scores for:
        1. Novelty
        2. Methodology
        3. Clarity
        4. Significance

        Return as JSON: {{"novelty": 8.5, "methodology": 7.0, ...}}
        """

        response = await self.gpt4.ainvoke(prompt)
        scores = self._parse_json_scores(response.content)

        return {"gpt4_scores": scores}

    async def _evaluate_claude_node(self, state: PaperState) -> Dict:
        """Claude 평가"""
        if not self.claude:
            return {"claude_scores": None}

        # GPT-4와 유사한 로직
        # ... (생략)

        return {"claude_scores": scores}

    async def _analyze_nemotron_node(self, state: PaperState) -> Dict:
        """Nemotron 기술 분석"""
        paper = state["paper"]

        prompt = f"""
        Analyze the technical aspects of this paper:

        Methods: {paper.sections.get('methods', '')[:1000]}
        Results: {paper.sections.get('results', '')[:1000]}

        Evaluate:
        1. Statistical rigor
        2. Experimental design
        3. Data analysis quality

        Return as JSON.
        """

        response = await self.nemotron_llm.ainvoke(prompt)
        analysis = self._parse_json_scores(response.content)

        return {"nemotron_analysis": analysis}

    async def _ensemble_node(self, state: PaperState) -> Dict:
        """점수 통합"""
        weights = {
            "gpt4": 0.4,
            "claude": 0.3,
            "nemotron": 0.3
        }

        final_scores = {}

        for dimension in ["novelty", "methodology", "clarity", "significance"]:
            scores = []

            if state["gpt4_scores"]:
                scores.append(state["gpt4_scores"].get(dimension, 0) * weights["gpt4"])

            if state["claude_scores"]:
                scores.append(state["claude_scores"].get(dimension, 0) * weights["claude"])

            if state["nemotron_analysis"]:
                scores.append(state["nemotron_analysis"].get(dimension, 0) * weights["nemotron"])

            final_scores[dimension] = sum(scores) / sum(w for w in weights.values() if w > 0)

        final_scores["overall"] = np.mean(list(final_scores.values()))

        evaluation = EvaluationResult(
            overall_score=final_scores["overall"],
            dimension_scores=final_scores,
            confidence=0.85,
            model_contributions={
                "gpt4": state["gpt4_scores"],
                "claude": state["claude_scores"],
                "nemotron": state["nemotron_analysis"]
            }
        )

        return {"final_evaluation": evaluation}

    async def _improve_node(self, state: PaperState) -> Dict:
        """개선 계획 생성"""
        evaluation = state["final_evaluation"]

        # 약점 식별
        weak_dimensions = [
            dim for dim, score in evaluation.dimension_scores.items()
            if score < 8.0 and dim != "overall"
        ]

        if not weak_dimensions:
            return {"improvement_plan": None}

        # Nemotron으로 개선안 생성
        prompt = f"""
        The paper scored:
        {evaluation.dimension_scores}

        Weak areas: {weak_dimensions}

        Suggest 3-5 specific, actionable improvements.
        Return as JSON list.
        """

        response = await self.nemotron_llm.ainvoke(prompt)
        suggestions = self._parse_suggestions(response.content)

        plan = ImprovementPlan(
            weak_dimensions=weak_dimensions,
            suggestions=suggestions,
            priority_order=self._prioritize_suggestions(suggestions, evaluation)
        )

        return {"improvement_plan": plan}

    async def analyze_paper(
        self,
        paper: ProcessedPaper,
        query: str = "Evaluate this scientific paper comprehensively"
    ) -> AnalysisResult:
        """논문 분석 실행"""

        # LangGraph 워크플로우 실행
        result = await self.workflow.ainvoke({
            "paper": paper,
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "gpt4_scores": None,
            "claude_scores": None,
            "nemotron_analysis": None,
            "final_evaluation": None,
            "improvement_plan": None
        })

        return AnalysisResult(
            evaluation=result["final_evaluation"],
            improvement_plan=result["improvement_plan"],
            retrieved_context=result["reranked_docs"]
        )
```

### 5.2 FastAPI 통합

```python
# src/api/v1/hybrid_endpoints.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from src.services.hybrid_rag_service import HybridRAGService
from src.services.document_processor import DocumentProcessor

router = APIRouter(prefix="/api/v1/hybrid", tags=["hybrid-rag"])

# 서비스 초기화
rag_service = HybridRAGService(
    nemotron_url=settings.NEMOTRON_URL,
    embedder_url=settings.EMBEDDER_URL,
    reranker_url=settings.RERANKER_URL,
    chromadb_client=chromadb_client,
    openai_api_key=settings.OPENAI_API_KEY,
    anthropic_api_key=settings.ANTHROPIC_API_KEY
)

doc_processor = DocumentProcessor()

@router.post("/analyze-paper")
async def analyze_paper(
    file: UploadFile = File(...),
    use_hybrid: bool = True
):
    """논문 분석 API (하이브리드 모드)"""

    try:
        # 1. 문서 전처리
        paper = await doc_processor.process_upload(file)

        # 2. 하이브리드 분석 실행
        result = await rag_service.analyze_paper(paper)

        # 3. 결과 반환
        return {
            "status": "success",
            "evaluation": {
                "overall_score": result.evaluation.overall_score,
                "dimension_scores": result.evaluation.dimension_scores,
                "confidence": result.evaluation.confidence,
                "model_contributions": result.evaluation.model_contributions
            },
            "improvement_plan": {
                "weak_dimensions": result.improvement_plan.weak_dimensions,
                "suggestions": result.improvement_plan.suggestions,
                "priority_order": result.improvement_plan.priority_order
            } if result.improvement_plan else None,
            "metadata": {
                "processing_time": result.processing_time,
                "tokens_used": result.tokens_used,
                "cost_estimate": result.cost_estimate
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve-and-rerank")
async def retrieve_and_rerank(query: str, top_k: int = 5):
    """검색 + 재순위화 테스트 엔드포인트"""

    # Embedding
    query_vector = await rag_service.embedder.aembed_query(query)

    # Retrieve
    docs = await rag_service.vectorstore.asimilarity_search_by_vector(
        query_vector, k=top_k * 2
    )

    # Rerank
    reranked = rag_service.reranker.compress_documents(
        documents=docs,
        query=query
    )

    return {
        "query": query,
        "retrieved_count": len(docs),
        "reranked_results": [
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata,
                "relevance_score": doc.metadata.get("relevance_score", 0)
            }
            for doc in reranked[:top_k]
        ]
    }
```

---

## 📚 Part 6: 참고 자료 및 다음 단계

### 6.1 핵심 문서 링크

1. **NVIDIA Nemotron**
   - [Nemotron 개발자 페이지](https://developer.nvidia.com/nemotron)
   - [Nemotron Technical Report](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf)

2. **NeMo Retriever**
   - [NeMo Retriever 문서](https://developer.nvidia.com/nemo-retriever)
   - [EmbedQA Getting Started](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/getting-started.html)

3. **LangGraph**
   - [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
   - [Agentic RAG with LangGraph 가이드](https://www.analyticsvidhya.com/blog/2024/07/building-agentic-rag-systems-with-langgraph/)

4. **NVIDIA NIM**
   - [NIM Deployment Guide](https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html)
   - [NIM Operator for Kubernetes](https://github.com/nvidia/nim-operator)

### 6.2 구현 우선순위

#### Phase 1: 기반 구축 (Week 1-2)
- ✅ Docker Compose로 로컬 Nemotron + NeMo Retriever 배포
- ✅ ChromaDB 통합 및 기존 데이터 마이그레이션
- ✅ 기본 LangGraph 워크플로우 구현
- ✅ Reranking 기능 추가 및 검증

#### Phase 2: 하이브리드 통합 (Week 3-4)
- ✅ GPT-4/Claude + Nemotron 하이브리드 서비스 구현
- ✅ FastAPI 엔드포인트 추가
- ✅ A/B 테스트 프레임워크 구축
- ✅ 성능 메트릭 수집 시작

#### Phase 3: 검증 및 최적화 (Week 5-6)
- ✅ 50+ 논문으로 품질 회귀 테스트
- ✅ 성능 벤치마크 및 최적화
- ✅ 배치 처리 및 캐싱 구현
- ✅ 프로덕션 준비 체크리스트 완료

#### Phase 4: Kubernetes 배포 (Week 7-8)
- ✅ NIM Operator 설정
- ✅ HPA 및 모니터링 구성
- ✅ 프로덕션 배포 및 검증

### 6.3 주요 의사결정 포인트

1. **Week 2 Decision**: Reranking 품질 평가
   - 목표: MRR > 0.85
   - Pass → Phase 2 진행
   - Fail → 파라미터 튜닝 또는 대안 검토

2. **Week 4 Decision**: 하이브리드 vs Full Nemotron
   - A/B 테스트 결과 분석
   - 품질 회귀 < -0.1 → 하이브리드 유지
   - 품질 유지 + 비용 절감 > 50% → Full Nemotron 고려

3. **Week 6 Decision**: 프로덕션 배포 승인
   - 모든 품질 체크 통과
   - 성능 목표 달성 (throughput > 100 tok/s)
   - 비용 분석 승인

### 6.4 리스크 관리

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|-----------|
| Nemotron 품질 부족 | Medium | High | 하이브리드 모드로 시작, 단계적 전환 |
| GPU 리소스 부족 | Low | Medium | 클라우드 GPU 대여 or 처리량 제한 |
| 통합 복잡도 과다 | Medium | Medium | 단순한 아키텍처 우선, 점진적 확장 |
| 배포 지연 | Low | Low | 명확한 마일스톤, 위험 신호 조기 감지 |

---

## 🎯 결론 및 권장사항

### 최종 권장 아키텍처: **하이브리드 접근법**

```
Core Evaluation (High Quality)        Retrieval & Processing (Efficient)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPT-4 / Claude (175B+)                Nemotron + NeMo Retriever
• 논문 품질 평가                       • 문서 임베딩 (EmbedQA 1B)
• Novelty 판단                        • 검색 결과 재순위화 (RerankQA 1B)
• 복잡한 추론                          • 요약 및 정보 추출 (Nemotron 9B)
• 개선 제안 검증                       • 패턴 매칭 및 분류

→ API 기반, 즉시 사용                  → 온프레미스, 비용 효율적
→ 최고 품질 보장                       → 높은 처리량
```

### 핵심 이점
1. ✅ **품질 유지**: GPT-4/Claude로 핵심 평가 품질 보장
2. ✅ **비용 최적화**: Nemotron으로 30-50% API 비용 절감
3. ✅ **검색 개선**: Reranking으로 25-40% 검색 품질 향상
4. ✅ **유연성**: 태스크별 최적 모델 선택
5. ✅ **확장성**: Kubernetes 기반 자동 스케일링

### 시작하기
```bash
# 1. 환경 설정
git clone https://github.com/your-org/AI-CoScientist.git
cd AI-CoScientist
cp .env.example .env.hybrid
# Edit .env.hybrid with NGC_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

# 2. Docker Compose로 시작
docker-compose -f docker-compose.nemotron.yml up -d

# 3. 서비스 확인
curl http://localhost:8000/health  # Nemotron LLM
curl http://localhost:8001/health  # NeMo Embedder
curl http://localhost:8002/health  # NeMo Reranker

# 4. 첫 논문 분석
python scripts/analyze_with_hybrid.py paper.docx
```

**다음 단계**: `claudedocs/HYBRID_IMPLEMENTATION_GUIDE.md` 참조
