# 심리학 시스템의 기존 AI-CoScientist 인프라 통합 계획

## 🔄 통합 아키텍처 개요

### 기존 시스템 활용 전략

```
🏗️ AI-CoScientist Infrastructure (기존)
├── 📊 데이터 레이어
│   ├── PostgreSQL (확장)
│   ├── ChromaDB (새 컬렉션 추가)
│   └── Redis (공유)
├── 🤖 에이전트 시스템
│   ├── AgentPool (새 에이전트 등록)
│   ├── UnifiedRAGOrchestrator (전략 추가)
│   └── LangGraph (워크플로 확장)
├── 🧠 RAG 시스템
│   ├── 6개 기존 전략 (재사용)
│   ├── Psychology RAG Strategy (신규)
│   └── RAGAS 평가 (공유)
└── 🌐 API & UI
    ├── FastAPI 라우터 (확장)
    ├── WebSocket (공유)
    └── React Frontend (새 모듈)

➕ Psychology Extension (신규)
├── 🧠 심리학 특화 컴포넌트
├── 📚 심리학 논문 처리
├── 💬 전문 챗봇 인터페이스
└── 🔬 연구 지원 도구
```

## 📂 파일 시스템 통합

### 1. 새로운 디렉토리 구조

```bash
src/
├── services/
│   ├── rag/
│   │   ├── psychology_rag_strategy.py        # 신규
│   │   ├── psychology_document_processor.py  # 신규
│   │   └── domain_classifier.py              # 신규
│   └── psychology/                           # 신규 모듈
│       ├── __init__.py
│       ├── chatbot/
│       │   ├── conversation_manager.py
│       │   ├── nlp_pipeline.py
│       │   └── response_generator.py
│       ├── research/
│       │   ├── paper_analyzer.py
│       │   ├── trend_analyzer.py
│       │   └── collaboration_matcher.py
│       └── ethics/
│           ├── ethics_reviewer.py
│           └── privacy_checker.py
├── agents/
│   ├── psychology_expert.py                  # 신규
│   ├── literature_reviewer.py                # 신규
│   └── methodology_advisor.py                # 신규
├── api/v1/
│   └── psychology/                           # 신규 라우터
│       ├── __init__.py
│       ├── chat.py
│       ├── papers.py
│       └── research.py
└── frontend/
    └── psychology/                           # 신규 React 모듈
        ├── components/
        ├── hooks/
        └── pages/
```

### 2. 데이터베이스 마이그레이션

```python
# alembic/versions/add_psychology_tables.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    """심리학 시스템 테이블 추가"""

    # 심리학과 교수진 테이블
    op.create_table(
        'psychology_faculty',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('email', sa.String(150)),
        sa.Column('research_areas', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    # 기존 papers 테이블에 psychology 관련 컬럼 추가
    op.add_column('papers', sa.Column('psychology_domain', sa.String(100)))
    op.add_column('papers', sa.Column('faculty_id', postgresql.UUID(as_uuid=True)))
    op.add_column('papers', sa.Column('korean_keywords', postgresql.ARRAY(sa.String)))

    # 심리학 챗봇 대화 기록
    op.create_table(
        'psychology_conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(200)),
        sa.Column('domain_focus', sa.String(100)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
```

## 🤖 에이전트 시스템 통합

### 1. 새로운 심리학 에이전트 등록

```python
# src/agents/psychology_expert.py
from src.agents.base import ResearchAgent
from src.services.psychology.nlp_pipeline import PsychologyNLPPipeline

class PsychologyExpertAgent(ResearchAgent):
    """심리학 전문 연구 에이전트"""

    agent_id = "psychology_expert"
    name = "Psychology Research Expert"
    description = "서울대 심리학과 전문 연구 지원 에이전트"

    specializations = [
        "cognitive_psychology",
        "social_psychology",
        "developmental_psychology",
        "clinical_psychology",
        "biological_psychology"
    ]

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.nlp_pipeline = PsychologyNLPPipeline()
        self.domain_expertise = self._load_domain_knowledge()

    async def process_task(self, task: ResearchTask) -> AgentResponse:
        """심리학 연구 작업 처리"""

        # 도메인 특화 쿼리 처리
        processed_query = self.nlp_pipeline.process_query(task.description)

        # 적절한 RAG 전략 선택
        rag_strategy = self._select_psychology_strategy(processed_query.domain)

        # 검색 실행
        results = await self.rag_orchestrator.search(
            query=processed_query.enhanced_query,
            strategy=rag_strategy,
            collection="psychology_papers"
        )

        # 심리학 특화 응답 생성
        response = await self._generate_psychology_response(
            query=processed_query,
            results=results,
            task=task
        )

        return AgentResponse(
            agent_id=self.agent_id,
            content=response.text,
            sources=response.citations,
            confidence=response.confidence,
            metadata={
                'psychology_domain': processed_query.domain,
                'methodology_suggestions': response.methods,
                'ethics_considerations': response.ethics_notes
            }
        )
```

### 2. 에이전트 풀 확장

```python
# src/agents/pool.py 수정
class AgentPool:
    def __init__(self):
        # 기존 에이전트들...
        self.agents.update({
            'psychology_expert': PsychologyExpertAgent,
            'literature_reviewer': LiteratureReviewAgent,
            'methodology_advisor': MethodologyAdvisorAgent,
        })

    def get_optimal_agent_team(self, task: ResearchTask) -> List[str]:
        """작업에 최적화된 에이전트 팀 선택"""

        # 기존 로직...

        # 심리학 관련 작업 감지
        if self._is_psychology_task(task):
            team.append('psychology_expert')

            # 세부 작업에 따른 추가 에이전트
            if 'literature review' in task.description.lower():
                team.append('literature_reviewer')

            if any(method in task.description.lower() for method in ['실험', '연구설계', 'methodology']):
                team.append('methodology_advisor')

        return team

    def _is_psychology_task(self, task: ResearchTask) -> bool:
        """심리학 관련 작업인지 판단"""
        psychology_keywords = [
            '심리', 'psychology', '인지', '행동', '뇌', '신경',
            '발달', '사회심리', '임상', '상담', '학습', '기억'
        ]
        return any(keyword in task.description.lower() for keyword in psychology_keywords)
```

## 🧠 RAG 시스템 통합

### 1. 심리학 특화 RAG 전략

```python
# src/services/rag/psychology_rag_strategy.py
from src.services.rag.base import RAGStrategy
from src.services.psychology.document_processor import PsychologyDocumentProcessor

class PsychologyRAGStrategy(RAGStrategy):
    """심리학 특화 RAG 전략"""

    name = "psychology_specialized"
    description = "서울대 심리학과 논문 특화 검색 전략"

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.document_processor = PsychologyDocumentProcessor()
        self.collections = ["psychology_papers", "psychology_metadata"]

    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """심리학 특화 검색"""

        # 1. 쿼리 전처리 (심리학 용어 확장)
        enhanced_query = self._enhance_psychology_query(query)

        # 2. 도메인별 가중치 적용
        domain_weights = self._calculate_domain_weights(query)

        # 3. 하이브리드 검색 (Dense + Sparse)
        dense_results = await self._dense_search(enhanced_query, **kwargs)
        sparse_results = await self._sparse_search(enhanced_query, **kwargs)

        # 4. 심리학 특화 재랭킹
        reranked_results = self._psychology_rerank(
            dense_results + sparse_results,
            original_query=query,
            domain_weights=domain_weights
        )

        return reranked_results

    def _enhance_psychology_query(self, query: str) -> str:
        """심리학 쿼리 향상"""
        # 한국어-영어 용어 매핑
        term_mappings = {
            '인지': 'cognitive cognition',
            '학습': 'learning acquisition',
            '기억': 'memory recall',
            '주의': 'attention focus',
            # ... 더 많은 매핑
        }

        enhanced_query = query
        for kr_term, en_terms in term_mappings.items():
            if kr_term in query:
                enhanced_query += f" {en_terms}"

        return enhanced_query
```

### 2. UnifiedRAGOrchestrator에 전략 등록

```python
# src/services/rag/unified_rag_orchestrator.py 수정
class UnifiedRAGOrchestrator:
    def __init__(self):
        # 기존 전략들...
        self.strategies.update({
            'psychology_specialized': PsychologyRAGStrategy,
        })

        # 심리학 특화 쿼리 라우팅 규칙
        self.routing_rules.update({
            'psychology_patterns': [
                r'심리.*연구',
                r'인지.*실험',
                r'behavioral.*study',
                r'neural.*mechanism'
            ]
        })

    def _route_query(self, query: str, context: Dict) -> str:
        """쿼리 라우팅 로직"""

        # 기존 라우팅...

        # 심리학 패턴 감지
        if any(re.search(pattern, query, re.IGNORECASE)
               for pattern in self.routing_rules['psychology_patterns']):
            return 'psychology_specialized'

        return self._default_routing(query, context)
```

## 🌐 API 및 프론트엔드 통합

### 1. 새로운 API 라우터 등록

```python
# src/main.py 수정
from src.api.v1.psychology import psychology_router

app = FastAPI(title="AI-CoScientist")

# 기존 라우터들...
app.include_router(api_router, prefix="/api/v1")

# 심리학 전용 라우터 추가
app.include_router(psychology_router, prefix="/api/v1/psychology")
```

### 2. 프론트엔드 모듈 통합

```typescript
// frontend/src/psychology/PsychologyModule.tsx
import React from 'react'
import { Route, Routes } from 'react-router-dom'
import { PsychologyChat } from './components/PsychologyChat'
import { ResearchDashboard } from './components/ResearchDashboard'

export const PsychologyModule: React.FC = () => {
  return (
    <Routes>
      <Route path="/chat" element={<PsychologyChat />} />
      <Route path="/research" element={<ResearchDashboard />} />
      <Route path="/papers" element={<PaperExplorer />} />
      <Route path="/collaboration" element={<CollaborationNetwork />} />
    </Routes>
  )
}

// frontend/src/App.tsx 수정
function App() {
  return (
    <Router>
      <Routes>
        {/* 기존 라우트들... */}
        <Route path="/psychology/*" element={<PsychologyModule />} />
      </Routes>
    </Router>
  )
}
```

## 📊 데이터 마이그레이션 계획

### 1. 심리학과 논문 데이터 처리

```python
# scripts/migrate_psychology_data.py
import asyncio
from pathlib import Path
from src.services.psychology.document_processor import PsychologyDocumentProcessor

async def migrate_psychology_papers():
    """심리학과 PDF 논문들을 시스템에 등록"""

    processor = PsychologyDocumentProcessor()
    psychology_data_path = Path("data/심리학과")

    for faculty_dir in psychology_data_path.iterdir():
        if faculty_dir.is_dir():
            faculty_name = faculty_dir.name
            print(f"Processing {faculty_name}'s papers...")

            # 교수진 정보 등록
            faculty_id = await register_faculty(faculty_name)

            # 논문 파일 처리
            for pdf_file in faculty_dir.glob("*.pdf"):
                print(f"  Processing: {pdf_file.name}")

                # PDF 메타데이터 추출
                paper_data = await processor.extract_paper_metadata(
                    pdf_path=pdf_file,
                    faculty_id=faculty_id
                )

                # 데이터베이스 등록
                await register_paper(paper_data)

                # 벡터 데이터베이스 등록
                await index_paper_vectors(paper_data)

                print(f"    ✅ {paper_data.title}")

if __name__ == "__main__":
    asyncio.run(migrate_psychology_papers())
```

### 2. ChromaDB 컬렉션 설정

```python
# scripts/setup_psychology_collections.py
import chromadb
from src.services.knowledge_base.vector_store import VectorStoreManager

async def setup_psychology_collections():
    """심리학 전용 ChromaDB 컬렉션 생성"""

    vector_manager = VectorStoreManager()

    # 심리학 논문 컬렉션
    await vector_manager.create_collection(
        name="psychology_papers",
        metadata={
            "description": "서울대 심리학과 교수진 논문",
            "embedding_model": "all-MiniLM-L6-v2",
            "language": "ko-en"
        }
    )

    # 심리학 메타데이터 컬렉션
    await vector_manager.create_collection(
        name="psychology_metadata",
        metadata={
            "description": "논문 메타데이터 및 구조화된 정보",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )

    print("✅ Psychology collections created successfully")
```

## 🔧 설정 및 환경변수

### 1. 새로운 설정 추가

```python
# src/core/config.py 수정
class Settings(BaseSettings):
    # 기존 설정들...

    # Psychology-specific settings
    psychology_enabled: bool = True
    psychology_data_path: str = "data/심리학과"
    psychology_model_path: str = "models/psychology"
    psychology_domain_classifier: str = "distilbert-base-uncased"

    # Korean language support
    korean_tokenizer: str = "monologg/kobert"
    korean_embedding_model: str = "jhgan/ko-sbert-nli"

    class Config:
        env_file = ".env"
        env_prefix = "AICOSCIENTIST_"
```

### 2. Docker 설정 업데이트

```yaml
# docker-compose.psychology.yml
version: '3.8'

services:
  ai-coscientist-psychology:
    build:
      context: .
      dockerfile: Dockerfile.psychology
    ports:
      - "8001:8000"
    environment:
      - PSYCHOLOGY_ENABLED=true
      - KOREAN_SUPPORT=true
    volumes:
      - "./data/심리학과:/app/data/psychology"
      - "./models/psychology:/app/models/psychology"
    depends_on:
      - postgres
      - redis
      - chromadb
```

## 🚀 점진적 배포 전략

### 1. 단계별 롤아웃

```python
# Phase 1: 기본 인프라 구축 (2주)
# ✅ 데이터베이스 마이그레이션
# ✅ ChromaDB 컬렉션 설정
# ✅ 기본 API 엔드포인트

# Phase 2: 심리학 전용 기능 (3주)
# ✅ Psychology RAG Strategy
# ✅ 전문 에이전트 개발
# ✅ 논문 처리 파이프라인

# Phase 3: 사용자 인터페이스 (2주)
# ✅ React 챗봇 모듈
# ✅ 연구 대시보드
# ✅ 시각화 컴포넌트

# Phase 4: 고도화 및 최적화 (2주)
# ✅ 성능 튜닝
# ✅ 한국어 지원 강화
# ✅ 사용자 테스트 및 피드백
```

### 2. A/B 테스트 설정

```python
class PsychologyFeatureFlags:
    """심리학 시스템 기능 플래그"""

    KOREAN_NLP_ENHANCED = "korean_nlp_v2"
    ADVANCED_COLLABORATION = "collab_matching_v2"
    ETHICS_AUTOMATION = "auto_ethics_review"

    @classmethod
    def is_enabled(cls, feature: str, user_id: str) -> bool:
        """사용자별 기능 플래그 확인"""
        # 특정 사용자 그룹에게만 새 기능 제공
        return FeatureFlagService.check(feature, user_id)
```

이 통합 계획은 기존 AI-CoScientist의 안정성을 유지하면서도 심리학 전문 기능을 점진적으로 추가할 수 있도록 설계되었습니다. 모든 새로운 컴포넌트는 기존 아키텍처와 조화롭게 통합되며, 독립적으로 개발 및 테스트 가능합니다.