# 🚀 서울대 심리학과 RAG 시스템 구현 로드맵

## 📋 프로젝트 개요

**프로젝트명**: Psych-CoScientist
**기간**: 9주 (약 2.5개월)
**팀 구성**: 풀스택 개발자 2명 + AI/ML 엔지니어 1명
**목표**: 서울대 심리학과 전용 지능형 연구 지원 시스템 구축

## 🗓️ 상세 구현 일정

### 📊 Phase 1: 데이터 인프라 구축 (Week 1-2)

#### Week 1: 환경 설정 및 데이터 분석
```bash
# Day 1-2: 개발 환경 구축
□ 프로젝트 브랜치 생성 (feature/psychology-rag)
□ Docker 환경 확장 (Korean language support)
□ 새로운 Python 의존성 설치
  - transformers[torch]
  - sentence-transformers
  - konlpy
  - PyMuPDF
  - python-docx

# Day 3-5: 데이터 분석 및 처리
□ 66편 PDF 논문 메타데이터 추출
□ 교수별 연구 분야 분류
□ 데이터 품질 검사 및 정제
```

**핵심 스크립트 개발:**
```python
# scripts/analyze_psychology_data.py
import asyncio
from pathlib import Path
import pandas as pd

async def analyze_psychology_corpus():
    """심리학과 논문 코퍼스 분석"""

    data_analysis = {
        'total_papers': 0,
        'faculty_distribution': {},
        'research_domains': {},
        'temporal_distribution': {},
        'language_distribution': {}
    }

    # 교수별 논문 분포
    psychology_path = Path("data/심리학과")
    for faculty_dir in psychology_path.iterdir():
        if faculty_dir.is_dir():
            pdf_count = len(list(faculty_dir.glob("*.pdf")))
            data_analysis['faculty_distribution'][faculty_dir.name] = pdf_count

    # 결과 저장
    with open("data/psychology_corpus_analysis.json", "w") as f:
        json.dump(data_analysis, f, ensure_ascii=False, indent=2)

    return data_analysis
```

#### Week 2: 데이터베이스 확장
```sql
-- 새로운 테이블 생성
CREATE TABLE psychology_faculty (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    department VARCHAR(100) DEFAULT '심리학과',
    research_areas TEXT[],
    email VARCHAR(150),
    office_location VARCHAR(100),
    h_index INTEGER DEFAULT 0,
    total_citations INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE psychology_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    faculty_id UUID REFERENCES psychology_faculty(id),
    abstract TEXT,
    keywords TEXT[],
    korean_keywords TEXT[],
    publication_year INTEGER,
    journal VARCHAR(200),
    doi VARCHAR(100),
    file_path VARCHAR(500),
    domain_classification VARCHAR(100),
    methodology_tags TEXT[],
    citation_count INTEGER DEFAULT 0,
    language VARCHAR(10) DEFAULT 'ko',
    indexed_at TIMESTAMP DEFAULT NOW()
);
```

**ChromaDB 컬렉션 설정:**
```python
# scripts/setup_psychology_chromadb.py
import chromadb

def setup_psychology_collections():
    client = chromadb.Client()

    # 논문 전문 컬렉션
    papers_collection = client.create_collection(
        name="psychology_papers",
        metadata={
            "hnsw:space": "cosine",
            "description": "서울대 심리학과 논문 전문",
            "embedding_model": "jhgan/ko-sbert-nli"
        }
    )

    # 연구 메타데이터 컬렉션
    metadata_collection = client.create_collection(
        name="psychology_metadata",
        metadata={
            "hnsw:space": "cosine",
            "description": "논문 메타데이터 및 구조화 정보",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )

    return papers_collection, metadata_collection
```

### 🧠 Phase 2: AI/RAG 엔진 개발 (Week 3-5)

#### Week 3: 심리학 특화 NLP 파이프라인
```python
# src/services/psychology/nlp_pipeline.py
from transformers import AutoTokenizer, AutoModel
from konlpy.tag import Okt

class PsychologyNLPPipeline:
    def __init__(self):
        # 한국어 심리학 전문 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
        self.model = AutoModel.from_pretrained("monologg/kobert")
        self.korean_analyzer = Okt()

        # 심리학 도메인 분류기
        self.domain_classifier = self._load_domain_classifier()

        # 전문용어 사전
        self.psych_terms = self._load_psychology_glossary()

    async def process_query(self, query: str) -> ProcessedQuery:
        """심리학 쿼리 처리"""

        # 1. 형태소 분석
        morphemes = self.korean_analyzer.morphs(query)

        # 2. 도메인 분류
        domain = await self.classify_psychology_domain(query)

        # 3. 전문용어 인식
        entities = self.extract_psychology_entities(query)

        # 4. 쿼리 확장
        expanded_query = self.expand_psychology_terms(query)

        return ProcessedQuery(
            original=query,
            morphemes=morphemes,
            domain=domain,
            entities=entities,
            expanded_query=expanded_query
        )
```

#### Week 4: 심리학 RAG 전략 구현
```python
# src/services/rag/psychology_rag_strategy.py
class PsychologyRAGStrategy(RAGStrategy):
    name = "psychology_specialized"

    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        # 1. 한국어-영어 하이브리드 검색
        korean_results = await self._korean_semantic_search(query)
        english_results = await self._english_semantic_search(query)

        # 2. 도메인 가중치 적용
        domain_weighted = self._apply_domain_weights(
            korean_results + english_results
        )

        # 3. 심리학 특화 재랭킹
        reranked = await self._psychology_rerank(domain_weighted)

        return reranked[:kwargs.get('top_k', 10)]

    async def _korean_semantic_search(self, query: str):
        """한국어 의미 검색"""
        # Ko-SBERT 임베딩 사용
        query_embedding = self.korean_embedder.encode(query)

        results = await self.vector_store.similarity_search(
            collection="psychology_papers",
            query_vector=query_embedding,
            where={"language": "ko"}
        )

        return results
```

#### Week 5: 심리학 전문 에이전트 개발
```python
# src/agents/psychology_expert.py
class PsychologyExpertAgent(ResearchAgent):
    def __init__(self):
        super().__init__()
        self.specializations = [
            "cognitive_psychology",
            "social_psychology",
            "developmental_psychology",
            "clinical_psychology"
        ]

        # 심리학 특화 프롬프트 템플릿
        self.prompt_templates = {
            'literature_review': self._load_literature_prompt(),
            'methodology_advice': self._load_methodology_prompt(),
            'ethics_review': self._load_ethics_prompt()
        }

    async def process_psychology_task(self, task: PsychologyTask):
        """심리학 연구 작업 처리"""

        # 작업 유형 분류
        task_type = self.classify_task_type(task.description)

        # 관련 논문 검색
        relevant_papers = await self.search_relevant_literature(
            query=task.research_question,
            domain=task.psychology_domain
        )

        # 전문가 응답 생성
        response = await self.generate_expert_response(
            task_type=task_type,
            context=relevant_papers,
            user_query=task.description
        )

        return response
```

### 💬 Phase 3: 챗봇 인터페이스 개발 (Week 6-7)

#### Week 6: Backend API 개발
```python
# src/api/v1/psychology/chat.py
from fastapi import APIRouter, WebSocket, Depends
from src.services.psychology.conversation_manager import ConversationManager

router = APIRouter()

@router.post("/chat/message")
async def process_chat_message(
    request: PsychologyMessageRequest,
    user: User = Depends(get_current_user)
):
    """심리학 챗봇 메시지 처리"""

    conversation_manager = ConversationManager()

    # 대화 컨텍스트 로드
    context = await conversation_manager.load_context(
        conversation_id=request.conversation_id,
        user_id=user.id
    )

    # 심리학 전문 처리
    response = await process_psychology_query(
        query=request.message,
        context=context,
        user_preferences=user.psychology_preferences
    )

    # 대화 기록 저장
    await conversation_manager.save_message(
        conversation_id=request.conversation_id,
        user_message=request.message,
        bot_response=response
    )

    return PsychologyMessageResponse(
        message=response.text,
        sources=response.citations,
        suggestions=response.follow_up_questions,
        visualization=response.viz_data
    )

@router.websocket("/chat/ws/{user_id}")
async def psychology_chat_websocket(websocket: WebSocket, user_id: str):
    """실시간 심리학 챗봇"""
    await websocket.accept()

    async for message in websocket.iter_text():
        # 스트리밍 응답
        async for chunk in stream_psychology_response(message):
            await websocket.send_json({
                "type": "chunk",
                "data": chunk
            })
```

#### Week 7: React 프론트엔드 개발
```typescript
// frontend/src/psychology/components/PsychologyChat.tsx
import React, { useState, useEffect } from 'react'
import { usePsychologyChat } from '../hooks/usePsychologyChat'

export const PsychologyChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputText, setInputText] = useState('')
  const { sendMessage, isLoading, streamResponse } = usePsychologyChat()

  const handleSendMessage = async () => {
    if (!inputText.trim()) return

    // 사용자 메시지 추가
    const userMessage: ChatMessage = {
      id: generateId(),
      type: 'user',
      content: inputText,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])

    // 봇 응답 스트리밍
    const botMessage: ChatMessage = {
      id: generateId(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      sources: [],
      suggestions: []
    }
    setMessages(prev => [...prev, botMessage])

    // 실시간 응답 수신
    await streamResponse(inputText, (chunk) => {
      setMessages(prev => prev.map(msg =>
        msg.id === botMessage.id
          ? { ...msg, content: msg.content + chunk }
          : msg
      ))
    })

    setInputText('')
  }

  return (
    <div className="psychology-chat-container">
      <ChatHeader title="심리학 연구 어시스턴트" />

      <MessageArea messages={messages} />

      <InputArea
        value={inputText}
        onChange={setInputText}
        onSend={handleSendMessage}
        isLoading={isLoading}
      />

      <QuickActions>
        <ActionButton
          icon="📚"
          text="논문 검색"
          onClick={() => setInputText("인지심리학 관련 최신 논문을 찾아주세요")}
        />
        <ActionButton
          icon="🧪"
          text="연구 설계"
          onClick={() => setInputText("실험 설계에 대해 조언해주세요")}
        />
        <ActionButton
          icon="👥"
          text="협업 매칭"
          onClick={() => setInputText("공동연구자를 찾고 있습니다")}
        />
      </QuickActions>
    </div>
  )
}
```

### 🎨 Phase 4: 고급 기능 및 시각화 (Week 8)

#### 연구 네트워크 시각화
```typescript
// frontend/src/psychology/components/ResearchNetwork.tsx
import { Network } from 'vis-network'

export const ResearchNetworkVisualization: React.FC = () => {
  const [networkData, setNetworkData] = useState<NetworkData>()

  useEffect(() => {
    const nodes = new DataSet([
      { id: 1, label: '안우영 교수', group: 'faculty', domain: 'addiction' },
      { id: 2, label: '이수현 교수', group: 'faculty', domain: 'neuroscience' },
      { id: 3, label: '한소원 교수', group: 'faculty', domain: 'ergonomics' }
    ])

    const edges = new DataSet([
      { from: 1, to: 2, label: '공동연구 2편' },
      { from: 2, to: 3, label: '학제간 프로젝트' }
    ])

    const network = new Network(containerRef.current, { nodes, edges }, options)
  }, [])

  return (
    <div className="research-network">
      <h3>심리학과 연구 협업 네트워크</h3>
      <div ref={containerRef} style={{ height: '600px' }} />
    </div>
  )
}
```

### 🚀 Phase 5: 배포 및 최적화 (Week 9)

#### 성능 최적화
```python
# src/services/psychology/performance_optimizer.py
class PsychologyPerformanceOptimizer:
    def __init__(self):
        self.cache_manager = RedisCacheManager()
        self.model_cache = {}

    async def optimize_search_latency(self):
        """검색 지연시간 최적화"""

        # 1. 임베딩 캐싱
        await self.cache_frequent_embeddings()

        # 2. 결과 캐싱
        await self.cache_popular_queries()

        # 3. 모델 워밍업
        await self.warmup_models()

    async def cache_frequent_embeddings(self):
        """자주 사용되는 임베딩 캐싱"""
        frequent_terms = [
            "인지심리학", "사회심리학", "발달심리학",
            "실험설계", "통계분석", "연구윤리"
        ]

        for term in frequent_terms:
            embedding = await self.embedding_service.encode(term)
            await self.cache_manager.set(
                f"embedding:{term}",
                embedding,
                expire=3600 * 24  # 24시간 캐시
            )
```

#### Docker 배포 설정
```yaml
# docker-compose.psychology.yml
version: '3.8'

services:
  psychology-backend:
    build:
      context: .
      dockerfile: Dockerfile.psychology
    ports:
      - "8001:8000"
    environment:
      - PSYCHOLOGY_ENABLED=true
      - KOREAN_NLP_ENABLED=true
      - CHROMADB_HOST=chromadb
    volumes:
      - "./data/심리학과:/app/data/psychology:ro"
      - "./models:/app/models"
    depends_on:
      - postgres
      - redis
      - chromadb

  psychology-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.psychology
    ports:
      - "3001:3000"
    environment:
      - REACT_APP_API_URL=http://psychology-backend:8000
      - REACT_APP_WS_URL=ws://psychology-backend:8000/ws
```

## 📊 마일스톤 및 성공 지표

### 🎯 주요 마일스톤

| Week | 마일스톤 | 성공 기준 |
|------|---------|-----------|
| 2 | 데이터 인프라 구축 완료 | 66편 논문 100% 인덱싱 |
| 5 | RAG 엔진 기본 기능 완료 | 검색 정확도 >80% |
| 7 | 챗봇 베타 버전 완료 | 기본 대화 기능 동작 |
| 8 | 고급 기능 통합 완료 | 시각화 및 분석 기능 |
| 9 | 프로덕션 배포 완료 | 안정적 서비스 운영 |

### 📈 성능 지표 (KPIs)

**기술적 성능:**
- 응답 시간: < 2초 (90%ile)
- 검색 정확도: > 85% (Top-5)
- 시스템 가용성: > 99.5%
- 동시 사용자: 50명 지원

**사용자 경험:**
- 답변 관련성: > 90% (사용자 피드백)
- 인용 정확도: > 95%
- 사용자 만족도: > 4.0/5.0

**데이터 품질:**
- 논문 커버리지: 100% (66편)
- 메타데이터 완성도: > 95%
- 한국어 처리 정확도: > 90%

## 🛠️ 개발 도구 및 환경

### 필수 개발 도구
```bash
# Backend
pip install fastapi uvicorn sqlalchemy alembic
pip install chromadb sentence-transformers transformers
pip install konlpy PyMuPDF python-docx

# Frontend
npm install react typescript @chakra-ui/react
npm install @tanstack/react-query axios
npm install vis-network d3 plotly.js

# Development
pip install pytest black ruff mypy
pip install pre-commit jupyter notebook
```

### 개발 환경 설정
```bash
# 1. 프로젝트 브랜치 생성
git checkout -b feature/psychology-rag

# 2. 개발 환경 구축
python -m venv venv-psychology
source venv-psychology/bin/activate
pip install -r requirements-psychology.txt

# 3. 데이터 준비
python scripts/setup_psychology_data.py

# 4. 개발 서버 실행
uvicorn src.main:app --reload --port 8001
```

## 🧪 테스트 전략

### 단위 테스트 (Week 3-8)
```python
# tests/psychology/test_nlp_pipeline.py
import pytest
from src.services.psychology.nlp_pipeline import PsychologyNLPPipeline

class TestPsychologyNLP:
    def setup_method(self):
        self.nlp = PsychologyNLPPipeline()

    async def test_korean_query_processing(self):
        query = "인지편향에 대한 실험연구를 찾고 있습니다"
        result = await self.nlp.process_query(query)

        assert result.domain == "cognitive_psychology"
        assert "실험" in result.methodologies
        assert "cognitive bias" in result.expanded_query

    async def test_domain_classification(self):
        queries = [
            ("아동 발달 연구", "developmental_psychology"),
            ("사회적 편견 연구", "social_psychology"),
            ("뇌영상 분석", "biological_psychology")
        ]

        for query, expected_domain in queries:
            result = await self.nlp.classify_domain(query)
            assert result == expected_domain
```

### 통합 테스트 (Week 7-8)
```python
# tests/psychology/test_end_to_end.py
async def test_complete_chat_workflow():
    """전체 챗봇 워크플로우 테스트"""

    # 1. 사용자 쿼리
    user_query = "학습과 기억에 관한 최신 연구를 알려주세요"

    # 2. API 호출
    response = await client.post("/api/v1/psychology/chat/message", json={
        "message": user_query,
        "conversation_id": test_conversation_id
    })

    # 3. 응답 검증
    assert response.status_code == 200
    data = response.json()

    assert len(data["sources"]) > 0
    assert "학습" in data["message"] or "기억" in data["message"]
    assert len(data["suggestions"]) > 0
```

## 📚 문서화 계획

### 기술 문서
- [ ] API 문서 (OpenAPI/Swagger)
- [ ] 아키텍처 가이드
- [ ] 배포 매뉴얼
- [ ] 성능 튜닝 가이드

### 사용자 문서
- [ ] 사용자 매뉴얼 (한국어)
- [ ] 튜토리얼 비디오
- [ ] FAQ 및 트러블슈팅
- [ ] 연구 활용 가이드

## 🚨 리스크 관리

### 기술적 리스크
| 리스크 | 확률 | 영향도 | 완화 방안 |
|--------|------|--------|-----------|
| 한국어 NLP 성능 부족 | 중간 | 높음 | 다중 모델 앙상블, 영어 번역 백업 |
| 임베딩 품질 문제 | 낮음 | 중간 | A/B 테스트, 점진적 모델 교체 |
| 확장성 문제 | 낮음 | 높음 | 로드 테스트, 캐싱 전략 |

### 비즈니스 리스크
| 리스크 | 확률 | 영향도 | 완화 방안 |
|--------|------|--------|-----------|
| 사용자 채택률 저조 | 중간 | 높음 | 조기 프로토타입, 사용자 피드백 |
| 데이터 품질 문제 | 낮음 | 중간 | 데이터 검증, 수동 큐레이션 |
| 저작권 이슈 | 낮음 | 높음 | Fair Use 가이드라인, 법무 검토 |

## 🎉 예상 성과

### 단기 성과 (3개월)
- **연구 효율성 향상**: 문헌 검색 시간 50% 단축
- **지식 접근성 개선**: 66편 논문 즉시 검색 가능
- **연구 협업 촉진**: 교수진 간 연구 연결점 발굴

### 중장기 성과 (6-12개월)
- **연구 질 향상**: 보다 포괄적인 문헌 검토 지원
- **학제간 연구 증가**: 도메인 간 연결 인사이트 제공
- **국제 경쟁력 강화**: 최신 AI 기술 활용 연구 환경

이 로드맵은 체계적이고 실현 가능한 일정으로 서울대 심리학과만의 혁신적인 연구 지원 시스템을 구축할 것입니다. 기존 AI-CoScientist의 강력한 인프라를 기반으로, 심리학 연구의 특수성을 완벽히 반영한 전문적인 솔루션이 될 것입니다.