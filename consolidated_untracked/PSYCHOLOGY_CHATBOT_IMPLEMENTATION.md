# 심리학 연구 챗봇 기술 구현 명세서

## 🤖 챗봇 핵심 기능 설계

### 1. 대화형 인터페이스 아키텍처

```typescript
// 챗봇 인터페이스 타입 정의
interface PsychologyResearchBot {
  // 기본 대화 기능
  chat: {
    processQuery(query: string, context: ChatContext): Promise<BotResponse>
    maintainMemory(conversationId: string): ConversationMemory
    generateFollowUp(response: BotResponse): string[]
  }

  // 전문 연구 기능
  research: {
    searchPapers(query: ResearchQuery): Promise<PaperResult[]>
    analyzeTrends(timeRange: DateRange, keywords: string[]): TrendAnalysis
    suggestCollaborations(researchArea: string): CollaborationSuggestion[]
    reviewMethodology(description: string): MethodologyFeedback
  }

  // 시각화 및 내보내기
  visualization: {
    generateCitationNetwork(papers: Paper[]): NetworkGraph
    createResearchTimeline(author: string): Timeline
    exportResults(format: 'pdf' | 'bibtex' | 'csv'): ExportData
  }
}
```

### 2. 대화 플로우 설계

```python
class PsychologyConversationFlow:
    """심리학 연구 특화 대화 플로우"""

    CONVERSATION_TYPES = {
        'LITERATURE_SEARCH': {
            'prompt_template': "심리학 연구 문헌을 검색해드리겠습니다. 어떤 주제에 관심이 있으신가요?",
            'follow_up_questions': [
                "특정 연구 방법론에 관심이 있나요?",
                "특정 연구자의 논문을 찾고 계신가요?",
                "최근 몇 년간의 연구에 집중하시겠어요?"
            ]
        },
        'RESEARCH_CONSULTATION': {
            'prompt_template': "연구 설계에 대해 상담해드리겠습니다. 연구 질문을 알려주세요.",
            'follow_up_questions': [
                "연구 대상(참여자)는 누구인가요?",
                "어떤 연구 방법을 고려하고 계신가요?",
                "윤리적 고려사항이 있나요?"
            ]
        },
        'COLLABORATION_MATCHING': {
            'prompt_template': "연구 협업을 위한 매칭을 도와드리겠습니다.",
            'follow_up_questions': [
                "어떤 전문성을 가진 연구자를 찾고 계신가요?",
                "학제간 연구에 관심이 있나요?",
                "국내외 협업을 모두 고려하시나요?"
            ]
        }
    }
```

### 3. 심리학 특화 NLP 파이프라인

```python
class PsychologyNLPPipeline:
    """심리학 도메인 특화 자연어 처리"""

    def __init__(self):
        self.domain_classifier = self._load_domain_classifier()
        self.entity_extractor = self._load_psychology_ner()
        self.intent_recognizer = self._load_intent_model()

    def process_query(self, query: str) -> ProcessedQuery:
        """사용자 쿼리를 심리학 도메인에 맞게 처리"""

        # 1. 도메인 분류 (인지심리, 사회심리, 발달심리 등)
        domain = self.domain_classifier.predict(query)

        # 2. 심리학 전문 용어 추출
        entities = self.entity_extractor.extract(query)

        # 3. 의도 파악 (검색, 분석, 상담, 매칭 등)
        intent = self.intent_recognizer.predict(query)

        # 4. 연구 방법론 키워드 감지
        methodologies = self._extract_methodologies(query)

        return ProcessedQuery(
            original=query,
            domain=domain,
            entities=entities,
            intent=intent,
            methodologies=methodologies,
            confidence=self._calculate_confidence(domain, entities, intent)
        )

    PSYCHOLOGY_DOMAINS = {
        'cognitive': ['인지', '기억', '학습', '지각', '주의', '언어'],
        'social': ['사회', '집단', '태도', '편견', '대인관계', '설득'],
        'developmental': ['발달', '아동', '청소년', '노화', '생애주기'],
        'clinical': ['임상', '치료', '우울', '불안', '트라우마', '정신건강'],
        'biological': ['뇌', '신경', '호르몬', '유전', 'fMRI', 'EEG'],
        'experimental': ['실험', '통계', '측정', '타당도', '신뢰도']
    }
```

## 🎨 사용자 인터페이스 설계

### 1. 웹 챗봇 인터페이스

```jsx
// React 컴포넌트 구조
const PsychologyResearchChatbot = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [isTyping, setIsTyping] = useState(false)
  const [currentContext, setCurrentContext] = useState<ResearchContext>()

  return (
    <div className="psychology-chatbot">
      <Header title="심리학 연구 어시스턴트" />

      <ChatArea>
        <MessageList messages={messages} />
        <TypingIndicator visible={isTyping} />
      </ChatArea>

      <InputArea>
        <MessageInput onSend={handleSendMessage} />
        <QuickActions>
          <ActionButton icon="📚" text="논문 검색" />
          <ActionButton icon="🧪" text="연구 설계" />
          <ActionButton icon="👥" text="협업 매칭" />
          <ActionButton icon="📊" text="트렌드 분석" />
        </QuickActions>
      </InputArea>

      <SidePanel>
        <SearchFilters />
        <CitationManager />
        <ExportOptions />
      </SidePanel>
    </div>
  )
}
```

### 2. 메시지 타입별 렌더링

```jsx
const MessageRenderer = ({ message, type }) => {
  switch (type) {
    case 'PAPER_RESULT':
      return <PaperCard paper={message.data} />

    case 'TREND_ANALYSIS':
      return <TrendChart data={message.data} />

    case 'COLLABORATION_SUGGESTION':
      return <CollaborationMatch researchers={message.data} />

    case 'METHODOLOGY_REVIEW':
      return <MethodologyFeedback feedback={message.data} />

    default:
      return <TextMessage content={message.text} />
  }
}

const PaperCard = ({ paper }) => (
  <div className="paper-card">
    <h3>{paper.title}</h3>
    <p className="authors">{paper.authors.join(', ')}</p>
    <p className="abstract">{paper.abstract_summary}</p>
    <div className="metadata">
      <span className="domain">{paper.domain}</span>
      <span className="year">{paper.year}</span>
      <span className="citations">{paper.citation_count} 인용</span>
    </div>
    <div className="actions">
      <Button onClick={() => openFullText(paper.id)}>전문 보기</Button>
      <Button onClick={() => addToCitation(paper)}>인용 추가</Button>
    </div>
  </div>
)
```

## 🔌 API 설계

### 1. RESTful API 엔드포인트

```python
# FastAPI 라우터 정의
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional

router = APIRouter(prefix="/api/v1/psychology", tags=["psychology"])

@router.post("/chat/message")
async def process_chat_message(
    request: ChatRequest,
    user: User = Depends(get_current_user)
) -> ChatResponse:
    """채팅 메시지 처리"""

    # 사용자 쿼리 처리
    processed_query = nlp_pipeline.process_query(request.message)

    # RAG 시스템으로 검색
    search_results = await rag_orchestrator.search(
        query=processed_query.enhanced_query,
        collection="psychology_papers",
        strategy="psychology_specialized"
    )

    # 응답 생성
    response = await generate_response(
        query=processed_query,
        context=search_results,
        conversation_history=request.conversation_id
    )

    return ChatResponse(
        message=response.text,
        sources=response.citations,
        suggestions=response.follow_up_questions,
        visualization_data=response.viz_data
    )

@router.get("/papers/search")
async def search_papers(
    query: str,
    domain: Optional[str] = None,
    author: Optional[str] = None,
    year_range: Optional[str] = None,
    limit: int = 10
) -> List[PaperResult]:
    """논문 검색"""

    search_params = SearchParams(
        query=query,
        filters={
            'domain': domain,
            'author': author,
            'year_range': parse_year_range(year_range)
        },
        limit=limit
    )

    results = await psychology_rag.search_papers(search_params)
    return [PaperResult.from_document(doc) for doc in results]

@router.get("/research/trends")
async def analyze_trends(
    keywords: List[str],
    time_range: str = "5years",
    domain: Optional[str] = None
) -> TrendAnalysis:
    """연구 트렌드 분석"""

    analysis = await trend_analyzer.analyze(
        keywords=keywords,
        time_range=parse_time_range(time_range),
        domain=domain
    )

    return TrendAnalysis(
        keywords=keywords,
        trends=analysis.trends,
        hot_topics=analysis.emerging_topics,
        collaboration_network=analysis.network_data
    )

@router.post("/research/methodology-review")
async def review_methodology(
    request: MethodologyReviewRequest
) -> MethodologyFeedback:
    """연구 방법론 검토"""

    feedback = await methodology_agent.review(
        research_question=request.research_question,
        proposed_method=request.method,
        target_population=request.population,
        ethical_considerations=request.ethics
    )

    return MethodologyFeedback(
        overall_score=feedback.score,
        strengths=feedback.strengths,
        weaknesses=feedback.weaknesses,
        suggestions=feedback.improvements,
        ethical_review=feedback.ethics_check
    )
```

### 2. WebSocket 실시간 통신

```python
from fastapi import WebSocket, WebSocketDisconnect
import json

@router.websocket("/ws/chat/{user_id}")
async def websocket_chat_endpoint(websocket: WebSocket, user_id: str):
    """실시간 채팅 WebSocket"""

    await websocket.accept()
    await connection_manager.add_connection(user_id, websocket)

    try:
        while True:
            # 클라이언트로부터 메시지 받기
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # 스트리밍 응답 시작
            await websocket.send_json({
                "type": "thinking",
                "message": "검색 중입니다..."
            })

            # RAG 검색 및 응답 생성
            async for chunk in stream_chat_response(message_data):
                await websocket.send_json({
                    "type": "chunk",
                    "data": chunk
                })

            # 응답 완료
            await websocket.send_json({
                "type": "complete",
                "suggestions": generate_follow_up_questions()
            })

    except WebSocketDisconnect:
        await connection_manager.remove_connection(user_id)
```

## 📊 데이터 모델 설계

### 1. 데이터베이스 스키마

```sql
-- 심리학과 교수진 정보
CREATE TABLE psychology_faculty (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150),
    department VARCHAR(100),
    research_areas TEXT[],
    office_location VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 논문 정보
CREATE TABLE psychology_papers (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT[],
    faculty_id UUID REFERENCES psychology_faculty(id),
    abstract TEXT,
    keywords TEXT[],
    publication_year INTEGER,
    journal VARCHAR(200),
    doi VARCHAR(100),
    file_path VARCHAR(500),
    domain_classification VARCHAR(100),
    citation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 사용자 대화 기록
CREATE TABLE chat_conversations (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    conversation_title VARCHAR(200),
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES chat_conversations(id),
    message_type VARCHAR(50) NOT NULL, -- 'user', 'assistant'
    content TEXT NOT NULL,
    metadata JSONB,
    sources JSONB, -- 인용된 논문들
    created_at TIMESTAMP DEFAULT NOW()
);

-- 연구 협업 네트워크
CREATE TABLE collaboration_networks (
    id UUID PRIMARY KEY,
    researcher_a UUID REFERENCES psychology_faculty(id),
    researcher_b UUID REFERENCES psychology_faculty(id),
    collaboration_type VARCHAR(100), -- 'coauthor', 'advisor', 'project'
    strength_score FLOAT,
    papers_together INTEGER DEFAULT 0
);
```

### 2. Pydantic 모델

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

class PsychologyPaper(BaseModel):
    id: UUID
    title: str
    authors: List[str]
    faculty_id: Optional[UUID]
    abstract: str
    keywords: List[str]
    publication_year: int
    journal: str
    doi: Optional[str]
    domain_classification: str
    citation_count: int = 0
    relevance_score: Optional[float] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[UUID]
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    message: str
    sources: List[PsychologyPaper] = []
    suggestions: List[str] = []
    visualization_data: Optional[Dict[str, Any]] = None
    confidence_score: float

class ResearchQuery(BaseModel):
    query: str
    domain: Optional[str] = None
    methodology: Optional[str] = None
    time_range: Optional[str] = None
    author_filter: Optional[str] = None

class TrendAnalysis(BaseModel):
    keywords: List[str]
    time_period: str
    trending_topics: List[Dict[str, Any]]
    collaboration_patterns: Dict[str, Any]
    emerging_areas: List[str]
    citation_trends: Dict[str, List[float]]
```

## 🎯 특화 기능 구현

### 1. 심리학 전문 용어 처리

```python
class PsychologyTermProcessor:
    """심리학 전문 용어 및 개념 처리"""

    PSYCHOLOGY_GLOSSARY = {
        '인지편향': ['cognitive bias', 'heuristic', '휴리스틱'],
        '조건화': ['conditioning', 'pavlovian', '파블로프'],
        '신경가소성': ['neuroplasticity', 'brain plasticity', '뇌가소성'],
        # ... 더 많은 용어들
    }

    def expand_query(self, query: str) -> str:
        """쿼리에서 전문 용어를 확장"""
        expanded_terms = []

        for term in self.extract_terms(query):
            if term in self.PSYCHOLOGY_GLOSSARY:
                expanded_terms.extend(self.PSYCHOLOGY_GLOSSARY[term])

        return self.reconstruct_query(query, expanded_terms)

    def detect_research_methodology(self, text: str) -> List[str]:
        """연구 방법론 감지"""
        methodologies = []

        method_patterns = {
            'experimental': ['실험', 'experiment', 'RCT', '무작위'],
            'survey': ['설문', 'survey', '조사', '질문지'],
            'interview': ['면접', 'interview', '인터뷰', '심층면담'],
            'observation': ['관찰', 'observation', '행동관찰'],
            'meta_analysis': ['메타분석', 'meta-analysis', '체계적 리뷰']
        }

        for method, keywords in method_patterns.items():
            if any(keyword in text for keyword in keywords):
                methodologies.append(method)

        return methodologies
```

### 2. 윤리 검토 시스템

```python
class ResearchEthicsReviewer:
    """연구 윤리 자동 검토 시스템"""

    def review_research_proposal(self, proposal: str) -> EthicsReview:
        """연구 제안서의 윤리적 이슈 검토"""

        ethical_concerns = []

        # 취약 집단 관련 키워드 검사
        vulnerable_groups = ['아동', '청소년', '정신질환', '장애', '노인']
        if any(group in proposal for group in vulnerable_groups):
            ethical_concerns.append({
                'level': 'high',
                'concern': '취약 집단 대상 연구',
                'recommendation': 'IRB 사전 승인 필수, 보호자 동의서 필요'
            })

        # 개인정보 관련 검사
        personal_info_keywords = ['개인정보', '신상', '프라이버시', 'SNS', '위치정보']
        if any(keyword in proposal for keyword in personal_info_keywords):
            ethical_concerns.append({
                'level': 'medium',
                'concern': '개인정보 수집 및 처리',
                'recommendation': '개인정보 보호법 준수, 익명화 처리 필요'
            })

        return EthicsReview(
            concerns=ethical_concerns,
            risk_level=self.calculate_risk_level(ethical_concerns),
            required_approvals=self.determine_approvals(ethical_concerns)
        )
```

이 구현 명세서는 서울대 심리학과를 위한 실용적이고 전문적인 연구 챗봇 시스템의 상세한 기술적 구현 방안을 제시합니다. 기존 AI-CoScientist 인프라를 최대한 활용하면서도 심리학 연구의 특수성을 반영한 혁신적인 솔루션입니다.