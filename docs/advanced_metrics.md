# 고급 메트릭 및 LLM 통합 가이드

이 문서는 AI-CoScientist의 고급 메트릭 시스템, Adversarial Reviewer, Defense Agent, 그리고 Golden Reference RAG 시스템의 작동 방식과 사용법을 설명합니다.

## 1. 개요

AI-CoScientist는 단순한 정량적 평가를 넘어, 실제 학술지 리뷰 프로세스를 시뮬레이션하고 최고 수준의 논문과 비교 분석하여 논문의 품질을 획기적으로 개선합니다.

### 주요 기능
- **Adversarial Reviewer (Reviewer #2)**: 비판적인 시각에서 논문의 약점을 찾아내고 가혹한 리뷰를 생성합니다.
- **Defense Agent**: 비판에 대한 방어 논리를 수립하고, 구체적인 개선 사항을 제안하여 논문을 강화합니다.
- **Golden Reference RAG**: Nature, Science 등 Top-tier 저널의 논문을 벤치마킹하여 격차 분석(Gap Analysis)을 수행합니다.
- **LLM 통합**: OpenAI GPT 모델을 활용하여 깊이 있는 텍스트 분석과 창의적인 개선 제안을 생성합니다.

---

## 2. Adversarial Reviewer (적대적 리뷰어)

`AdversarialReviewer` 클래스는 까다로운 저널 리뷰어("Reviewer #2")를 시뮬레이션합니다.

### 작동 방식
1. **LLM 분석**: 논문 내용을 LLM에 전달하여 방법론, 명확성, 독창성, 중요성 측면에서 비판적인 분석을 수행합니다.
2. **약점 식별**: 구체적인 약점을 찾아내고 심각도(Severity) 점수(0.0 - 1.0)를 부여합니다.
3. **피드백 생성**: 저널 리뷰 형식의 상세한 피드백 리포트를 작성합니다.

### 사용 예시
```python
from src.services.paper.adversarial_reviewer import AdversarialReviewer
from src.core.config import settings

# LLM 클라이언트와 함께 초기화
reviewer = AdversarialReviewer(llm_service=settings.openai_client)

review_result = await reviewer.review(paper_content)
print(review_result["feedback"])
print(review_result["weaknesses"])
```

---

## 3. Defense Agent (방어 에이전트)

`DefenseAgent`는 리뷰어의 비판을 분석하고 이를 바탕으로 논문을 개선하는 역할을 합니다.

### 작동 방식
1. **비판 분석**: 리뷰어의 지적 사항을 분석하여 타당성을 평가합니다.
2. **방어 전략 수립**: 각 비판에 대해 수용(Accept), 반박(Rebut), 또는 보완(Clarify) 전략을 수립합니다.
3. **개선 제안**: 논문 본문을 구체적으로 어떻게 수정해야 할지 제안하고, 예상되는 점수 향상폭을 계산합니다.

### 사용 예시
```python
from src.services.paper.defense_agent import DefenseAgent

agent = DefenseAgent(llm_service=settings.openai_client)

# 리뷰 결과에 대한 방어 및 개선 제안 생성
defense_plan = await agent.analyze_and_defend(review_result)
improvements = await agent.generate_improvements(paper_content, review_result)

for imp in improvements["improvements"]:
    print(f"Priority: {imp['priority']}, Impact: {imp['expected_impact']}")
    print(f"Action: {imp['description']}")
```

---

## 4. Golden Reference RAG (골든 레퍼런스 벤치마킹)

최고 수준의 저널(Nature, Science, Cell 등)에 게재된 논문들과 비교하여 현재 논문의 위치를 파악합니다.

### 작동 방식
1. **레퍼런스 검색**: 주제와 관련된 Top-tier 논문을 검색합니다 (RAG/Vector DB 활용).
2. **격차 분석 (Gap Analysis)**: 벤치마크 논문들의 특징(통계적 엄밀성, 서사 구조 등)과 비교하여 부족한 점을 분석합니다.
3. **티어 평가**: 현재 논문의 예상 저널 티어(Top, High, Mid)를 판정합니다.

### 사용 예시
```python
from src.services.paper.metrics import PaperMetrics

# 골든 레퍼런스 검색
references = await PaperMetrics.retrieve_golden_references(
    topic="Deep Learning for fMRI",
    journal_tier="top"
)

# 벤치마킹 점수 계산
benchmark = await PaperMetrics.score_against_golden_references(
    paper_content,
    target_journal_tier="top"
)

print(f"Benchmark Score: {benchmark['benchmark_score']}/10")
print(f"Tier Assessment: {benchmark['tier_assessment']}")
```

---

## 5. 설정 및 환경 변수

이 기능들을 사용하기 위해서는 `.env` 파일에 OpenAI API 키가 설정되어 있어야 합니다.

```dotenv
OPENAI_API_KEY=sk-proj-...
```

## 6. 데모 실행

전체 워크플로우를 시연하는 스크립트가 준비되어 있습니다.

```bash
python scripts/demo_advanced_metrics.py
```

이 스크립트는 다음 과정을 자동으로 수행합니다:
1. 샘플 논문 생성
2. Adversarial Review 수행
3. Defense Agent를 통한 개선 제안 생성
4. Golden Reference 벤치마킹
5. 개선 전후 결과 비교
