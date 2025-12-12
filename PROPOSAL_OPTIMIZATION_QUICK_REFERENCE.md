# 🚀 제안서 최적화 퀵 레퍼런스 가이드

> **One-Click 제안서 품질 90+ 점 달성 시스템**

## 📋 **즉시 실행 명령어**

### **🎯 원클릭 실행 (추천)**
```bash
# 완전 최적화 (모든 단계, 최고 품질)
poetry run python scripts/proposal_optimizer.py optimize \
    --input "proposal.md" --mode full

# 빠른 개선 (핵심 단계만, 시간 절약)
poetry run python scripts/proposal_optimizer.py optimize \
    --input "proposal.md" --mode quick

# 대화형 마법사 (초보자 추천)
poetry run python scripts/proposal_optimizer.py wizard
```

### **⚡ 5단계 개별 실행**

| 단계 | 명령어 | 설명 | 시간 |
|------|-------|------|------|
| 🔍 **1단계** | `poetry run python scripts/map_proposal_to_evidence.py --proposal "proposal.md" --output "evidence.json"` | 과학적 주장 분석 & 근거 진단 | 2-3분 |
| ⚡ **2단계** | `poetry run python scripts/validate_proposal_claims.py --input "proposal.md" --interactive` | 실시간 주장 검증 & 수정 | 10-30분 |
| 📚 **3단계** | `poetry run python scripts/enhanced_dd_query.py --mode systematic_review --topic "korean brain genomics"` | 체계적 문헌 검토 | 5-10분 |
| 🤖 **4단계** | `poetry run python scripts/multi_agent_proposal_pipeline.py --mode full_pipeline --input "proposal.md" --output "enhanced.md"` | 6개 AI 에이전트 협업 | 15-45분 |
| 📖 **5단계** | `poetry run python scripts/automated_citation_generator.py --input "enhanced.md" --mode auto_cite --output "final.md"` | 자동 Citation 생성 | 3-5분 |

## 🎯 **실행 모드별 가이드**

### **🏆 완전 최적화 모드 (full)**
```bash
poetry run python scripts/proposal_optimizer.py optimize \
    --input "data/발달장애/제안서.md" \
    --mode full \
    --interactive
```
- **단계**: 1→2→3→4→5 (모든 단계)
- **품질**: 최고 (90+ 점 목표)
- **시간**: 35-95분
- **용도**: 최종 제출용, 1등급 달성

### **⚡ 빠른 개선 모드 (quick)**
```bash
poetry run python scripts/proposal_optimizer.py optimize \
    --input "proposal.md" \
    --mode quick
```
- **단계**: 1→2→5 (핵심만)
- **품질**: 양호 (80+ 점)
- **시간**: 15-40분
- **용도**: 시간 부족, 긴급 개선

### **📚 연구 중심 모드 (research)**
```bash
poetry run python scripts/proposal_optimizer.py optimize \
    --input "proposal.md" \
    --mode research
```
- **단계**: 1→3→4→5 (문헌 중심)
- **품질**: 높음 (85+ 점)
- **시간**: 25-65분
- **용도**: 학술적 엄밀성 강화

### **🔍 검증 중심 모드 (validation)**
```bash
poetry run python scripts/proposal_optimizer.py optimize \
    --input "proposal.md" \
    --mode validation
```
- **단계**: 1→2→4 (검증 집중)
- **품질**: 높음 (85+ 점)
- **시간**: 20-50분
- **용도**: 주장 검증, 오류 수정

## 🛠️ **고급 활용법**

### **🎯 타겟팅 검색**
```bash
# 특정 주제 깊이 탐색
poetry run python scripts/enhanced_dd_query.py \
    --mode systematic_review \
    --topic "korean autism foundation model genomics" \
    --n_results 20 \
    --export "research.json"

# 다중 배치 검색
echo "korean brain imaging validation
genomics foundation model clinical
WES autism korean population" > queries.txt

poetry run python scripts/enhanced_dd_query.py \
    --batch "queries.txt" \
    --n_results 5 \
    --export "batch_results.json"
```

### **🤖 전문 에이전트 개별 활용**
```bash
# 통계 분석 전문가
poetry run python scripts/multi_agent_proposal_pipeline.py \
    --mode agent_specific \
    --agent statistical_analyst \
    --input "proposal.md"

# 가설 생성 전문가
poetry run python scripts/multi_agent_proposal_pipeline.py \
    --mode agent_specific \
    --agent hypothesis_generator \
    --input "proposal.md"

# 임상 검증 전문가
poetry run python scripts/multi_agent_proposal_pipeline.py \
    --mode agent_specific \
    --agent clinical_validation_agent \
    --input "proposal.md"
```

### **📖 Citation 세밀 제어**
```bash
# 엄격한 자동 Citation (높은 품질)
poetry run python scripts/automated_citation_generator.py \
    --input "proposal.md" \
    --mode auto_cite \
    --threshold 0.8 \
    --output "high_quality.md"

# 관대한 자동 Citation (높은 커버리지)
poetry run python scripts/automated_citation_generator.py \
    --input "proposal.md" \
    --mode auto_cite \
    --threshold 0.6 \
    --output "comprehensive.md"

# 대화형 정밀 Citation
poetry run python scripts/automated_citation_generator.py \
    --input "proposal.md" \
    --mode interactive
```

## 📊 **품질 모니터링**

### **🎯 점수 추적**
```bash
# 현재 품질 점수 확인
poetry run python scripts/map_proposal_to_evidence.py \
    --proposal "proposal.md" \
    --output "score.json"

# 점수 확인 (JSON에서 추출)
cat score.json | jq '.summary.scientific_rigor_score'
```

### **📈 개선 전후 비교**
```bash
# Before
poetry run python scripts/map_proposal_to_evidence.py \
    --proposal "original.md" \
    --output "before.json"

# After optimization...

# After
poetry run python scripts/map_proposal_to_evidence.py \
    --proposal "optimized.md" \
    --output "after.json"

# 비교
echo "Before: $(cat before.json | jq '.summary.scientific_rigor_score')"
echo "After:  $(cat after.json | jq '.summary.scientific_rigor_score')"
```

## 🚨 **문제 해결**

### **❌ ChromaDB 오류**
```bash
# DD-RAPTOR 데이터베이스 재생성
poetry run python scripts/load_json_to_chromadb_dd.py
```

### **⚠️ 모델 로딩 실패**
```bash
# 모델 캐시 초기화
rm -rf ~/.cache/huggingface/
poetry run python scripts/enhanced_dd_query.py --query "test" -n 1
```

### **🔧 스크립트 오류**
```bash
# 의존성 재설치
poetry install --no-cache

# 권한 확인
chmod +x scripts/*.py
```

## 💡 **실전 사용 시나리오**

### **🚨 긴급 상황 (2시간 내)**
```bash
# 1. 빠른 진단 (5분)
poetry run python scripts/map_proposal_to_evidence.py \
    --proposal "urgent.md" --output "diagnosis.json"

# 2. 핵심 수정 (90분)
poetry run python scripts/proposal_optimizer.py optimize \
    --input "urgent.md" --mode quick --interactive

# 3. 최종 확인 (5분)
cat optimization_output/*/execution_log.json | jq '.final_score'
```

### **🏆 완벽한 제안서 (1일)**
```bash
# Morning: 진단 및 계획
poetry run python scripts/proposal_optimizer.py wizard

# Afternoon: 전체 최적화
poetry run python scripts/proposal_optimizer.py optimize \
    --input "proposal.md" --mode full --interactive

# Evening: 수동 정제 및 최종 검토
# (AI 출력을 바탕으로 수동 편집)
```

### **🔄 반복 개선 루프**
```bash
#!/bin/bash
# 수렴할 때까지 반복 개선
THRESHOLD=90
CURRENT_SCORE=0

for round in {1..5}; do
    echo "🔄 개선 라운드 $round"

    # 최적화 실행
    poetry run python scripts/proposal_optimizer.py optimize \
        --input "proposal_v$round.md" --mode research

    # 점수 확인
    CURRENT_SCORE=$(cat optimization_output/*/execution_log.json | jq '.final_score')
    echo "📊 현재 점수: $CURRENT_SCORE"

    # 목표 달성 시 종료
    if (( $(echo "$CURRENT_SCORE >= $THRESHOLD" | bc -l) )); then
        echo "🎯 목표 달성! ($THRESHOLD+ 점)"
        break
    fi

    # 다음 라운드를 위한 파일 복사
    cp optimization_output/*/optimized_*.md proposal_v$((round+1)).md
done
```

## 🎓 **프롬프트 엔지니어링 팁**

### **🎯 Claude에게 줄 최적 프롬프트**

**1단계 후 진단 프롬프트**:
```
evidence_diagnosis.json을 분석하여:
1. 가장 시급한 unsupported claims Top 5
2. 섹션별 개선 우선순위
3. DD-RAPTOR 추가 검색이 필요한 키워드
4. 다음 단계 실행 전략
```

**최종 품질 검증 프롬프트**:
```
optimized_proposal.md를 삼성미래기술육성사업 관점에서 검토:
1. 혁신성 점수 (30점 만점)
2. 실현가능성 점수 (30점 만점)
3. 파급효과 점수 (30점 만점)
4. 과학적 엄밀성 점수 (10점 만점)
5. 1등급 달성 가능성과 최종 개선점
```

## 📁 **폴더 구조**

```
AI-CoScientist/
├── scripts/
│   ├── proposal_optimizer.py           # 🎯 메인 통합 도구
│   ├── map_proposal_to_evidence.py     # 🔍 1단계: 진단
│   ├── validate_proposal_claims.py     # ⚡ 2단계: 검증
│   ├── enhanced_dd_query.py            # 📚 3단계: 문헌검토
│   ├── multi_agent_proposal_pipeline.py # 🤖 4단계: AI협업
│   └── automated_citation_generator.py # 📖 5단계: Citation
├── optimization_output/                # 📊 모든 실행 결과
└── PROPOSAL_OPTIMIZATION_QUICK_REFERENCE.md # 📖 이 가이드
```

## 🏆 **성과 목표**

| 점수 구간 | 평가 | 달성 전략 |
|----------|------|----------|
| **90+ 점** | 🏆 삼성 1등급 | `full` 모드 + 수동 정제 |
| **80-89점** | 🥈 우수 | `research` 모드 |
| **70-79점** | 🥉 양호 | `quick` 모드 |
| **<70 점** | ⚠️ 개선필요 | `validation` 모드 반복 |

---

## 🚀 **즉시 시작하기**

### **Step 1**: 현재 상태 확인
```bash
poetry run python scripts/map_proposal_to_evidence.py \
    --proposal "data/발달장애/과학적_엄밀성_기반_제안서_수정계획_FINAL_2025.md" \
    --output "current_status.json"
```

### **Step 2**: 최적화 실행
```bash
poetry run python scripts/proposal_optimizer.py optimize \
    --input "data/발달장애/과학적_엄밀성_기반_제안서_수정계획_FINAL_2025.md" \
    --mode full \
    --interactive
```

### **Step 3**: 결과 확인
```bash
ls -la optimization_output/
cat optimization_output/*/execution_log.json | jq '.final_score'
```

---

**🎯 목표**: 현재 0.0점 → 90+ 점 달성 → 삼성미래기술육성사업 1등급! 🏆

이 가이드를 북마크해두고 언제든 참조하세요! 📌