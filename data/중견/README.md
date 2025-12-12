# 중견연구자를 위한 AI-CoScientist 가이드

## 📁 이 폴더에 포함된 내용

- **가이드라인.md** - 한국연구재단 중견연구자 지원사업 제안서 작성 종합 가이드
- **AI_COSCIENTIST_중견연구자_온보딩_가이드.md** - AI-CoScientist & UPE 시스템 완전 온보딩 가이드
- **샘플 제안서들** - 다양한 분야의 성공 제안서 예시들

## 🚀 빠른 시작

### 1. 처음 사용하시는 분
```bash
# 대화형 마법사로 시작하세요
poetry run python scripts/proposal_wizard.py
```

### 2. 바로 최적화하고 싶은 분
```bash
# 완전 자동 최적화 (95+ 점수 목표)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "제안서_초안.md" --mode full --enable-cross-domain
```

### 3. 상세한 사용법이 필요한 분
**AI_COSCIENTIST_중견연구자_온보딩_가이드.md** 파일을 참조하세요.

## 📋 주요 기능

- **🎯 95+ 점수 목표** - Samsung Future Technology Grant 1등급 수준
- **🔍 7-Strategy RAG** - 다양한 검색 전략
- **🤖 6-Agent 협업** - 전문가 AI 에이전트
- **📊 실시간 평가** - 제안서 품질 분석
- **🌐 Cross-Domain** - 다영역 융합 지원

## 📞 도움이 필요하시면

1. **AI_COSCIENTIST_중견연구자_온보딩_가이드.md** - 상세 사용법
2. **가이드라인.md** - 한국연구재단 제안서 작성 가이드
3. GitHub Issues 또는 내부 지원 채널

---

*최신 업데이트: 2025년 12월*