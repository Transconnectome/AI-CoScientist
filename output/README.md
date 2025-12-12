# Output Directory

이 디렉토리는 AI-CoScientist가 생성한 모든 파일을 저장합니다.

## 📁 디렉토리 구조

```
output/
├── papers/           # 논문 개선 결과물
├── rebuttals/        # Rebuttal letter 개선 결과물
└── comparisons/      # 개선 전후 비교 분석 문서
```

## 📄 파일 타입

### papers/
- `paper_*_rag_iteration_N.txt` - N번째 반복 개선 결과
- `paper_*_rag_improved_final.txt` - 최종 개선 버전
- `paper_*_rag_improved_final.docx` - Word 형식 (생성 시)

### rebuttals/
- `response_improved_rag.txt` - 개선된 rebuttal letter
- `response_improved_rag.docx` - Word 형식 (생성 시)

### comparisons/
- `*_comparison_rag.md` - 개선 전후 상세 비교 분석
- RAG 학습 효과 및 패턴 분석 포함

## 🔄 자동 정리

스크립트 실행 시 자동으로 적절한 디렉토리에 파일이 저장됩니다:

```bash
# Paper 개선
python scripts/apply_improvements_with_rag.py
# → output/papers/에 저장

# Rebuttal 개선
python scripts/improve_rebuttal_with_rag.py
# → output/rebuttals/에 저장
```

## 📊 RAG 패턴 저장소

생성된 개선 패턴은 `chromadb_data/`에 별도로 저장되어 향후 개선에 활용됩니다.

## 🧹 정리

오래된 파일 정리:
```bash
# 30일 이상 된 파일 삭제
find output/ -name "*.txt" -mtime +30 -delete
```
