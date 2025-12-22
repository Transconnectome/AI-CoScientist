#!/bin/bash
# Figure 렌더링 스크립트
# 사전 요구사항: npm install -g @mermaid-js/mermaid-cli

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 중견연구 제안서 Figure 렌더링 ==="
echo ""

# Mermaid CLI 확인
if ! command -v mmdc &> /dev/null; then
    echo "[ERROR] mermaid-cli가 설치되어 있지 않습니다."
    echo "설치 방법: npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

# 렌더링 설정
CONFIG='{
  "theme": "default",
  "themeVariables": {
    "fontSize": "14px"
  }
}'

# Figure 1: 문제-갭-가설-기여
echo "[1/4] Fig 1: 문제-갭-가설-기여 렌더링..."
mmdc -i "$SCRIPT_DIR/fig1_problem_gap.mmd" \
     -o "$SCRIPT_DIR/fig1_problem_gap.png" \
     -w 1200 -H 900 \
     --backgroundColor white
echo "  → fig1_problem_gap.png 생성 완료"

# Figure 2: 모델 아키텍처
echo "[2/4] Fig 2: 모델 아키텍처 렌더링..."
mmdc -i "$SCRIPT_DIR/fig2_model_architecture.mmd" \
     -o "$SCRIPT_DIR/fig2_model_architecture.png" \
     -w 1400 -H 1000 \
     --backgroundColor white
echo "  → fig2_model_architecture.png 생성 완료"

# Figure 3: 데이터 파이프라인
echo "[3/4] Fig 3: 데이터 파이프라인 렌더링..."
mmdc -i "$SCRIPT_DIR/fig3_data_pipeline.mmd" \
     -o "$SCRIPT_DIR/fig3_data_pipeline.png" \
     -w 1800 -H 600 \
     --backgroundColor white
echo "  → fig3_data_pipeline.png 생성 완료"

# Figure 4: Gantt 로드맵
echo "[4/4] Fig 4: 5년 로드맵 렌더링..."
mmdc -i "$SCRIPT_DIR/fig4_gantt_roadmap.mmd" \
     -o "$SCRIPT_DIR/fig4_gantt_roadmap.png" \
     -w 1600 -H 800 \
     --backgroundColor white
echo "  → fig4_gantt_roadmap.png 생성 완료"

echo ""
echo "=== 렌더링 완료 ==="
echo "생성된 파일:"
ls -la "$SCRIPT_DIR"/*.png 2>/dev/null || echo "  (PNG 파일 없음 - 렌더링 오류 확인)"

echo ""
echo "다음 단계:"
echo "  1. 생성된 PNG 파일을 Word/HWP 문서에 삽입"
echo "  2. 필요시 크기/해상도 조정"
echo "  3. 캡션 추가 (Fig 1, Fig 2 등)"
