# 🧪 다이어그램 파이프라인 테스트 실행 가이드

## 📁 파일 구조

```
claudedocs/diagram_pipelines/
├── TEST_EXECUTION_GUIDE.md      ← 현재 파일 (테스트 가이드)
├── pipeline_1_claude_mermaid.md  ← Claude + Mermaid/Code 파이프라인
├── pipeline_2_image_ai.md        ← Image AI 직접 생성 파이프라인
├── pipeline_3_kimi_k2_code.md    ← Kimi K2 코드 생성 파이프라인
└── test_results/                 ← 테스트 결과 저장 폴더 (생성 필요)
```

---

## 🎯 3가지 파이프라인 요약

| 파이프라인 | 방식 | 장점 | 단점 | 비용 |
|-----------|------|------|------|------|
| **1. Claude + Mermaid** | 코드 생성 → 렌더링 | 편집 가능, 버전 관리 | 렌더링 단계 필요 | 무료 (Claude 구독 별도) |
| **2. Image AI** | 프롬프트 → 이미지 | 빠른 시각화, 유연함 | 편집 어려움, 텍스트 오류 | 무료~$20/월 |
| **3. Kimi K2 + Code** | 코딩 → 렌더링 | 정밀, 복잡한 로직 | 렌더링 단계 필요 | 무료 |

---

## 📋 테스트 절차

### Step 1: 테스트 환경 준비

```bash
# 결과 저장 폴더 생성
mkdir -p ~/Desktop/_중견/claudedocs/diagram_pipelines/test_results

# Python 환경 (코드 기반 파이프라인용)
pip install matplotlib numpy networkx mermaid-py

# Graphviz 설치 (Mac)
brew install graphviz

# Mermaid CLI 설치
npm install -g @mermaid-js/mermaid-cli
```

### Step 2: 테스트 케이스 선택

**권장 테스트 순서**:
1. **Transformer Architecture** - 가장 복잡, 종합 테스트
2. **CNN Architecture** - 중간 복잡도
3. **Attention Mechanism** - 상세한 수학적 표현 필요

### Step 3: 각 파이프라인 실행

---

## 🔵 Pipeline 1 테스트: Claude + Mermaid

### 실행 방법

1. **Claude에게 요청** (이 세션에서 바로 가능):
```
Transformer 아키텍처를 Mermaid 다이어그램으로 만들어줘.
인코더, 디코더, Cross-Attention 연결 포함.
한글 레이블 사용.
```

2. **Mermaid 코드 복사**

3. **렌더링**:
   - 웹: https://mermaid.live/
   - CLI: `mmdc -i transformer.mmd -o test_results/p1_transformer.png`

4. **결과 저장**: `test_results/p1_transformer.png`

### 예시 요청
```
Create a Mermaid flowchart for Transformer architecture with:
- Encoder block (blue): Input Embedding → Positional Encoding → Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm
- Decoder block (orange): Output Embedding → Masked Attention → Cross-Attention → FFN
- Cross-attention connection labeled "K, V"
- Korean labels
- Use subgraph for grouping
```

---

## 🟢 Pipeline 2 테스트: Image AI

### ChatGPT 4o 테스트

1. **접속**: https://chat.openai.com (Plus 필요)

2. **프롬프트 입력**:
```
Create a professional scientific diagram of the Transformer architecture for a research paper.

Structure to show:
1. LEFT SIDE - ENCODER (blue theme):
   - Input Embedding → Positional Encoding
   - Multi-Head Self-Attention → Add & Norm
   - Feed Forward → Add & Norm
   - Label: "Encoder × N"

2. RIGHT SIDE - DECODER (orange theme):
   - Output Embedding → Positional Encoding
   - Masked Self-Attention → Add & Norm
   - Cross-Attention (arrow from encoder with "K, V" label) → Add & Norm
   - Feed Forward → Add & Norm
   - Label: "Decoder × N"

3. FINAL: Linear → Softmax → Output Probabilities

Requirements:
- White background, academic publication quality
- All text sharp and readable
- Clear arrows showing data flow
```

3. **이미지 다운로드**: `test_results/p2_transformer_chatgpt.png`

### Gemini 테스트

1. **접속**: https://gemini.google.com 또는 https://aistudio.google.com

2. **동일한 프롬프트 사용**

3. **저장**: `test_results/p2_transformer_gemini.png`

### Ideogram 테스트 (무료)

1. **접속**: https://ideogram.ai

2. **프롬프트**:
```
Transformer architecture diagram, scientific illustration,
encoder block in blue, decoder block in orange,
Multi-Head Attention, Feed Forward Network, Add & Norm layers,
K V labels on cross-attention arrow,
white background, academic publication style, sharp text labels
```

3. **저장**: `test_results/p2_transformer_ideogram.png`

---

## 🔴 Pipeline 3 테스트: Kimi K2

### 실행 방법

1. **접속**: https://kimi.moonshot.cn

2. **프롬프트**:
```
Create a complete Python script using matplotlib to draw a professional Transformer architecture diagram.

Requirements:
- Encoder (left, blue) and Decoder (right, orange) side by side
- All components: Embeddings, Attention, FFN, Add&Norm
- Cross-attention connection with "K, V" label
- White background, 300 DPI, publication quality
- Save as PNG

Output complete, runnable Python code.
```

3. **코드 복사 후 실행**:
```bash
python transformer_kimi.py
```

4. **저장**: `test_results/p3_transformer_kimi.png`

---

## 📊 평가 매트릭스

### 개별 평가 (각 결과물당)

```markdown
## 평가: [Pipeline X] - [Model Name] - [Test Case]

### 정량 평가 (1-5점)

| 항목 | 점수 | 비고 |
|------|------|------|
| 구조 정확성 | /5 | 모델 구조 정확히 반영 |
| 텍스트 품질 | /5 | 레이블 선명도, 오류 없음 |
| 시각적 품질 | /5 | 해상도, 색상, 레이아웃 |
| 학술 적합성 | /5 | 논문/제안서 사용 가능 |
| 편집 용이성 | /5 | 수정 가능 여부 |

**총점: /25**

### 정성 평가

**강점**:
-

**약점**:
-

**특이사항**:
-
```

---

## 📈 종합 비교표

테스트 완료 후 아래 표 작성:

```markdown
## 종합 비교 결과

| 항목 | Pipeline 1 (Claude+Mermaid) | Pipeline 2 (Image AI) | Pipeline 3 (Kimi K2) |
|------|---------------------------|---------------------|---------------------|
| **구조 정확성** | /5 | /5 | /5 |
| **텍스트 품질** | /5 | /5 | /5 |
| **시각적 품질** | /5 | /5 | /5 |
| **학술 적합성** | /5 | /5 | /5 |
| **편집 용이성** | /5 | /5 | /5 |
| **생성 속도** | /5 | /5 | /5 |
| **비용 효율** | /5 | /5 | /5 |
| **총점** | /35 | /35 | /35 |

### 최종 순위
1. 🥇
2. 🥈
3. 🥉

### 용도별 추천
- **논문 출판**:
- **프레젠테이션**:
- **빠른 시각화**:
- **협업/버전관리**:
```

---

## ⚡ 빠른 테스트 스크립트

### 한 번에 모든 파이프라인 테스트 (Python)

```python
#!/usr/bin/env python3
"""
다이어그램 파이프라인 테스트 스크립트
실행: python test_pipelines.py
"""

import os
import subprocess
from datetime import datetime

# 테스트 결과 폴더
RESULTS_DIR = os.path.expanduser("~/Desktop/_중견/claudedocs/diagram_pipelines/test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def test_pipeline_1_mermaid():
    """Pipeline 1: Mermaid 렌더링 테스트"""
    print("\n🔵 Pipeline 1: Mermaid 테스트")

    mermaid_code = '''
flowchart TB
    subgraph Encoder["🔷 인코더"]
        IE[Input Embedding] --> PE1[Positional Encoding]
        PE1 --> MHA1[Multi-Head Self-Attention]
        MHA1 --> AN1[Add & Norm]
        AN1 --> FF1[Feed Forward]
        FF1 --> AN2[Add & Norm]
    end

    subgraph Decoder["🔶 디코더"]
        OE[Output Embedding] --> PE2[Positional Encoding]
        PE2 --> MMHA[Masked Self-Attention]
        MMHA --> AN3[Add & Norm]
        AN3 --> MHA2[Cross-Attention]
        AN2 -.->|"K, V"| MHA2
        MHA2 --> AN4[Add & Norm]
        AN4 --> FF2[Feed Forward]
        FF2 --> AN5[Add & Norm]
    end

    AN5 --> Linear[Linear]
    Linear --> SM[Softmax]

    style Encoder fill:#e3f2fd
    style Decoder fill:#fff3e0
'''

    # Mermaid 파일 저장
    mmd_file = os.path.join(RESULTS_DIR, "p1_transformer.mmd")
    png_file = os.path.join(RESULTS_DIR, "p1_transformer.png")

    with open(mmd_file, 'w') as f:
        f.write(mermaid_code)

    # mmdc로 렌더링
    try:
        subprocess.run([
            'mmdc', '-i', mmd_file, '-o', png_file,
            '-t', 'default', '-b', 'white'
        ], check=True)
        print(f"  ✅ 저장됨: {png_file}")
    except Exception as e:
        print(f"  ❌ 렌더링 실패: {e}")
        print("  💡 mmdc 설치: npm install -g @mermaid-js/mermaid-cli")

def test_pipeline_3_matplotlib():
    """Pipeline 3: matplotlib 코드 테스트"""
    print("\n🔴 Pipeline 3: matplotlib 테스트")

    # 간단한 테스트 코드
    code = '''
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# 간단한 Transformer 블록
def draw_box(x, y, w, h, text, color):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)

# Encoder
draw_box(1, 5, 2.5, 0.8, "Input Embedding", "#e3f2fd")
draw_box(1, 3.5, 2.5, 0.8, "Self-Attention", "#e8f5e9")
draw_box(1, 2, 2.5, 0.8, "Feed Forward", "#fce4ec")

# Decoder
draw_box(5, 5, 2.5, 0.8, "Output Embedding", "#fff3e0")
draw_box(5, 3.5, 2.5, 0.8, "Cross-Attention", "#bbdefb")
draw_box(5, 2, 2.5, 0.8, "Feed Forward", "#fce4ec")

# Final
draw_box(9, 3.5, 2, 0.8, "Linear", "#f3e5f5")
draw_box(9, 2, 2, 0.8, "Softmax", "#f3e5f5")

ax.set_title("Transformer Architecture (Simple)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("RESULTS_DIR/p3_transformer_simple.png", dpi=300, facecolor='white')
print("Saved!")
'''.replace("RESULTS_DIR", RESULTS_DIR)

    # 코드 실행
    try:
        exec(code)
        print(f"  ✅ 저장됨: {RESULTS_DIR}/p3_transformer_simple.png")
    except Exception as e:
        print(f"  ❌ 실행 실패: {e}")

def main():
    print("=" * 50)
    print("🧪 다이어그램 파이프라인 테스트")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    test_pipeline_1_mermaid()
    test_pipeline_3_matplotlib()

    print("\n" + "=" * 50)
    print("✅ 테스트 완료!")
    print(f"📁 결과 폴더: {RESULTS_DIR}")
    print("\n⚠️ Pipeline 2 (Image AI)는 웹에서 수동 테스트 필요:")
    print("   - ChatGPT: https://chat.openai.com")
    print("   - Gemini: https://gemini.google.com")
    print("   - Ideogram: https://ideogram.ai")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

### 실행
```bash
cd ~/Desktop/_중견/claudedocs/diagram_pipelines
python test_pipelines.py
```

---

## 🏁 테스트 완료 체크리스트

- [ ] Pipeline 1 (Claude + Mermaid) 테스트 완료
- [ ] Pipeline 2 (ChatGPT 4o) 테스트 완료
- [ ] Pipeline 2 (Gemini) 테스트 완료
- [ ] Pipeline 2 (Ideogram) 테스트 완료
- [ ] Pipeline 3 (Kimi K2 + matplotlib) 테스트 완료
- [ ] Pipeline 3 (Kimi K2 + Graphviz) 테스트 완료
- [ ] 모든 결과 test_results 폴더에 저장
- [ ] 평가 매트릭스 작성
- [ ] 종합 비교표 완성
- [ ] 최종 추천 결정

---

## 💡 추가 팁

1. **동일한 테스트 케이스**: 모든 파이프라인에 동일한 Transformer 구조로 테스트해야 공정한 비교 가능

2. **여러 번 생성**: Image AI는 매번 다른 결과 → 3번 생성 후 최선 선택

3. **고해상도 저장**: 모든 결과를 300 DPI 이상으로 저장

4. **텍스트 확인**: 특히 Image AI 결과에서 텍스트 오류 꼼꼼히 확인

5. **실제 사용 시나리오**: "논문에 넣을 건가, 발표용인가"에 따라 다른 파이프라인이 적합할 수 있음
