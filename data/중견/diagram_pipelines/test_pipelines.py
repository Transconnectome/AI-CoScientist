#!/usr/bin/env python3
"""
다이어그램 파이프라인 테스트 스크립트
======================================

이 스크립트는 3가지 다이어그램 생성 파이프라인을 테스트합니다:
- Pipeline 1: Claude + Mermaid (코드 기반)
- Pipeline 2: Image AI (수동 테스트 가이드)
- Pipeline 3: Kimi K2 + matplotlib (코드 기반)

실행 방법:
    python test_pipelines.py

필요 패키지:
    pip install matplotlib numpy

선택 패키지:
    npm install -g @mermaid-js/mermaid-cli  # Mermaid 렌더링용
    brew install graphviz  # Graphviz 렌더링용 (Mac)
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

# ========== 설정 ==========
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "test_results"


def setup_directories():
    """결과 폴더 생성"""
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"📁 결과 폴더: {RESULTS_DIR}")


# ========== Pipeline 1: Mermaid ==========
MERMAID_TRANSFORMER = '''%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e3f2fd', 'secondaryColor': '#fff3e0'}}}%%
flowchart TB
    subgraph Input["📥 입력"]
        I[/"Input Tokens"/] --> IE[Input Embedding]
        IE --> PE1[⊕ Positional Encoding]
    end

    subgraph Encoder["🔷 인코더 블록 ×N"]
        PE1 --> MHA1[Multi-Head<br/>Self-Attention]
        MHA1 --> AN1[Add & Norm]
        AN1 --> FF1[Feed Forward]
        FF1 --> AN2[Add & Norm]
    end

    subgraph Output["📤 출력"]
        O[/"Output Tokens"/] --> OE[Output Embedding]
        OE --> PE2[⊕ Positional Encoding]
    end

    subgraph Decoder["🔶 디코더 블록 ×N"]
        PE2 --> MMHA[Masked Multi-Head<br/>Self-Attention]
        MMHA --> AN3[Add & Norm]
        AN3 --> MHA2[Multi-Head<br/>Cross-Attention]
        AN2 -.->|"K, V"| MHA2
        MHA2 --> AN4[Add & Norm]
        AN4 --> FF2[Feed Forward]
        FF2 --> AN5[Add & Norm]
    end

    subgraph Final["🎯 최종 출력"]
        AN5 --> Linear[Linear Layer]
        Linear --> SM[Softmax]
        SM --> Prob[/"Output Probabilities"/]
    end

    style Input fill:#e3f2fd,stroke:#1976d2
    style Encoder fill:#e8f5e9,stroke:#388e3c
    style Output fill:#fff3e0,stroke:#f57c00
    style Decoder fill:#fce4ec,stroke:#c2185b
    style Final fill:#f3e5f5,stroke:#7b1fa2
'''


def test_pipeline_1_mermaid():
    """Pipeline 1: Mermaid 코드 기반 테스트"""
    print("\n" + "=" * 50)
    print("🔵 Pipeline 1: Claude + Mermaid 테스트")
    print("=" * 50)

    # Mermaid 파일 저장
    mmd_file = RESULTS_DIR / "p1_transformer.mmd"
    png_file = RESULTS_DIR / "p1_transformer.png"

    with open(mmd_file, 'w', encoding='utf-8') as f:
        f.write(MERMAID_TRANSFORMER)
    print(f"  📝 Mermaid 코드 저장: {mmd_file}")

    # mmdc로 렌더링 시도
    try:
        result = subprocess.run(
            ['mmdc', '-i', str(mmd_file), '-o', str(png_file),
             '-t', 'default', '-b', 'white', '-w', '1200'],
            capture_output=True, text=True, check=True
        )
        print(f"  ✅ PNG 저장됨: {png_file}")
        return True
    except FileNotFoundError:
        print("  ⚠️ mmdc 미설치 - 수동 렌더링 필요")
        print("  💡 설치: npm install -g @mermaid-js/mermaid-cli")
        print(f"  💡 또는 https://mermaid.live/ 에서 {mmd_file} 내용 복사하여 렌더링")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 렌더링 실패: {e.stderr}")
        return False


# ========== Pipeline 3: matplotlib ==========
def test_pipeline_3_matplotlib():
    """Pipeline 3: matplotlib 코드 기반 테스트"""
    print("\n" + "=" * 50)
    print("🔴 Pipeline 3: Kimi K2 스타일 matplotlib 테스트")
    print("=" * 50)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError:
        print("  ❌ matplotlib 미설치")
        print("  💡 설치: pip install matplotlib")
        return False

    # Figure 설정
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('white')

    # 색상 정의
    COLORS = {
        'encoder': '#e3f2fd',
        'encoder_border': '#1976d2',
        'decoder': '#fff3e0',
        'decoder_border': '#f57c00',
        'attention': '#e8f5e9',
        'ffn': '#fce4ec',
        'final': '#f3e5f5',
        'norm': '#ffffff'
    }

    def draw_box(x, y, w, h, text, facecolor, edgecolor='black', fontsize=9):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')

    def draw_arrow(start, end, color='black'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # ===== ENCODER (왼쪽) =====
    enc_x = 3.5
    ax.text(enc_x, 9.3, 'ENCODER', fontsize=12, fontweight='bold',
            ha='center', color=COLORS['encoder_border'])

    draw_box(enc_x, 8.5, 2.8, 0.7, 'Input\nEmbedding', COLORS['encoder'], COLORS['encoder_border'])
    draw_box(enc_x, 7.5, 2.8, 0.7, '⊕ Positional\nEncoding', COLORS['encoder'], COLORS['encoder_border'])
    draw_box(enc_x, 6.3, 2.8, 0.8, 'Multi-Head\nSelf-Attention', COLORS['attention'], '#388e3c')
    draw_box(enc_x, 5.3, 2.8, 0.5, 'Add & Norm', COLORS['norm'], '#666666')
    draw_box(enc_x, 4.4, 2.8, 0.7, 'Feed Forward', COLORS['ffn'], '#c2185b')
    draw_box(enc_x, 3.5, 2.8, 0.5, 'Add & Norm', COLORS['norm'], '#666666')

    ax.text(enc_x, 2.9, '× N', fontsize=11, ha='center', style='italic', color='#666666')

    # 인코더 화살표
    for y1, y2 in [(8.15, 7.85), (7.15, 6.7), (5.9, 5.55), (5.05, 4.75), (4.05, 3.75)]:
        draw_arrow((enc_x, y1), (enc_x, y2))

    # ===== DECODER (오른쪽) =====
    dec_x = 10.5
    ax.text(dec_x, 9.3, 'DECODER', fontsize=12, fontweight='bold',
            ha='center', color=COLORS['decoder_border'])

    draw_box(dec_x, 8.5, 2.8, 0.7, 'Output\nEmbedding', COLORS['decoder'], COLORS['decoder_border'])
    draw_box(dec_x, 7.5, 2.8, 0.7, '⊕ Positional\nEncoding', COLORS['decoder'], COLORS['decoder_border'])
    draw_box(dec_x, 6.3, 2.8, 0.8, 'Masked Multi-Head\nSelf-Attention', COLORS['attention'], '#388e3c')
    draw_box(dec_x, 5.3, 2.8, 0.5, 'Add & Norm', COLORS['norm'], '#666666')
    draw_box(dec_x, 4.3, 2.8, 0.8, 'Multi-Head\nCross-Attention', '#bbdefb', '#1565c0')
    draw_box(dec_x, 3.3, 2.8, 0.5, 'Add & Norm', COLORS['norm'], '#666666')
    draw_box(dec_x, 2.4, 2.8, 0.7, 'Feed Forward', COLORS['ffn'], '#c2185b')
    draw_box(dec_x, 1.5, 2.8, 0.5, 'Add & Norm', COLORS['norm'], '#666666')

    ax.text(dec_x, 0.9, '× N', fontsize=11, ha='center', style='italic', color='#666666')

    # 디코더 화살표
    for y1, y2 in [(8.15, 7.85), (7.15, 6.7), (5.9, 5.55), (5.05, 4.7), (3.9, 3.55), (3.05, 2.75), (2.05, 1.75)]:
        draw_arrow((dec_x, y1), (dec_x, y2))

    # Cross-Attention 연결
    ax.annotate('', xy=(dec_x - 1.4, 4.3), xytext=(enc_x + 1.4, 3.5),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(7, 4.2, 'K, V', fontsize=10, fontweight='bold', color='#1565c0')

    # ===== FINAL LAYERS =====
    final_x = 10.5
    draw_box(final_x, 0.5, 2, 0.5, 'Linear', COLORS['final'], '#7b1fa2')

    # 화살표
    draw_arrow((dec_x, 1.25), (final_x, 0.75))

    # Output 레이블
    ax.text(final_x + 1.5, 0.5, '→ Softmax → Output', fontsize=10,
            ha='left', va='center', style='italic')

    # 제목
    ax.text(7, 10, 'Transformer Architecture', fontsize=16, fontweight='bold', ha='center')

    # 저장
    output_file = RESULTS_DIR / "p3_transformer_matplotlib.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  ✅ PNG 저장됨: {output_file}")
    return True


# ========== Pipeline 2: Image AI 가이드 ==========
def show_pipeline_2_guide():
    """Pipeline 2: Image AI 수동 테스트 가이드"""
    print("\n" + "=" * 50)
    print("🟢 Pipeline 2: Image AI 수동 테스트 가이드")
    print("=" * 50)

    prompt = '''Create a professional scientific diagram of the Transformer architecture.

ENCODER (left, blue theme):
- Input Embedding → Positional Encoding
- Multi-Head Self-Attention → Add & Norm
- Feed Forward → Add & Norm
- Label: "Encoder × N"

DECODER (right, orange theme):
- Output Embedding → Positional Encoding
- Masked Self-Attention → Add & Norm
- Cross-Attention (with "K, V" arrow from encoder) → Add & Norm
- Feed Forward → Add & Norm
- Label: "Decoder × N"

FINAL: Linear → Softmax → Output

Style: White background, academic publication quality, sharp text labels.'''

    print("\n📋 아래 프롬프트를 복사하여 각 서비스에서 테스트하세요:\n")
    print("-" * 40)
    print(prompt)
    print("-" * 40)

    print("\n🌐 테스트 서비스:")
    print("  1. ChatGPT 4o: https://chat.openai.com (Plus 필요)")
    print("  2. Gemini: https://gemini.google.com 또는 https://aistudio.google.com")
    print("  3. Ideogram: https://ideogram.ai (무료)")

    print(f"\n📁 결과 저장 위치: {RESULTS_DIR}")
    print("  - p2_transformer_chatgpt.png")
    print("  - p2_transformer_gemini.png")
    print("  - p2_transformer_ideogram.png")

    # 프롬프트 파일로 저장
    prompt_file = RESULTS_DIR / "p2_image_ai_prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"\n💾 프롬프트 파일 저장됨: {prompt_file}")


# ========== 평가 템플릿 ==========
def create_evaluation_template():
    """평가 템플릿 생성"""
    print("\n" + "=" * 50)
    print("📊 평가 템플릿 생성")
    print("=" * 50)

    template = '''# 다이어그램 파이프라인 평가 결과

테스트 일시: {date}

---

## Pipeline 1: Claude + Mermaid

| 항목 | 점수 (1-5) | 비고 |
|------|----------|------|
| 구조 정확성 | | |
| 텍스트 품질 | | |
| 시각적 품질 | | |
| 학술 적합성 | | |
| 편집 용이성 | | |
| **총점** | **/25** | |

**강점**:
-

**약점**:
-

---

## Pipeline 2: Image AI

### ChatGPT 4o
| 항목 | 점수 (1-5) | 비고 |
|------|----------|------|
| 구조 정확성 | | |
| 텍스트 품질 | | |
| 시각적 품질 | | |
| 학술 적합성 | | |
| **총점** | **/20** | |

### Gemini
| 항목 | 점수 (1-5) | 비고 |
|------|----------|------|
| 구조 정확성 | | |
| 텍스트 품질 | | |
| 시각적 품질 | | |
| 학술 적합성 | | |
| **총점** | **/20** | |

### Ideogram
| 항목 | 점수 (1-5) | 비고 |
|------|----------|------|
| 구조 정확성 | | |
| 텍스트 품질 | | |
| 시각적 품질 | | |
| 학술 적합성 | | |
| **총점** | **/20** | |

---

## Pipeline 3: Kimi K2 + matplotlib

| 항목 | 점수 (1-5) | 비고 |
|------|----------|------|
| 구조 정확성 | | |
| 코드 품질 | | |
| 시각적 품질 | | |
| 학술 적합성 | | |
| 편집 용이성 | | |
| **총점** | **/25** | |

**강점**:
-

**약점**:
-

---

## 종합 비교

| 순위 | 파이프라인 | 총점 | 추천 용도 |
|------|-----------|------|----------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## 최종 결론

'''.format(date=datetime.now().strftime('%Y-%m-%d %H:%M'))

    template_file = RESULTS_DIR / "evaluation_template.md"
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(template)

    print(f"  ✅ 평가 템플릿 저장됨: {template_file}")


# ========== 메인 ==========
def main():
    print("\n" + "=" * 60)
    print("🧪 다이어그램 파이프라인 테스트 스크립트")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    setup_directories()

    # Pipeline 1: Mermaid
    test_pipeline_1_mermaid()

    # Pipeline 2: Image AI (가이드)
    show_pipeline_2_guide()

    # Pipeline 3: matplotlib
    test_pipeline_3_matplotlib()

    # 평가 템플릿
    create_evaluation_template()

    # 완료 메시지
    print("\n" + "=" * 60)
    print("✅ 테스트 준비 완료!")
    print("=" * 60)
    print(f"\n📁 결과 폴더: {RESULTS_DIR}")
    print("\n📋 다음 단계:")
    print("  1. test_results/ 폴더의 p1, p3 이미지 확인")
    print("  2. p2_image_ai_prompt.txt의 프롬프트로 웹 서비스 테스트")
    print("  3. 모든 결과를 test_results/에 저장")
    print("  4. evaluation_template.md로 평가 진행")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
