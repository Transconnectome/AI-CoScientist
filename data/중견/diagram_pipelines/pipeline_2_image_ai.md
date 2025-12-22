# Pipeline 2: Image AI 직접 생성 파이프라인

## 개요
- **방식**: AI 이미지 생성 모델에 프롬프트 → 직접 이미지 출력
- **장점**: 빠른 시각화, 코딩 불필요, 유연한 스타일
- **단점**: 편집 어려움, 텍스트 렌더링 이슈, 비용 발생 가능

---

## 🎯 추천 모델 (2025년 12월 기준)

| 모델 | 텍스트 품질 | 속도 | 비용 | 접근성 |
|------|-----------|------|------|--------|
| **Nano Banana Pro** (Gemini 3 Pro Image) | ⭐⭐⭐⭐⭐ | <1초 | API 비용 | Google AI Studio |
| **ChatGPT 4o** | ⭐⭐⭐⭐⭐ | 1-1.5분 | $20/월 | chat.openai.com |
| **DALL-E 3** | ⭐⭐⭐⭐ | 10-20초 | API 비용 | OpenAI API |
| **Ideogram 2.0** | ⭐⭐⭐⭐⭐ | 빠름 | 무료 티어 | ideogram.ai |
| **Gemini 2.5 Flash** | ⭐⭐⭐⭐ | <1초 | 무료 티어 | aistudio.google.com |

---

## 🔧 사용 방법

### Option A: ChatGPT 4o (가장 추천)

**접속**: https://chat.openai.com (ChatGPT Plus 필요)

**프롬프트 템플릿**:
```
Create a professional scientific diagram showing [구조 설명].

Requirements:
- Clean, academic style suitable for research papers
- White background
- Clear labels in English (or Korean if specified)
- Logical flow with arrows showing data/information direction
- Color coding for different components
- High resolution, print-quality output
- No decorative elements, pure technical illustration

Specific structure:
[상세 구조 설명]
```

### Option B: Google Gemini (무료)

**접속**: https://aistudio.google.com 또는 https://gemini.google.com

**프롬프트 템플릿**:
```
Generate a technical diagram illustrating [구조 설명].

Style requirements:
- Scientific publication quality
- Minimalist design with white background
- Sharp, readable text labels
- Consistent visual hierarchy
- Professional color scheme (blues, greens)
- Box-and-arrow style for architecture diagrams

Include these components:
[컴포넌트 목록]
```

### Option C: Ideogram 2.0 (무료, 텍스트 최강)

**접속**: https://ideogram.ai

**프롬프트 템플릿**:
```
Technical architecture diagram, [구조 설명],
academic illustration style, white background,
labeled boxes connected by arrows,
professional scientific figure, high detail,
sharp text rendering, publication quality
```

---

## 📋 테스트 케이스별 프롬프트

### 테스트 1: Transformer Architecture

**ChatGPT 4o 프롬프트**:
```
Create a professional scientific diagram of the Transformer architecture for a research paper.

Structure to show:
1. LEFT SIDE - ENCODER:
   - Input Embedding → Positional Encoding
   - Multi-Head Self-Attention block
   - Add & Normalize layer
   - Feed Forward Network
   - Add & Normalize layer
   - Label: "Encoder Block × N"

2. RIGHT SIDE - DECODER:
   - Output Embedding → Positional Encoding
   - Masked Multi-Head Self-Attention
   - Add & Normalize
   - Multi-Head Cross-Attention (with arrow from Encoder)
   - Add & Normalize
   - Feed Forward Network
   - Add & Normalize
   - Label: "Decoder Block × N"

3. TOP - Final layers:
   - Linear layer
   - Softmax
   - Output Probabilities

Requirements:
- White background
- Blue tones for encoder, orange/red tones for decoder
- Clear arrows showing data flow
- "K, V" label on arrow from encoder to decoder cross-attention
- Professional, clean style suitable for academic publication
- All text must be sharp and readable
```

**Gemini 프롬프트**:
```
Generate a Transformer neural network architecture diagram.

Layout: Side-by-side encoder (left, blue) and decoder (right, orange).

Encoder stack: Input Embedding → Positional Encoding → Multi-Head Attention → Add&Norm → FFN → Add&Norm

Decoder stack: Output Embedding → Positional Encoding → Masked Attention → Add&Norm → Cross-Attention (connected to encoder) → Add&Norm → FFN → Add&Norm → Linear → Softmax

Style: Clean academic illustration, white background, labeled components, arrows showing flow, publication quality.
```

**Ideogram 프롬프트**:
```
Transformer architecture diagram, scientific illustration,
encoder block in blue (Input Embedding, Positional Encoding, Multi-Head Self-Attention, Add & Norm, Feed Forward, Add & Norm),
decoder block in orange (Output Embedding, Masked Attention, Cross-Attention, Feed Forward),
arrows showing data flow, K V labels,
white background, academic publication style, sharp text labels, high detail technical diagram
```

---

### 테스트 2: CNN Architecture

**ChatGPT 4o 프롬프트**:
```
Create a Convolutional Neural Network (CNN) architecture diagram for image classification.

Structure (left to right flow):
1. INPUT: Image icon labeled "224×224×3"
2. CONV BLOCK 1: Conv2D(64) → BatchNorm → ReLU → MaxPool, labeled "112×112×64"
3. CONV BLOCK 2: Conv2D(128) → BatchNorm → ReLU → MaxPool, labeled "56×56×128"
4. CONV BLOCK 3: Conv2D(256) → BatchNorm → ReLU → MaxPool, labeled "28×28×256"
5. FLATTEN: Show tensor being flattened
6. FC LAYERS: Dense(512) → Dropout(0.5) → Dense(10)
7. OUTPUT: Softmax → Class probabilities

Style requirements:
- Horizontal flow from left to right
- Show feature map dimensions at each stage
- Use green gradient for conv layers
- Use yellow for FC layers
- White background, clean academic style
- All labels clearly readable
```

**Gemini 프롬프트**:
```
CNN architecture diagram for image classification.

Flow: Input image (224x224x3) → Conv blocks (Conv2D, BatchNorm, ReLU, MaxPool) × 3 with increasing filters (64, 128, 256) → Flatten → Dense layers (512, dropout, 10) → Softmax output.

Show feature map dimensions at each stage. Horizontal layout. Green conv layers, yellow FC layers. White background, academic publication quality, clear labels.
```

---

### 테스트 3: GAN Architecture

**ChatGPT 4o 프롬프트**:
```
Create a Generative Adversarial Network (GAN) architecture diagram.

Structure:
1. GENERATOR (left side, green theme):
   - Input: Random noise vector z (latent space)
   - Multiple upsampling/deconv layers
   - Output: Generated fake image

2. DISCRIMINATOR (right side, red theme):
   - Input: Both real images and fake images
   - Multiple conv layers
   - Output: Real/Fake probability

3. TRAINING FLOW:
   - Show adversarial training loop
   - Generator tries to fool discriminator
   - Discriminator tries to distinguish real vs fake
   - Arrows showing the competition

Requirements:
- Two-column layout
- Green for generator path
- Red for discriminator path
- Show both training and inference flow
- Include loss function labels
- White background, publication quality
```

---

### 테스트 4: BERT Architecture

**ChatGPT 4o 프롬프트**:
```
Create a BERT (Bidirectional Encoder Representations from Transformers) architecture diagram.

Structure:
1. INPUT PROCESSING:
   - [CLS] token + Word tokens + [SEP] token
   - Token Embeddings + Segment Embeddings + Position Embeddings
   - Sum of all embeddings

2. TRANSFORMER ENCODER STACK:
   - 12 identical encoder layers (BERT-base)
   - Each layer: Multi-Head Attention → Add&Norm → FFN → Add&Norm
   - Bidirectional attention (show arrows in both directions)

3. OUTPUT:
   - [CLS] output for classification tasks
   - Token outputs for sequence labeling

4. PRE-TRAINING TASKS (optional annotation):
   - MLM (Masked Language Model)
   - NSP (Next Sentence Prediction)

Style: Vertical flow, blue theme, academic publication quality, white background.
```

---

### 테스트 5: Attention Mechanism

**ChatGPT 4o 프롬프트**:
```
Create a detailed Scaled Dot-Product Attention mechanism diagram.

Structure:
1. INPUTS (three parallel branches):
   - Query (Q) matrix
   - Key (K) matrix
   - Value (V) matrix

2. COMPUTATION STEPS:
   - Q × K^T (matrix multiplication)
   - Scale by √d_k
   - Softmax (show attention weights visualization)
   - Multiply by V

3. OUTPUT:
   - Attention output matrix

4. FORMULA annotation:
   Attention(Q, K, V) = softmax(QK^T / √d_k)V

Style requirements:
- Show matrix shapes at each step
- Use color gradient to visualize attention weights
- Include mathematical notation
- Clean, academic style
- White background
```

---

## 🐍 Python API 사용 (자동화용)

### OpenAI DALL-E 3 API
```python
import openai
from pathlib import Path

client = openai.OpenAI(api_key="your-api-key")

def generate_diagram(prompt: str, output_path: str):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",  # 가로 레이아웃
        quality="hd",
        n=1
    )

    image_url = response.data[0].url

    # 이미지 다운로드
    import requests
    img_data = requests.get(image_url).content
    with open(output_path, 'wb') as f:
        f.write(img_data)

    return image_url

# 사용 예시
prompt = """
Create a professional scientific diagram of the Transformer architecture...
[전체 프롬프트]
"""

generate_diagram(prompt, "transformer_dalle3.png")
```

### Google Gemini API
```python
import google.generativeai as genai
from PIL import Image
import io

genai.configure(api_key="your-api-key")

def generate_with_gemini(prompt: str, output_path: str):
    model = genai.GenerativeModel('gemini-2.0-flash-exp')  # 이미지 생성 지원 모델

    response = model.generate_content(prompt)

    # 이미지 추출 및 저장
    for part in response.parts:
        if hasattr(part, 'inline_data'):
            image_data = part.inline_data.data
            image = Image.open(io.BytesIO(image_data))
            image.save(output_path)
            return output_path

    return None

# 사용 예시
prompt = """
Generate a Transformer neural network architecture diagram...
[전체 프롬프트]
"""

generate_with_gemini(prompt, "transformer_gemini.png")
```

---

## ✅ 평가 기준

| 항목 | 점수 (1-5) | 비고 |
|------|----------|------|
| 구조 정확성 | | 모델 구조 정확히 반영 |
| 텍스트 렌더링 | | 레이블 선명도, 오류 없음 |
| 시각적 품질 | | 해상도, 색상, 레이아웃 |
| 학술적 적합성 | | 논문/제안서 사용 가능 |
| 생성 속도 | | 응답 시간 |
| 비용 효율성 | | 가격 대비 품질 |

**총점**: ___ / 30

---

## ⚠️ 주의사항

1. **텍스트 오류**: AI 이미지 생성은 종종 텍스트를 잘못 렌더링함
   - 해결: 생성 후 수동 검토 필수
   - 또는 텍스트 없이 생성 후 별도 추가

2. **일관성**: 같은 프롬프트도 매번 다른 결과
   - 해결: 여러 번 생성 후 최선 선택

3. **저작권**: 상업적 사용 시 라이선스 확인
   - OpenAI, Google: 생성 이미지 상업 사용 가능
   - 단, 학술 출판물 정책 확인 필요
