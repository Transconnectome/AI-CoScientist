#!/usr/bin/env python3
"""
Nano Banana (Gemini Image) 테스트 스크립트
==========================================
- Nano Banana: gemini-2.5-flash-image
- Nano Banana Pro: gemini-3-pro-image-preview
- Imagen 4.0: imagen-4.0-generate-001
"""

import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 설정
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)

load_dotenv(SCRIPT_DIR / ".env")

# Transformer 다이어그램 프롬프트
DIAGRAM_PROMPT = """Create a professional scientific diagram of the Transformer architecture for a research paper.

Structure:
- ENCODER (left, blue): Input Embedding → Positional Encoding → Multi-Head Self-Attention → Add & Norm → Feed Forward → Add & Norm
- DECODER (right, orange): Output Embedding → Positional Encoding → Masked Self-Attention → Add & Norm → Cross-Attention (K,V from encoder) → Add & Norm → Feed Forward → Add & Norm → Linear → Softmax

Style: Clean, professional, white background, clear arrows showing data flow, readable text labels.
"""

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "START": "🚀"}
    print(f"[{timestamp}] {icons.get(level, '•')} {msg}")


def test_nano_banana():
    """Nano Banana (gemini-2.5-flash-image) 테스트"""
    log("Nano Banana (gemini-2.5-flash-image) 테스트 시작", "START")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log("GOOGLE_API_KEY 없음", "ERROR")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        log("이미지 생성 중...")
        start_time = time.time()

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[DIAGRAM_PROMPT],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE', 'TEXT']
            )
        )

        elapsed = time.time() - start_time
        output_path = RESULTS_DIR / "nano_banana.png"

        # 이미지 추출
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                with open(output_path, 'wb') as f:
                    f.write(part.inline_data.data)
                log(f"저장 완료: {output_path} ({elapsed:.1f}초)", "SUCCESS")
                return {"model": "Nano Banana", "output": str(output_path), "time": elapsed, "success": True}

        # 이미지가 없으면 텍스트 응답 저장
        text_path = RESULTS_DIR / "nano_banana_response.txt"
        with open(text_path, 'w') as f:
            f.write(response.text if hasattr(response, 'text') else str(response))
        log("이미지 대신 텍스트 응답", "WARN")
        return {"model": "Nano Banana", "note": "텍스트 응답", "time": elapsed, "success": False}

    except Exception as e:
        log(f"오류: {e}", "ERROR")
        return {"model": "Nano Banana", "error": str(e), "success": False}


def test_nano_banana_pro():
    """Nano Banana Pro (gemini-3-pro-image-preview) 테스트"""
    log("Nano Banana Pro (gemini-3-pro-image-preview) 테스트 시작", "START")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log("GOOGLE_API_KEY 없음", "ERROR")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        log("이미지 생성 중...")
        start_time = time.time()

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[DIAGRAM_PROMPT],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE', 'TEXT']
            )
        )

        elapsed = time.time() - start_time
        output_path = RESULTS_DIR / "nano_banana_pro.png"

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                with open(output_path, 'wb') as f:
                    f.write(part.inline_data.data)
                log(f"저장 완료: {output_path} ({elapsed:.1f}초)", "SUCCESS")
                return {"model": "Nano Banana Pro", "output": str(output_path), "time": elapsed, "success": True}

        text_path = RESULTS_DIR / "nano_banana_pro_response.txt"
        with open(text_path, 'w') as f:
            f.write(response.text if hasattr(response, 'text') else str(response))
        log("이미지 대신 텍스트 응답", "WARN")
        return {"model": "Nano Banana Pro", "note": "텍스트 응답", "time": elapsed, "success": False}

    except Exception as e:
        log(f"오류: {e}", "ERROR")
        return {"model": "Nano Banana Pro", "error": str(e), "success": False}


def test_imagen4():
    """Imagen 4.0 테스트"""
    log("Imagen 4.0 테스트 시작", "START")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log("GOOGLE_API_KEY 없음", "ERROR")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        log("이미지 생성 중...")
        start_time = time.time()

        # Imagen 4.0은 다른 API 사용
        response = client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=DIAGRAM_PROMPT,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9"
            )
        )

        elapsed = time.time() - start_time
        output_path = RESULTS_DIR / "imagen4.png"

        if response.generated_images:
            img = response.generated_images[0]
            img.image.save(output_path)
            log(f"저장 완료: {output_path} ({elapsed:.1f}초)", "SUCCESS")
            return {"model": "Imagen 4.0", "output": str(output_path), "time": elapsed, "success": True}

        log("이미지 생성 실패", "WARN")
        return {"model": "Imagen 4.0", "note": "이미지 없음", "time": elapsed, "success": False}

    except Exception as e:
        log(f"오류: {e}", "ERROR")
        return {"model": "Imagen 4.0", "error": str(e), "success": False}


def main():
    print("\n" + "=" * 60)
    print("🍌 Nano Banana 이미지 생성 테스트")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    # 1. Nano Banana
    print("\n" + "-" * 40)
    results.append(test_nano_banana())

    # 2. Nano Banana Pro
    print("\n" + "-" * 40)
    results.append(test_nano_banana_pro())

    # 3. Imagen 4.0
    print("\n" + "-" * 40)
    results.append(test_imagen4())

    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 완료!")
    print("=" * 60)

    success = sum(1 for r in results if r and r.get('success'))
    total = sum(1 for r in results if r)

    print(f"\n✅ 성공: {success}/{total}")
    print(f"📁 결과: {RESULTS_DIR}")

    for r in results:
        if r:
            status = "✅" if r.get('success') else "❌"
            print(f"  {status} {r.get('model')}: {r.get('output', r.get('error', r.get('note', '-')))}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
