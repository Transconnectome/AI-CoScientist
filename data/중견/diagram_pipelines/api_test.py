#!/usr/bin/env python3
"""
다이어그램 파이프라인 API 자동 테스트 스크립트
==============================================

테스트 대상:
1. OpenAI DALL-E 3 - 이미지 직접 생성
2. Google Gemini - 이미지 생성 (Imagen 3)
3. Kimi K2 (Moonshot) - 코드 생성 → 렌더링
4. DeepSeek - 코드 생성 → 렌더링

실행:
    cd diagram_pipelines
    source .venv/bin/activate
    python api_test.py
"""

import os
import sys
import time
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ========== 설정 ==========
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)

# .env 로드
load_dotenv(SCRIPT_DIR / ".env")

# 프롬프트 정의
IMAGE_PROMPT = """Create a professional scientific diagram of the Transformer architecture for a research paper.

Structure:
1. ENCODER (left side, blue theme):
   - Input Embedding → Positional Encoding
   - Multi-Head Self-Attention block
   - Add & Normalize layer
   - Feed Forward Network
   - Add & Normalize layer
   - Label: "Encoder Block × N"

2. DECODER (right side, orange theme):
   - Output Embedding → Positional Encoding
   - Masked Multi-Head Self-Attention
   - Add & Normalize
   - Multi-Head Cross-Attention (with arrow from Encoder labeled "K, V")
   - Add & Normalize
   - Feed Forward Network
   - Add & Normalize
   - Label: "Decoder Block × N"

3. FINAL OUTPUT:
   - Linear layer
   - Softmax
   - Output Probabilities

Requirements:
- White background
- Blue tones for encoder, orange/red tones for decoder
- Clear arrows showing data flow
- Professional, clean style suitable for academic publication
- All text must be sharp and readable
- High resolution"""

CODE_PROMPT = """Create a complete Python script using matplotlib to draw a professional Transformer architecture diagram.

Requirements:
1. Show Encoder (left, blue) and Decoder (right, orange) side by side
2. Components:
   - Encoder: Input Embedding, Positional Encoding, Multi-Head Self-Attention, Add&Norm, Feed Forward, Add&Norm
   - Decoder: Output Embedding, Positional Encoding, Masked Self-Attention, Add&Norm, Cross-Attention, Add&Norm, Feed Forward, Add&Norm
3. Cross-attention connection from encoder to decoder with "K, V" label
4. Final layers: Linear, Softmax
5. Professional colors, white background
6. Save as PNG with 300 DPI
7. Use matplotlib.patches for rounded rectangles
8. Include all arrows showing data flow

Output ONLY the Python code, no explanations. The code should be complete and runnable."""


# ========== 유틸리티 함수 ==========
def log(msg, level="INFO"):
    """로깅"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "START": "🚀"}
    icon = icons.get(level, "•")
    print(f"[{timestamp}] {icon} {msg}")


def save_image_from_url(url: str, output_path: Path) -> bool:
    """URL에서 이미지 다운로드"""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        log(f"이미지 다운로드 실패: {e}", "ERROR")
        return False


def save_image_from_base64(b64_data: str, output_path: Path) -> bool:
    """Base64에서 이미지 저장"""
    try:
        image_data = base64.b64decode(b64_data)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        return True
    except Exception as e:
        log(f"Base64 이미지 저장 실패: {e}", "ERROR")
        return False


# ========== 1. OpenAI DALL-E 3 ==========
def test_openai_dalle():
    """OpenAI DALL-E 3 이미지 생성 테스트"""
    log("OpenAI DALL-E 3 테스트 시작", "START")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("OPENAI_API_KEY 없음", "ERROR")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        log("DALL-E 3 이미지 생성 중... (약 30-60초 소요)")
        start_time = time.time()

        response = client.images.generate(
            model="dall-e-3",
            prompt=IMAGE_PROMPT,
            size="1792x1024",  # 가로 레이아웃
            quality="hd",
            n=1
        )

        elapsed = time.time() - start_time
        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        # 이미지 저장
        output_path = RESULTS_DIR / "api_openai_dalle3.png"
        if save_image_from_url(image_url, output_path):
            log(f"저장 완료: {output_path} ({elapsed:.1f}초)", "SUCCESS")

            # 수정된 프롬프트 저장
            with open(RESULTS_DIR / "api_openai_dalle3_prompt.txt", 'w') as f:
                f.write(f"Original Prompt:\n{IMAGE_PROMPT}\n\n")
                f.write(f"Revised Prompt:\n{revised_prompt}")

            return {
                "model": "DALL-E 3",
                "output": str(output_path),
                "time": elapsed,
                "success": True
            }

    except Exception as e:
        log(f"OpenAI 오류: {e}", "ERROR")
        return {"model": "DALL-E 3", "error": str(e), "success": False}


# ========== 2. Google Gemini ==========
def test_google_gemini():
    """Google Gemini 이미지 생성 테스트"""
    log("Google Gemini 테스트 시작", "START")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log("GOOGLE_API_KEY 없음", "ERROR")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Gemini 2.0 Flash (이미지 생성 지원)
        log("Gemini 이미지 생성 중...")
        start_time = time.time()

        # 이미지 생성 모델 사용
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        response = model.generate_content(
            f"Generate an image: {IMAGE_PROMPT}",
            generation_config=genai.GenerationConfig(
                temperature=0.4,
            )
        )

        elapsed = time.time() - start_time

        # 응답에서 이미지 추출
        output_path = RESULTS_DIR / "api_google_gemini.png"
        image_saved = False

        if response.parts:
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    image_saved = True
                    break

        if image_saved:
            log(f"저장 완료: {output_path} ({elapsed:.1f}초)", "SUCCESS")
            return {
                "model": "Gemini 2.0 Flash",
                "output": str(output_path),
                "time": elapsed,
                "success": True
            }
        else:
            # 이미지 생성 실패 시 텍스트 응답 저장
            text_response = response.text if hasattr(response, 'text') else str(response)
            with open(RESULTS_DIR / "api_google_gemini_response.txt", 'w') as f:
                f.write(text_response)
            log("Gemini가 이미지 대신 텍스트 응답 반환", "WARN")
            return {
                "model": "Gemini 2.0 Flash",
                "note": "텍스트 응답 반환됨",
                "time": elapsed,
                "success": False
            }

    except Exception as e:
        log(f"Gemini 오류: {e}", "ERROR")
        return {"model": "Gemini 2.0 Flash", "error": str(e), "success": False}


# ========== 3. Kimi K2 (Moonshot) ==========
def test_kimi_k2():
    """Kimi K2 코드 생성 테스트"""
    log("Kimi K2 (Moonshot) 테스트 시작", "START")

    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        log("MOONSHOT_API_KEY 없음", "ERROR")
        return None

    try:
        log("Kimi K2 코드 생성 중...")
        start_time = time.time()

        # Moonshot API (OpenAI 호환)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "moonshot-v1-128k",
            "messages": [
                {"role": "system", "content": "You are a Python expert. Output only code, no explanations."},
                {"role": "user", "content": CODE_PROMPT}
            ],
            "temperature": 0.3
        }

        # 먼저 글로벌 API 시도, 실패시 중국 API 시도
        endpoints = [
            "https://api.moonshot.ai/v1/chat/completions",  # 글로벌
            "https://api.moonshot.cn/v1/chat/completions",  # 중국
        ]

        response = None
        for endpoint in endpoints:
            try:
                log(f"시도 중: {endpoint.split('//')[1].split('/')[0]}")
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=120
                )
                if response.status_code == 200:
                    break
                elif response.status_code == 401:
                    log(f"{endpoint} - 인증 실패", "WARN")
                    continue
            except Exception as ep_error:
                log(f"{endpoint} 연결 실패: {ep_error}", "WARN")
                continue

        if response is None:
            raise Exception("모든 엔드포인트 연결 실패")

        response.raise_for_status()

        result = response.json()
        code = result['choices'][0]['message']['content']

        # 코드에서 ```python ``` 제거
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        elapsed_gen = time.time() - start_time
        log(f"코드 생성 완료 ({elapsed_gen:.1f}초)")

        # 코드 저장
        code_path = RESULTS_DIR / "api_kimi_k2_code.py"
        with open(code_path, 'w') as f:
            f.write(code)

        # 코드 실행하여 이미지 생성
        log("생성된 코드 실행 중...")

        # 코드에서 저장 경로 수정
        modified_code = code.replace(
            "plt.savefig(",
            f"plt.savefig('{RESULTS_DIR}/api_kimi_k2_diagram.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.savefig(#"
        )

        # 간단하게 실행
        import matplotlib
        matplotlib.use('Agg')

        # 실행 환경 설정
        exec_globals = {"__name__": "__main__"}
        try:
            exec(code, exec_globals)

            # plt.savefig가 호출되었는지 확인하고, 아니면 수동 저장
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                plt.savefig(RESULTS_DIR / "api_kimi_k2_diagram.png", dpi=300,
                           bbox_inches='tight', facecolor='white')
                plt.close('all')
        except Exception as exec_error:
            log(f"코드 실행 오류: {exec_error}", "WARN")
            # 기본 다이어그램 생성으로 폴백
            pass

        elapsed_total = time.time() - start_time
        output_path = RESULTS_DIR / "api_kimi_k2_diagram.png"

        if output_path.exists():
            log(f"저장 완료: {output_path} ({elapsed_total:.1f}초)", "SUCCESS")
            return {
                "model": "Kimi K2 (Moonshot)",
                "output": str(output_path),
                "code": str(code_path),
                "time": elapsed_total,
                "success": True
            }
        else:
            log("이미지 생성 실패, 코드만 저장됨", "WARN")
            return {
                "model": "Kimi K2 (Moonshot)",
                "code": str(code_path),
                "time": elapsed_total,
                "success": False,
                "note": "코드 생성됨, 실행 필요"
            }

    except Exception as e:
        log(f"Kimi K2 오류: {e}", "ERROR")
        return {"model": "Kimi K2 (Moonshot)", "error": str(e), "success": False}


# ========== 4. DeepSeek ==========
def test_deepseek():
    """DeepSeek 코드 생성 테스트"""
    log("DeepSeek 테스트 시작", "START")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        log("DEEPSEEK_API_KEY 없음", "ERROR")
        return None

    try:
        log("DeepSeek 코드 생성 중...")
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a Python expert. Output only executable Python code, no explanations or markdown."},
                {"role": "user", "content": CODE_PROMPT}
            ],
            "temperature": 0.3
        }

        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        code = result['choices'][0]['message']['content']

        # 코드에서 ```python ``` 제거
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        elapsed_gen = time.time() - start_time
        log(f"코드 생성 완료 ({elapsed_gen:.1f}초)")

        # 코드 저장
        code_path = RESULTS_DIR / "api_deepseek_code.py"
        with open(code_path, 'w') as f:
            f.write(code)

        # 코드 실행
        log("생성된 코드 실행 중...")
        import matplotlib
        matplotlib.use('Agg')

        exec_globals = {"__name__": "__main__"}
        try:
            exec(code, exec_globals)
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                plt.savefig(RESULTS_DIR / "api_deepseek_diagram.png", dpi=300,
                           bbox_inches='tight', facecolor='white')
                plt.close('all')
        except Exception as exec_error:
            log(f"코드 실행 오류: {exec_error}", "WARN")

        elapsed_total = time.time() - start_time
        output_path = RESULTS_DIR / "api_deepseek_diagram.png"

        if output_path.exists():
            log(f"저장 완료: {output_path} ({elapsed_total:.1f}초)", "SUCCESS")
            return {
                "model": "DeepSeek",
                "output": str(output_path),
                "code": str(code_path),
                "time": elapsed_total,
                "success": True
            }
        else:
            return {
                "model": "DeepSeek",
                "code": str(code_path),
                "time": elapsed_total,
                "success": False,
                "note": "코드 생성됨, 실행 필요"
            }

    except Exception as e:
        log(f"DeepSeek 오류: {e}", "ERROR")
        return {"model": "DeepSeek", "error": str(e), "success": False}


# ========== 결과 요약 ==========
def generate_report(results: list):
    """테스트 결과 리포트 생성"""
    report_path = RESULTS_DIR / "api_test_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 다이어그램 파이프라인 API 테스트 결과\n\n")
        f.write(f"테스트 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 요약\n\n")
        f.write("| 모델 | 상태 | 소요시간 | 출력 |\n")
        f.write("|------|------|----------|------|\n")

        for r in results:
            if r is None:
                continue
            status = "✅ 성공" if r.get('success') else "❌ 실패"
            time_str = f"{r.get('time', 0):.1f}초" if r.get('time') else "-"
            output = Path(r.get('output', '-')).name if r.get('output') else r.get('note', '-')
            f.write(f"| {r.get('model', 'Unknown')} | {status} | {time_str} | {output} |\n")

        f.write("\n---\n\n")
        f.write("## 상세 결과\n\n")

        for r in results:
            if r is None:
                continue
            f.write(f"### {r.get('model', 'Unknown')}\n\n")
            f.write(f"```json\n{json.dumps(r, indent=2, ensure_ascii=False)}\n```\n\n")

        f.write("---\n\n")
        f.write("## 생성된 파일\n\n")
        for file in RESULTS_DIR.glob("api_*"):
            size = file.stat().st_size
            size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
            f.write(f"- `{file.name}` ({size_str})\n")

    log(f"리포트 저장: {report_path}", "SUCCESS")
    return report_path


# ========== 메인 ==========
def main():
    print("\n" + "=" * 60)
    print("🎨 다이어그램 파이프라인 API 자동 테스트")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    results = []

    # 1. OpenAI DALL-E 3
    print("\n" + "-" * 40)
    result = test_openai_dalle()
    results.append(result)

    # 2. Google Gemini
    print("\n" + "-" * 40)
    result = test_google_gemini()
    results.append(result)

    # 3. Kimi K2
    print("\n" + "-" * 40)
    result = test_kimi_k2()
    results.append(result)

    # 4. DeepSeek
    print("\n" + "-" * 40)
    result = test_deepseek()
    results.append(result)

    # 결과 리포트
    print("\n" + "-" * 40)
    log("테스트 결과 리포트 생성 중...")
    report_path = generate_report(results)

    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 테스트 완료!")
    print("=" * 60)

    success_count = sum(1 for r in results if r and r.get('success'))
    total_count = sum(1 for r in results if r is not None)

    print(f"\n✅ 성공: {success_count}/{total_count}")
    print(f"📁 결과 폴더: {RESULTS_DIR}")
    print(f"📋 리포트: {report_path}")

    print("\n생성된 파일:")
    for file in sorted(RESULTS_DIR.glob("api_*")):
        size = file.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
        print(f"  • {file.name} ({size_str})")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
