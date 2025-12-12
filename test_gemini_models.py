#!/usr/bin/env python3
"""
Gemini 3 Availability Test Script

직접 Google Gemini API를 테스트해서 사용 가능한 모델들을 확인하고
특히 Gemini 3가 사용 가능한지 테스트합니다.
"""

import asyncio
import os
from typing import List, Dict
import google.generativeai as genai

# API 키 설정
API_KEY = "AIzaSyBYqawvrwfKbEyn0b9Fi7jgE1SGQ6-R2Vs"

async def test_gemini_models():
    """Gemini 모델들 테스트"""

    print("🔍 Gemini API 모델 가용성 테스트 시작...")
    print("=" * 60)

    # API 키 설정
    genai.configure(api_key=API_KEY)

    # 1. 사용 가능한 모델 목록 확인
    print("\n📋 사용 가능한 모델 목록:")
    try:
        models = genai.list_models()
        available_models = []

        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                available_models.append(model_name)
                print(f"  ✅ {model_name}")

                # Gemini 3 관련 모델 체크
                if 'gemini-3' in model_name.lower():
                    print(f"  🎯 GEMINI 3 발견! -> {model_name}")

    except Exception as e:
        print(f"  ❌ 모델 목록 조회 실패: {e}")
        return False

    # 2. 특정 Gemini 3 모델들 직접 테스트
    gemini_3_candidates = [
        "gemini-3-pro",
        "gemini-3.0-pro",
        "gemini-3-flash",
        "gemini-3.0-flash",
        "gemini-3-ultra",
        "gemini-3.0-ultra",
        "gemini-3",
        "models/gemini-3-pro",
        "models/gemini-3.0-pro"
    ]

    print(f"\n🧪 Gemini 3 후보 모델들 직접 테스트:")
    gemini_3_working = []

    for model_name in gemini_3_candidates:
        try:
            print(f"  🔧 테스트 중: {model_name}")

            # 모델 초기화 시도
            model = genai.GenerativeModel(model_name)

            # 간단한 테스트 쿼리
            response = model.generate_content(
                "Hello, test message. Please respond with 'Gemini 3 working'",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=50,
                    temperature=0.1
                )
            )

            print(f"    ✅ 성공! 응답: {response.text[:100]}")
            gemini_3_working.append(model_name)

        except Exception as e:
            print(f"    ❌ 실패: {str(e)[:100]}")

    # 3. 최신 모델 확인 (2.5 Pro와 비교)
    print(f"\n🚀 최신 모델들 성능 테스트:")

    test_models = [
        "gemini-2.5-pro",
        "gemini-1.5-pro",
    ]

    # Gemini 3가 발견된 경우 추가
    if gemini_3_working:
        test_models.extend(gemini_3_working[:2])  # 최대 2개

    for model_name in test_models:
        try:
            print(f"  🔧 성능 테스트: {model_name}")

            model = genai.GenerativeModel(model_name)

            # 복잡한 테스트 쿼리
            response = model.generate_content(
                "What is the latest version of Google Gemini models? List all available versions.",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.3
                )
            )

            print(f"    📝 응답 (첫 150자): {response.text[:150]}...")

            # 모델 정보 확인
            if hasattr(model, '_model_name'):
                print(f"    🏷️  실제 모델명: {model._model_name}")

        except Exception as e:
            print(f"    ❌ 테스트 실패: {str(e)[:100]}")

    # 4. 결론
    print(f"\n📊 테스트 결과 요약:")
    print(f"  • 전체 사용 가능 모델: {len(available_models)}개")
    print(f"  • Gemini 3 사용 가능: {'✅ ' + str(len(gemini_3_working)) + '개' if gemini_3_working else '❌ 없음'}")

    if gemini_3_working:
        print(f"\n🎉 사용 가능한 Gemini 3 모델들:")
        for model in gemini_3_working:
            print(f"    ✅ {model}")
        return gemini_3_working
    else:
        print(f"\n💡 대안: 현재 사용 가능한 최고급 모델은 Gemini 2.5 Pro입니다.")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_gemini_models())

    if result:
        print(f"\n🚀 결론: Gemini 3 사용 가능! 모델을 시스템에 추가할 수 있습니다.")
    else:
        print(f"\n📋 결론: Gemini 3 현재 사용 불가. Gemini 2.5 Pro가 최신입니다.")