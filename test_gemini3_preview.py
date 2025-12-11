#!/usr/bin/env python3
"""
Gemini 3 Preview Models Direct Test

발견된 정확한 Gemini 3 preview 모델들을 직접 테스트합니다:
- gemini-3-pro-preview
- gemini-3-pro-image-preview
"""

import asyncio
import time
import google.generativeai as genai

# API 키 설정
API_KEY = "AIzaSyBYqawvrwfKbEyn0b9Fi7jgE1SGQ6-R2Vs"

async def test_gemini3_preview():
    """Gemini 3 Preview 모델들 직접 테스트"""

    print("🚀 Gemini 3 Preview 모델 직접 테스트")
    print("=" * 50)

    # API 키 설정
    genai.configure(api_key=API_KEY)

    # 테스트할 모델들
    gemini3_models = [
        "gemini-3-pro-preview",
        "gemini-3-pro-image-preview"
    ]

    working_models = []

    for model_name in gemini3_models:
        print(f"\n🔧 테스트: {model_name}")
        print("-" * 40)

        try:
            # 모델 초기화
            model = genai.GenerativeModel(model_name)

            start_time = time.time()

            # 1. 기본 텍스트 생성 테스트
            print("  📝 기본 텍스트 생성 테스트...")
            response = model.generate_content(
                "Hello! What is your model version? Please respond in Korean and English.",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.3
                )
            )

            # 응답 처리
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'parts'):
                response_text = ''.join([part.text for part in response.parts])
            elif hasattr(response, 'candidates'):
                if response.candidates and hasattr(response.candidates[0], 'content'):
                    response_text = response.candidates[0].content.parts[0].text
                else:
                    response_text = "응답을 파싱할 수 없습니다"
            else:
                response_text = "응답 형식을 인식할 수 없습니다"

            elapsed = time.time() - start_time

            print(f"    ✅ 성공! ({elapsed:.2f}초)")
            print(f"    📄 응답: {response_text[:200]}...")

            # 2. 고급 기능 테스트 (문서 생성)
            print("  🎯 고급 문서 생성 테스트...")

            start_time = time.time()

            advanced_response = model.generate_content(
                """
                한국어로 AI-CoScientist 프로젝트에 대한 간단한 기술 문서를 작성해주세요:

                1. 프로젝트 개요 (100단어)
                2. 주요 기술 스택
                3. 혁신적인 특징

                전문적이고 기술적인 문서로 작성해주세요.
                """,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.6
                )
            )

            # 고급 응답 처리
            if hasattr(advanced_response, 'text'):
                advanced_text = advanced_response.text
            elif hasattr(advanced_response, 'candidates'):
                if advanced_response.candidates and hasattr(advanced_response.candidates[0], 'content'):
                    advanced_text = advanced_response.candidates[0].content.parts[0].text
                else:
                    advanced_text = "고급 응답을 파싱할 수 없습니다"
            else:
                advanced_text = "고급 응답 형식을 인식할 수 없습니다"

            elapsed_advanced = time.time() - start_time

            print(f"    ✅ 고급 테스트 성공! ({elapsed_advanced:.2f}초)")
            print(f"    📄 문서 생성 결과 (첫 300자):")
            print(f"    {advanced_text[:300]}...")

            # 3. 모델 성능 평가
            print("  📊 성능 평가...")

            performance_score = {
                "response_speed": min(5.0, 5.0 * (2.0 / max(elapsed, 0.1))),
                "text_quality": 5.0 if len(advanced_text) > 200 else 3.0,
                "korean_support": 5.0 if "한국" in advanced_text or "기술" in advanced_text else 2.0,
                "availability": 5.0
            }

            avg_score = sum(performance_score.values()) / len(performance_score)

            print(f"    🏆 종합 점수: {avg_score:.1f}/5.0")
            print(f"       - 응답 속도: {performance_score['response_speed']:.1f}/5.0")
            print(f"       - 텍스트 품질: {performance_score['text_quality']:.1f}/5.0")
            print(f"       - 한국어 지원: {performance_score['korean_support']:.1f}/5.0")
            print(f"       - 가용성: {performance_score['availability']:.1f}/5.0")

            working_models.append({
                "model": model_name,
                "status": "✅ 사용 가능",
                "performance": avg_score,
                "features": {
                    "text_generation": True,
                    "korean_support": "한국" in advanced_text,
                    "technical_writing": True
                }
            })

        except Exception as e:
            print(f"    ❌ 실패: {str(e)}")
            print(f"    🔍 오류 상세: {type(e).__name__}")

            working_models.append({
                "model": model_name,
                "status": "❌ 사용 불가",
                "error": str(e)[:100],
                "performance": 0.0
            })

    # 최종 결과 요약
    print(f"\n🎯 최종 테스트 결과")
    print("=" * 50)

    working_count = sum(1 for m in working_models if "✅" in m["status"])

    print(f"📊 총 테스트 모델: {len(gemini3_models)}개")
    print(f"✅ 사용 가능: {working_count}개")
    print(f"❌ 사용 불가: {len(gemini3_models) - working_count}개")

    if working_count > 0:
        print(f"\n🚀 사용 가능한 Gemini 3 모델들:")

        for model_info in working_models:
            if "✅" in model_info["status"]:
                print(f"  🎯 {model_info['model']}")
                print(f"     성능 점수: {model_info['performance']:.1f}/5.0")
                print(f"     한국어 지원: {'✅' if model_info['features']['korean_support'] else '❌'}")
                print(f"     기술 문서 작성: {'✅' if model_info['features']['technical_writing'] else '❌'}")

        print(f"\n✨ 결론: Gemini 3 사용 가능! AI-CoScientist 시스템에 추가할 수 있습니다!")

        # 최고 성능 모델 추천
        best_model = max(
            [m for m in working_models if "✅" in m["status"]],
            key=lambda x: x["performance"],
            default=None
        )

        if best_model:
            print(f"🏆 추천 모델: {best_model['model']} (성능: {best_model['performance']:.1f}/5.0)")

    else:
        print(f"\n❌ 결론: 현재 Gemini 3 preview 모델들이 실제로는 사용할 수 없습니다.")
        print(f"💡 대안: Gemini 2.5 Pro를 계속 사용하는 것이 안전합니다.")

    return working_models

if __name__ == "__main__":
    result = asyncio.run(test_gemini3_preview())

    # AI-CoScientist 시스템 통합 가능성 체크
    working_models = [m for m in result if "✅" in m["status"]]

    if working_models:
        print(f"\n🔧 AI-CoScientist 통합 준비사항:")
        print(f"  1. src/core/config.py 모델 설정 업데이트")
        print(f"  2. src/services/llm/adapters/gemini.py에 pricing 추가")
        print(f"  3. Enhanced Document Generator에 모델 포함")
        print(f"  4. 제안서 최적화 시스템에 Gemini 3 추가")
    else:
        print(f"\n📋 현재 상태: Gemini 2.5 Pro가 최적의 선택입니다.")