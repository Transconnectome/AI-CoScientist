#!/usr/bin/env python3
"""
특정 Gemini 3 모델명 직접 테스트

사용자 제공 정확한 모델명들로 테스트:
- gemini-3-pro-preview
- gemini-3-pro-preview-11-2025
- gemini-3-pro-preview-11-2025-thinking
"""

import asyncio
import time
import google.generativeai as genai

# API 키 설정
API_KEY = "AIzaSyBYqawvrwfKbEyn0b9Fi7jgE1SGQ6-R2Vs"

async def test_specific_gemini3_models():
    """사용자 제공 정확한 Gemini 3 모델명들 테스트"""

    print("🎯 사용자 제공 정확한 Gemini 3 모델 테스트")
    print("=" * 60)

    # API 키 설정
    genai.configure(api_key=API_KEY)

    # 테스트할 정확한 모델명들
    target_models = [
        "gemini-3-pro-preview",
        "gemini-3-pro-preview-11-2025",
        "gemini-3-pro-preview-11-2025-thinking"
    ]

    working_models = []

    for model_name in target_models:
        print(f"\n🔧 테스트: {model_name}")
        print("-" * 50)

        try:
            # 모델 초기화 시도
            print("  🚀 모델 초기화 중...")
            model = genai.GenerativeModel(model_name)

            # 1. 기본 텍스트 생성 테스트
            print("  📝 기본 응답 테스트...")
            start_time = time.time()

            response = model.generate_content(
                "안녕하세요! 당신의 모델명과 버전을 알려주세요. 한국어와 영어로 간단히 응답해주세요.",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.3
                )
            )

            elapsed = time.time() - start_time

            # 응답 처리 (여러 방법 시도)
            response_text = None

            # 방법 1: response.text
            if hasattr(response, 'text') and response.text:
                response_text = response.text
                print(f"    ✅ response.text 방식 성공!")

            # 방법 2: response.candidates
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content') and response.candidates[0].content:
                    if hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                        response_text = response.candidates[0].content.parts[0].text
                        print(f"    ✅ candidates.content.parts 방식 성공!")

            # 방법 3: response.parts
            elif hasattr(response, 'parts') and response.parts:
                response_text = ''.join([part.text for part in response.parts if hasattr(part, 'text')])
                print(f"    ✅ response.parts 방식 성공!")

            if response_text:
                print(f"    📄 응답 내용 (첫 200자): {response_text[:200]}...")
                print(f"    ⏱️  응답 시간: {elapsed:.2f}초")

                # 2. 고급 AI-CoScientist 문서 생성 테스트
                print("  🎯 AI-CoScientist 문서 생성 테스트...")

                advanced_start = time.time()

                advanced_response = model.generate_content(
                    """
AI-CoScientist 발달장애 연구 제안서에서 "NeuroX-Fusion 10B Foundation Model"의 혁신적 특징을 전문적으로 설명해주세요.

포함할 내용:
1. 발달장애 특화 AI 아키텍처
2. 멀티모달 학습 능력
3. 임상 적용 가능성

500-600단어로 학술적이고 설득력 있게 작성해주세요.
                    """,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=1000,
                        temperature=0.7
                    )
                )

                advanced_elapsed = time.time() - advanced_start

                # 고급 응답 처리
                advanced_text = None

                if hasattr(advanced_response, 'text') and advanced_response.text:
                    advanced_text = advanced_response.text
                elif hasattr(advanced_response, 'candidates') and advanced_response.candidates:
                    if (hasattr(advanced_response.candidates[0], 'content') and
                        advanced_response.candidates[0].content and
                        hasattr(advanced_response.candidates[0].content, 'parts') and
                        advanced_response.candidates[0].content.parts):
                        advanced_text = advanced_response.candidates[0].content.parts[0].text

                if advanced_text:
                    print(f"    ✅ 고급 테스트 성공! ({advanced_elapsed:.2f}초)")
                    print(f"    📄 생성된 문서 (첫 300자):")
                    print(f"    {advanced_text[:300]}...")

                    # 3. 품질 평가
                    print("  📊 품질 평가...")

                    # 한국어 지원 확인
                    korean_words = ["발달장애", "아키텍처", "모델", "연구", "학습", "임상"]
                    korean_support = any(word in advanced_text for word in korean_words)

                    # 기술적 깊이 확인
                    technical_words = ["AI", "멀티모달", "알고리즘", "성능", "벤치마크", "신경망"]
                    technical_depth = any(word in advanced_text for word in technical_words)

                    # 혁신성 내용 확인
                    innovation_words = ["혁신", "최초", "개선", "특화", "향상", "breakthrough"]
                    innovation_content = any(word in advanced_text for word in innovation_words)

                    # 종합 점수 계산
                    speed_score = min(5.0, 5.0 * (3.0 / max(elapsed, 0.1)))
                    quality_score = (
                        (5.0 if korean_support else 2.0) * 0.3 +
                        (5.0 if technical_depth else 2.0) * 0.4 +
                        (5.0 if innovation_content else 2.0) * 0.3
                    )
                    length_score = min(5.0, len(advanced_text) / 200)

                    total_score = (speed_score + quality_score + length_score) / 3

                    print(f"    🏆 종합 성능: {total_score:.1f}/5.0")
                    print(f"       - 응답 속도: {speed_score:.1f}/5.0")
                    print(f"       - 내용 품질: {quality_score:.1f}/5.0")
                    print(f"       - 길이 적절성: {length_score:.1f}/5.0")
                    print(f"    ✨ 기능 평가:")
                    print(f"       - 한국어 지원: {'✅' if korean_support else '❌'}")
                    print(f"       - 기술적 깊이: {'✅' if technical_depth else '❌'}")
                    print(f"       - 혁신성 내용: {'✅' if innovation_content else '❌'}")

                    # 4. 모델별 특성 분석
                    print("  🔍 모델 특성 분석...")

                    model_features = {}

                    # Thinking 모델 여부
                    if "thinking" in model_name.lower():
                        model_features["reasoning_enhanced"] = True
                        print(f"    🧠 추론 강화 모델 (Thinking)")

                    # Preview 모델 여부
                    if "preview" in model_name.lower():
                        model_features["preview_features"] = True
                        print(f"    🚀 프리뷰 기능 탑재")

                    # 날짜별 모델
                    if "11-2025" in model_name:
                        model_features["latest_version"] = True
                        print(f"    📅 최신 버전 (2025년 11월)")

                    # 성공 모델 정보 저장
                    working_models.append({
                        "model": model_name,
                        "status": "✅ 사용 가능",
                        "basic_response_time": elapsed,
                        "advanced_response_time": advanced_elapsed,
                        "performance_score": total_score,
                        "features": {
                            "korean_support": korean_support,
                            "technical_depth": technical_depth,
                            "innovation_content": innovation_content,
                            "document_generation": True,
                            **model_features
                        },
                        "basic_response": response_text[:200] + "...",
                        "advanced_response": advanced_text[:300] + "..."
                    })

                else:
                    print("    ❌ 고급 테스트 응답 파싱 실패")

            else:
                print("    ❌ 기본 응답 파싱 실패")

                # 응답 상태 디버깅
                print("    🔍 응답 디버깅:")
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        print(f"       finish_reason: {candidate.finish_reason}")
                    if hasattr(candidate, 'safety_ratings'):
                        print(f"       safety_ratings: {candidate.safety_ratings}")

        except Exception as e:
            print(f"    ❌ 테스트 실패: {str(e)}")
            print(f"    🔍 오류 유형: {type(e).__name__}")

            working_models.append({
                "model": model_name,
                "status": "❌ 사용 불가",
                "error": str(e)[:200],
                "error_type": type(e).__name__
            })

    # 최종 결과 요약
    print(f"\n🎯 최종 테스트 결과 요약")
    print("=" * 60)

    working_count = len([m for m in working_models if "✅" in m["status"]])
    failed_count = len([m for m in working_models if "❌" in m["status"]])

    print(f"📊 총 테스트 모델: {len(target_models)}개")
    print(f"✅ 사용 가능: {working_count}개")
    print(f"❌ 사용 불가: {failed_count}개")

    if working_count > 0:
        print(f"\n🚀 사용 가능한 Gemini 3 모델들:")

        # 성능순 정렬
        successful_models = [m for m in working_models if "✅" in m["status"]]
        successful_models.sort(key=lambda x: x["performance_score"], reverse=True)

        for i, model_info in enumerate(successful_models):
            print(f"  {i+1}. 🎯 {model_info['model']}")
            print(f"     성능 점수: {model_info['performance_score']:.1f}/5.0")
            print(f"     기본 응답: {model_info['basic_response_time']:.2f}초")
            print(f"     문서 생성: {model_info['advanced_response_time']:.2f}초")

            features = model_info['features']
            print(f"     특징:")
            print(f"       - 한국어: {'✅' if features['korean_support'] else '❌'}")
            print(f"       - 기술문서: {'✅' if features['technical_depth'] else '❌'}")
            if features.get('reasoning_enhanced'):
                print(f"       - 추론강화: ✅")
            if features.get('latest_version'):
                print(f"       - 최신버전: ✅")

        # 최고 성능 모델 추천
        best_model = successful_models[0] if successful_models else None

        if best_model:
            print(f"\n🏆 최고 성능 모델: {best_model['model']}")
            print(f"    종합 점수: {best_model['performance_score']:.1f}/5.0")

        print(f"\n✨ 결론: Gemini 3 사용 가능!")
        print(f"🔧 AI-CoScientist UPE 시스템에 즉시 통합 가능합니다!")

        return successful_models

    else:
        print(f"\n❌ 결론: 제공된 Gemini 3 모델들도 접근 불가")
        print(f"🔍 오류 분석:")
        for model_info in working_models:
            if "❌" in model_info["status"]:
                print(f"  • {model_info['model']}: {model_info['error'][:100]}...")

        print(f"\n💡 대안: Gemini 2.5 Pro + Claude Sonnet 4.5 조합 계속 사용")

        return []

if __name__ == "__main__":
    result = asyncio.run(test_specific_gemini3_models())

    if result:
        print(f"\n🚀 다음 단계: Gemini 3 모델들을 Enhanced Document Generator에 통합")
        for model in result:
            print(f"  • {model['model']} (점수: {model['performance_score']:.1f}/5.0)")
    else:
        print(f"\n📋 다음 단계: Gemini 2.5 Pro + Claude Sonnet 4.5로 UPE 보강")