#!/usr/bin/env python3
"""
특정 GPT-5/GPT-5.1 모델명 직접 테스트

사용자 제공 정확한 모델명들로 테스트:
- gpt-5
- gpt-5-thinking
- gpt-5-main
- gpt-5.1
- gpt-5.1-chat-latest
"""

import asyncio
import openai
import time

# 새로운 API 키
API_KEY = "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"

async def test_specific_models():
    """사용자 제공 정확한 모델명들 테스트"""

    print("🎯 사용자 제공 정확한 GPT-5/GPT-5.1 모델 테스트")
    print("=" * 60)

    # OpenAI 클라이언트 초기화
    client = openai.AsyncOpenAI(api_key=API_KEY)

    # 테스트할 정확한 모델명들
    target_models = [
        "gpt-5",
        "gpt-5-thinking",
        "gpt-5-main",
        "gpt-5.1",
        "gpt-5.1-chat-latest"
    ]

    working_models = []

    for model_name in target_models:
        print(f"\n🔧 테스트: {model_name}")
        print("-" * 40)

        try:
            start_time = time.time()

            # 1. 기본 테스트
            print("  📝 기본 응답 테스트...")
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "안녕하세요! 당신의 모델명과 버전을 알려주세요. 한국어와 영어로 응답해주세요."}
                ],
                max_tokens=200,
                temperature=0.3
            )

            elapsed = time.time() - start_time

            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content

                print(f"    ✅ 성공! (응답 시간: {elapsed:.2f}초)")
                print(f"    📄 응답: {content[:250]}...")

                # 2. 고급 기능 테스트 - AI-CoScientist 관련
                print("  🚀 AI-CoScientist 문서 생성 테스트...")

                start_time = time.time()

                advanced_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": """
AI-CoScientist 발달장애 연구 제안서의 "NeuroX-Fusion 10B Foundation Model" 섹션을 전문적으로 작성해주세요.

다음 요소들을 포함해주세요:
1. 기술적 혁신점 (3가지)
2. 발달장애 특화 기능
3. 성능 벤치마크 예상
4. 임상 적용 가능성

500-700단어로 학술적이고 설득력 있게 작성해주세요.
                        """
                    }],
                    max_tokens=1000,
                    temperature=0.7
                )

                advanced_elapsed = time.time() - start_time

                if advanced_response.choices and advanced_response.choices[0].message.content:
                    advanced_content = advanced_response.choices[0].message.content

                    print(f"    ✅ 고급 테스트 성공! (응답 시간: {advanced_elapsed:.2f}초)")
                    print(f"    📄 문서 생성 결과 (첫 300자):")
                    print(f"    {advanced_content[:300]}...")

                    # 3. 성능 평가
                    print("  📊 성능 평가...")

                    # 응답 품질 분석
                    korean_support = any(word in advanced_content for word in ["발달장애", "모델", "연구", "기술"])
                    technical_depth = any(word in advanced_content for word in ["알고리즘", "아키텍처", "성능", "벤치마크", "임상"])
                    innovation_content = any(word in advanced_content for word in ["혁신", "특화", "최초", "개선"])

                    # 종합 점수
                    speed_score = min(5.0, 5.0 * (2.0 / max(elapsed, 0.1)))
                    quality_score = (
                        (5.0 if korean_support else 2.0) * 0.3 +
                        (5.0 if technical_depth else 2.0) * 0.4 +
                        (5.0 if innovation_content else 2.0) * 0.3
                    )
                    length_score = min(5.0, len(advanced_content) / 200)

                    total_score = (speed_score + quality_score + length_score) / 3

                    print(f"    🏆 종합 성능 점수: {total_score:.1f}/5.0")
                    print(f"       - 응답 속도: {speed_score:.1f}/5.0")
                    print(f"       - 내용 품질: {quality_score:.1f}/5.0")
                    print(f"       - 길이 적절성: {length_score:.1f}/5.0")
                    print(f"       - 한국어 지원: {'✅' if korean_support else '❌'}")
                    print(f"       - 기술적 깊이: {'✅' if technical_depth else '❌'}")
                    print(f"       - 혁신성 내용: {'✅' if innovation_content else '❌'}")

                    # 모델 정보 저장
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
                            "document_generation": True
                        },
                        "content_sample": advanced_content[:200] + "..."
                    })

                else:
                    print("    ❌ 고급 테스트 응답 없음")

            else:
                print("    ❌ 기본 테스트 응답 없음")

        except Exception as e:
            print(f"    ❌ 실패: {str(e)}")

            working_models.append({
                "model": model_name,
                "status": "❌ 사용 불가",
                "error": str(e)[:150],
                "performance_score": 0.0
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
        print(f"\n🚀 사용 가능한 모델들:")

        # 성능순 정렬
        successful_models = [m for m in working_models if "✅" in m["status"]]
        successful_models.sort(key=lambda x: x["performance_score"], reverse=True)

        for i, model_info in enumerate(successful_models):
            print(f"  {i+1}. 🎯 {model_info['model']}")
            print(f"     성능 점수: {model_info['performance_score']:.1f}/5.0")
            print(f"     기본 응답: {model_info['basic_response_time']:.2f}초")
            print(f"     문서 생성: {model_info['advanced_response_time']:.2f}초")
            print(f"     한국어 지원: {'✅' if model_info['features']['korean_support'] else '❌'}")
            print(f"     기술 문서: {'✅' if model_info['features']['technical_depth'] else '❌'}")

        # 최고 성능 모델 추천
        best_model = successful_models[0] if successful_models else None

        if best_model:
            print(f"\n🏆 최고 성능 모델: {best_model['model']}")
            print(f"    종합 점수: {best_model['performance_score']:.1f}/5.0")
            print(f"    📄 생성 샘플:")
            print(f"    {best_model['content_sample']}")

        print(f"\n✨ 결론: GPT-5/GPT-5.1 사용 가능!")
        print(f"🔧 AI-CoScientist UPE 시스템에 즉시 통합 가능합니다!")

        return successful_models

    else:
        print(f"\n❌ 결론: 제공된 모델명들도 접근 불가")
        print(f"🔍 오류 분석:")
        for model_info in working_models:
            if "❌" in model_info["status"]:
                print(f"  • {model_info['model']}: {model_info['error'][:100]}...")

        print(f"\n💡 대안: Gemini 2.5 Pro + Claude Sonnet 4.5 조합 사용 권장")

        return []

if __name__ == "__main__":
    result = asyncio.run(test_specific_models())

    if result:
        print(f"\n🚀 다음 단계: Enhanced Document Generator에 성공 모델들 통합")
        for model in result:
            print(f"  • {model['model']} (점수: {model['performance_score']:.1f}/5.0)")
    else:
        print(f"\n📋 다음 단계: Gemini 2.5 Pro + Claude Sonnet 4.5로 UPE 보강")