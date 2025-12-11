#!/usr/bin/env python3
"""
GPT-5.1 Availability Test Script

GPT-5.1이 OpenAI API를 통해 사용 가능한지 확인합니다.
"""

import asyncio
import openai
import os
from typing import List, Dict
import time

# 새로운 API 키 설정
API_KEY = "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"

async def test_gpt51():
    """GPT-5.1 및 최신 GPT 모델들 테스트"""

    print("🔍 GPT-5.1 및 최신 OpenAI 모델 가용성 테스트")
    print("=" * 60)

    # OpenAI 클라이언트 초기화
    client = openai.AsyncOpenAI(api_key=API_KEY)

    # 1. 사용 가능한 모델 목록 조회
    print("\n📋 사용 가능한 OpenAI 모델 목록:")
    try:
        models = await client.models.list()
        available_models = []

        for model in models.data:
            model_id = model.id
            available_models.append(model_id)

            # GPT-5 관련 모델 하이라이트
            if 'gpt-5' in model_id.lower():
                print(f"  🎯 GPT-5 관련: {model_id}")
            elif 'gpt-4' in model_id:
                print(f"  ✅ GPT-4 계열: {model_id}")
            elif model_id in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k']:
                print(f"  📝 GPT-3.5: {model_id}")

        print(f"\n총 사용 가능 모델: {len(available_models)}개")

    except Exception as e:
        print(f"  ❌ 모델 목록 조회 실패: {e}")
        return False

    # 2. GPT-5.1 후보 모델들 직접 테스트
    gpt51_candidates = [
        "gpt-5.1",
        "gpt-5.1-turbo",
        "gpt-5.1-preview",
        "gpt-5-1",
        "gpt-5-1-turbo",
        "gpt-5.1-pro",
        "gpt-5.1-ultra"
    ]

    print(f"\n🧪 GPT-5.1 후보 모델들 직접 테스트:")
    working_gpt51 = []

    for model_name in gpt51_candidates:
        try:
            print(f"  🔧 테스트 중: {model_name}")

            start_time = time.time()

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is your model version? Respond briefly."}
                ],
                max_tokens=100,
                temperature=0.1
            )

            elapsed = time.time() - start_time

            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                print(f"    ✅ 성공! ({elapsed:.2f}초)")
                print(f"    📄 응답: {content[:150]}...")

                working_gpt51.append({
                    "model": model_name,
                    "response_time": elapsed,
                    "content": content
                })

        except Exception as e:
            print(f"    ❌ 실패: {str(e)[:100]}")

    # 3. GPT-5 기본 모델 테스트
    print(f"\n🚀 GPT-5 기본 모델 테스트:")

    gpt5_models = ["gpt-5", "gpt-5-turbo", "gpt-5-preview"]
    working_gpt5 = []

    for model_name in gpt5_models:
        try:
            print(f"  🔧 테스트 중: {model_name}")

            start_time = time.time()

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": "안녕하세요! AI-CoScientist 프로젝트에 대한 기술 문서를 1문단으로 작성해주세요."}
                ],
                max_tokens=300,
                temperature=0.6
            )

            elapsed = time.time() - start_time

            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                print(f"    ✅ 성공! ({elapsed:.2f}초)")
                print(f"    📄 한국어 문서 생성: {content[:200]}...")

                working_gpt5.append({
                    "model": model_name,
                    "response_time": elapsed,
                    "korean_support": "한국" in content or "기술" in content,
                    "content_length": len(content)
                })

        except Exception as e:
            print(f"    ❌ 실패: {str(e)[:100]}")

    # 4. 최신 GPT 모델들 성능 비교
    print(f"\n📊 성능 비교 테스트:")

    comparison_models = []
    comparison_models.extend([m["model"] for m in working_gpt51])
    comparison_models.extend([m["model"] for m in working_gpt5])

    if not comparison_models:
        comparison_models = ["gpt-4o", "gpt-4-turbo", "gpt-4"]

    best_model = None
    best_score = 0

    for model_name in comparison_models[:3]:  # 최대 3개 모델만 비교
        try:
            print(f"  📈 성능 테스트: {model_name}")

            start_time = time.time()

            response = await client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": """AI-CoScientist 제안서 최적화 시스템의 혁신적 특징 3가지를 기술적으로 설명해주세요:
1. Multi-Agent 협업 시스템
2. 6-Strategy RAG Orchestrator
3. Enhanced Document Generation

각각 2-3문장으로 전문적이고 구체적으로 설명해주세요."""
                }],
                max_tokens=600,
                temperature=0.7
            )

            elapsed = time.time() - start_time

            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content

                # 성능 점수 계산
                speed_score = min(5.0, 5.0 * (3.0 / max(elapsed, 0.1)))
                length_score = min(5.0, len(content) / 200)
                korean_score = 5.0 if any(word in content for word in ["시스템", "기술", "협업", "최적화"]) else 2.0

                total_score = (speed_score + length_score + korean_score) / 3

                print(f"    🏆 종합 점수: {total_score:.1f}/5.0")
                print(f"       - 응답 속도: {speed_score:.1f}/5.0 ({elapsed:.2f}s)")
                print(f"       - 내용 풍부도: {length_score:.1f}/5.0 ({len(content)}자)")
                print(f"       - 한국어 지원: {korean_score:.1f}/5.0")

                if total_score > best_score:
                    best_score = total_score
                    best_model = model_name

        except Exception as e:
            print(f"    ❌ 성능 테스트 실패: {str(e)[:100]}")

    # 5. 최종 결과 및 추천
    print(f"\n🎯 최종 테스트 결과")
    print("=" * 50)

    print(f"📊 GPT-5.1 후보: {len(working_gpt51)}개 사용 가능")
    print(f"📊 GPT-5 기본: {len(working_gpt5)}개 사용 가능")

    if working_gpt51:
        print(f"\n🚀 사용 가능한 GPT-5.1 모델들:")
        for model_info in working_gpt51:
            print(f"  🎯 {model_info['model']} (응답시간: {model_info['response_time']:.2f}s)")

    if working_gpt5:
        print(f"\n✅ 사용 가능한 GPT-5 모델들:")
        for model_info in working_gpt5:
            print(f"  🎯 {model_info['model']} (한국어: {'✅' if model_info['korean_support'] else '❌'})")

    if best_model:
        print(f"\n🏆 최고 성능 모델: {best_model} (점수: {best_score:.1f}/5.0)")
        print(f"💡 AI-CoScientist에 통합 추천!")

    # AI-CoScientist 통합 가능성
    all_working_models = working_gpt51 + working_gpt5

    if all_working_models:
        print(f"\n🔧 AI-CoScientist Enhanced Document Generator 통합 계획:")
        print(f"  1. 최고 성능 모델을 primary로 설정")
        print(f"  2. 기존 GPT-5와 함께 ensemble 구성")
        print(f"  3. 제안서 최적화에 특화 설정")
        return True
    else:
        print(f"\n📋 결론: 현재 GPT-5가 최적의 선택입니다.")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_gpt51())

    if result:
        print(f"\n✨ GPT-5.1 사용 가능! 시스템 업그레이드를 진행할 수 있습니다.")
    else:
        print(f"\n📋 GPT-5 + Gemini 2.5 Pro 조합이 현재 최적입니다.")