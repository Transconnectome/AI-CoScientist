#!/usr/bin/env python3
"""
Nano Banana Pro MCP Server
==========================
Claude Code에서 사용할 수 있는 이미지 생성 도구

모델:
- nano_banana: gemini-2.5-flash-image (빠름, 8초)
- nano_banana_pro: gemini-3-pro-image-preview (고품질, 17초)
"""

import os
import sys
import json
import time
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

# Google GenAI
from google import genai
from google.genai import types

# 설정
OUTPUT_DIR = Path.home() / "Desktop" / "nano_banana_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

server = Server("nano-banana")

def get_client():
    """Google GenAI 클라이언트 생성"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다")
    return genai.Client(api_key=api_key)


@server.list_tools()
async def list_tools():
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="generate_diagram",
            description="Nano Banana Pro를 사용하여 과학/기술 다이어그램 생성. 학술 논문용 고품질 다이어그램에 최적화됨.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "생성할 다이어그램에 대한 상세 설명 (영어 권장)"
                    },
                    "style": {
                        "type": "string",
                        "description": "스타일 지정 (예: clean, professional, colorful)",
                        "default": "clean, professional, white background"
                    },
                    "filename": {
                        "type": "string",
                        "description": "저장할 파일명 (확장자 제외)",
                        "default": "diagram"
                    }
                },
                "required": ["description"]
            }
        ),
        Tool(
            name="generate_image",
            description="Nano Banana Pro를 사용하여 일반 이미지 생성",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "생성할 이미지에 대한 프롬프트"
                    },
                    "model": {
                        "type": "string",
                        "description": "사용할 모델: nano_banana (빠름) 또는 nano_banana_pro (고품질)",
                        "enum": ["nano_banana", "nano_banana_pro"],
                        "default": "nano_banana_pro"
                    },
                    "filename": {
                        "type": "string",
                        "description": "저장할 파일명 (확장자 제외)",
                        "default": "image"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="list_generated_images",
            description="생성된 이미지 목록 조회",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "최대 개수",
                        "default": 10
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """도구 실행"""

    if name == "generate_diagram":
        return await generate_diagram(
            description=arguments["description"],
            style=arguments.get("style", "clean, professional, white background"),
            filename=arguments.get("filename", "diagram")
        )

    elif name == "generate_image":
        return await generate_image(
            prompt=arguments["prompt"],
            model=arguments.get("model", "nano_banana_pro"),
            filename=arguments.get("filename", "image")
        )

    elif name == "list_generated_images":
        return await list_generated_images(
            limit=arguments.get("limit", 10)
        )

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def generate_diagram(description: str, style: str, filename: str):
    """다이어그램 생성 (Nano Banana Pro 사용)"""

    # 다이어그램 최적화 프롬프트
    prompt = f"""Create a professional scientific diagram for a research paper.

Content: {description}

Style requirements:
- {style}
- Clear arrows showing data flow
- Readable text labels with proper spelling
- Well-organized layout
- High contrast for printing
"""

    try:
        client = get_client()
        start_time = time.time()

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE', 'TEXT']
            )
        )

        elapsed = time.time() - start_time

        # 이미지 추출 및 저장
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"{filename}_{timestamp}.png"

                with open(output_path, 'wb') as f:
                    f.write(part.inline_data.data)

                # 이미지를 base64로 인코딩하여 반환
                image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')

                return [
                    TextContent(
                        type="text",
                        text=f"✅ 다이어그램 생성 완료!\n📁 저장 위치: {output_path}\n⏱️ 소요 시간: {elapsed:.1f}초\n🎨 모델: Nano Banana Pro (gemini-3-pro-image-preview)"
                    ),
                    ImageContent(
                        type="image",
                        data=image_b64,
                        mimeType="image/png"
                    )
                ]

        # 텍스트만 반환된 경우
        text_response = response.text if hasattr(response, 'text') else str(response)
        return [TextContent(type="text", text=f"⚠️ 이미지 대신 텍스트 응답:\n{text_response}")]

    except Exception as e:
        return [TextContent(type="text", text=f"❌ 오류 발생: {str(e)}")]


async def generate_image(prompt: str, model: str, filename: str):
    """일반 이미지 생성"""

    model_map = {
        "nano_banana": "gemini-2.5-flash-image",
        "nano_banana_pro": "gemini-3-pro-image-preview"
    }
    model_id = model_map.get(model, "gemini-3-pro-image-preview")

    try:
        client = get_client()
        start_time = time.time()

        response = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE', 'TEXT']
            )
        )

        elapsed = time.time() - start_time

        # 이미지 추출 및 저장
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"{filename}_{timestamp}.png"

                with open(output_path, 'wb') as f:
                    f.write(part.inline_data.data)

                image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')

                return [
                    TextContent(
                        type="text",
                        text=f"✅ 이미지 생성 완료!\n📁 저장 위치: {output_path}\n⏱️ 소요 시간: {elapsed:.1f}초\n🎨 모델: {model} ({model_id})"
                    ),
                    ImageContent(
                        type="image",
                        data=image_b64,
                        mimeType="image/png"
                    )
                ]

        text_response = response.text if hasattr(response, 'text') else str(response)
        return [TextContent(type="text", text=f"⚠️ 이미지 대신 텍스트 응답:\n{text_response}")]

    except Exception as e:
        return [TextContent(type="text", text=f"❌ 오류 발생: {str(e)}")]


async def list_generated_images(limit: int):
    """생성된 이미지 목록"""

    images = sorted(OUTPUT_DIR.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]

    if not images:
        return [TextContent(type="text", text=f"📁 생성된 이미지가 없습니다.\n저장 위치: {OUTPUT_DIR}")]

    lines = [f"📁 생성된 이미지 목록 ({OUTPUT_DIR}):\n"]
    for img in images:
        mtime = datetime.fromtimestamp(img.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = img.stat().st_size / 1024
        lines.append(f"  • {img.name} ({size_kb:.1f} KB) - {mtime}")

    return [TextContent(type="text", text="\n".join(lines))]


async def main():
    """MCP 서버 실행"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
