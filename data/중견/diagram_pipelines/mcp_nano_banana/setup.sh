#!/bin/bash
# Nano Banana MCP Server 설치 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "🍌 Nano Banana MCP Server 설치"
echo "=============================="

# Python 가상환경 생성
echo "📦 가상환경 생성 중..."
python3 -m venv "$VENV_DIR"

# 활성화 및 패키지 설치
echo "📥 패키지 설치 중..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "✅ 설치 완료!"
echo ""
echo "📝 다음 단계:"
echo "1. Claude Code 설정 파일에 MCP 서버 추가"
echo ""
echo "   ~/.claude/settings.json에 다음 추가:"
echo '   {
     "mcpServers": {
       "nano-banana": {
         "command": "'$VENV_DIR'/bin/python",
         "args": ["'$SCRIPT_DIR'/server.py"],
         "env": {
           "GOOGLE_API_KEY": "YOUR_GOOGLE_API_KEY"
         }
       }
     }
   }'
echo ""
echo "2. Claude Code 재시작"
echo ""
echo "🎨 사용 방법:"
echo "   - generate_diagram: 학술용 다이어그램 생성"
echo "   - generate_image: 일반 이미지 생성"
echo "   - list_generated_images: 생성된 이미지 목록"
