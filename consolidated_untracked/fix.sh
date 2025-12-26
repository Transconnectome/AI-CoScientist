#!/bin/bash

# Claude Code 모델 설정 수정 스크립트
# 잘못된 모델명(claude-sonnet-4-20250514)을 올바른 모델명으로 변경합니다.

echo "🔧 Claude Code 모델 설정 수정 스크립트"
echo "========================================"

# 사용할 모델 선택
echo ""
echo "사용할 모델을 선택하세요:"
echo "1) claude-sonnet-4-20250514 (Claude Sonnet 4)"
echo "2) claude-sonnet-4-20250514-20250514 (Claude Opus 4)"
echo "3) claude-sonnet-4-5-20250929 (Claude Sonnet 4.5)"
echo "4) claude-sonnet-4-20250514-5-20251101 (Claude Opus 4.5 - 최신)"
echo ""
read -p "선택 (1-4): " choice

case $choice in
    1) MODEL="claude-sonnet-4-20250514" ;;
    2) MODEL="claude-sonnet-4-20250514-20250514" ;;
    3) MODEL="claude-sonnet-4-5-20250929" ;;
    4) MODEL="claude-sonnet-4-20250514-5-20251101" ;;
    *) echo "❌ 잘못된 선택입니다."; exit 1 ;;
esac

echo ""
echo "선택한 모델: $MODEL"
echo ""

# 가능한 설정 파일 경로들
CONFIG_PATHS=(
    "$HOME/.claude.json"
    "$HOME/.config/claude/settings.json"
    "$HOME/.claude/settings.json"
    "$HOME/.config/claude-code/settings.json"
)

FOUND=false

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    if [ -f "$CONFIG_PATH" ]; then
        echo "📁 설정 파일 발견: $CONFIG_PATH"
        
        # 백업 생성
        cp "$CONFIG_PATH" "${CONFIG_PATH}.backup"
        echo "💾 백업 생성: ${CONFIG_PATH}.backup"
        
        # 모델명 수정 (여러 패턴 처리)
        sed -i.tmp 's/"model"[[:space:]]*:[[:space:]]*"claude-sonnet-4-20250514"/"model": "'"$MODEL"'"/g' "$CONFIG_PATH"
        sed -i.tmp 's/"model"[[:space:]]*:[[:space:]]*"claude-sonnet-4-20250514-[^"]*"/"model": "'"$MODEL"'"/g' "$CONFIG_PATH"
        rm -f "${CONFIG_PATH}.tmp"
        
        echo "✅ 모델 설정 변경 완료!"
        echo ""
        echo "📄 변경된 설정 파일 내용:"
        cat "$CONFIG_PATH"
        FOUND=true
        break
    fi
done

if [ "$FOUND" = false ]; then
    echo "⚠️  기존 설정 파일을 찾을 수 없습니다."
    echo ""
    echo "새 설정 파일을 생성합니다..."
    
    # 디렉토리 생성
    mkdir -p "$HOME/.claude"
    
    # 새 설정 파일 생성
    cat > "$HOME/.claude/settings.json" << EOF
{
  "model": "$MODEL"
}
EOF
    
    echo "✅ 새 설정 파일 생성: $HOME/.claude/settings.json"
    cat "$HOME/.claude/settings.json"
fi

echo ""
echo "========================================"
echo "🎉 완료! 이제 claude 명령어를 다시 실행해보세요."
echo ""
echo "💡 그래도 안 되면 다음 명령어로 직접 실행:"
echo "   claude --model $MODEL"
echo ""
echo "💡 또는 환경 변수 설정:"
echo "   export ANTHROPIC_MODEL=$MODEL"
