#!/bin/bash

# Claude Code 모델 설정 진단 및 수정 스크립트 v2
# 모든 가능한 위치에서 잘못된 모델명을 찾아 수정합니다.

echo "🔍 Claude Code 모델 설정 진단 스크립트 v2"
echo "==========================================="
echo ""

# 수정할 모델명
NEW_MODEL="claude-sonnet-4-20250514"

echo "🎯 목표: 'claude-sonnet-4-20250514' → '$NEW_MODEL' 로 변경"
echo ""

# 1. 잘못된 모델명이 있는 모든 파일 찾기
echo "📂 [1단계] 'claude-sonnet-4-20250514' 설정이 있는 파일 검색 중..."
echo ""

FOUND_FILES=$(grep -rl "claude-sonnet-4-20250514" ~ 2>/dev/null)

if [ -z "$FOUND_FILES" ]; then
    echo "⚠️  홈 디렉토리에서 찾을 수 없음. 시스템 전체 검색..."
    FOUND_FILES=$(sudo grep -rl "claude-sonnet-4-20250514" /etc /usr/local 2>/dev/null)
fi

if [ -z "$FOUND_FILES" ]; then
    echo "❌ 'claude-sonnet-4-20250514' 설정을 찾을 수 없습니다."
    echo ""
    echo "📋 다른 방법을 시도합니다..."
else
    echo "✅ 발견된 파일들:"
    echo "$FOUND_FILES"
    echo ""
    
    # 각 파일 수정
    for FILE in $FOUND_FILES; do
        echo "🔧 수정 중: $FILE"
        cp "$FILE" "${FILE}.backup.$(date +%Y%m%d%H%M%S)"
        sed -i 's/claude-sonnet-4-20250514/'"$NEW_MODEL"'/g' "$FILE"
        echo "   ✅ 완료 (백업 생성됨)"
    done
    echo ""
fi

# 2. Claude Code 설정 디렉토리 강제 생성 및 설정
echo "📂 [2단계] Claude Code 설정 강제 적용..."
echo ""

# 가능한 모든 설정 위치에 설정 파일 생성/수정
CONFIG_DIRS=(
    "$HOME/.claude"
    "$HOME/.config/claude"
    "$HOME/.config/claude-code"
    "$HOME/.claude-code"
)

for DIR in "${CONFIG_DIRS[@]}"; do
    mkdir -p "$DIR" 2>/dev/null
    CONFIG_FILE="$DIR/settings.json"
    
    echo '{"model": "'"$NEW_MODEL"'"}' > "$CONFIG_FILE"
    echo "   📄 생성: $CONFIG_FILE"
done

# .claude.json 파일도 생성
echo '{"model": "'"$NEW_MODEL"'"}' > "$HOME/.claude.json"
echo "   📄 생성: $HOME/.claude.json"

echo ""

# 3. 환경 변수 설정
echo "📂 [3단계] 환경 변수 설정..."
echo ""

# 현재 세션에 적용
export ANTHROPIC_MODEL="$NEW_MODEL"
export CLAUDE_MODEL="$NEW_MODEL"

# .bashrc에 추가
if ! grep -q "ANTHROPIC_MODEL" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# Claude Code 모델 설정" >> "$HOME/.bashrc"
    echo "export ANTHROPIC_MODEL=\"$NEW_MODEL\"" >> "$HOME/.bashrc"
    echo "export CLAUDE_MODEL=\"$NEW_MODEL\"" >> "$HOME/.bashrc"
    echo "   ✅ .bashrc에 환경 변수 추가됨"
fi

# .zshrc에 추가 (zsh 사용자용)
if [ -f "$HOME/.zshrc" ]; then
    if ! grep -q "ANTHROPIC_MODEL" "$HOME/.zshrc" 2>/dev/null; then
        echo "" >> "$HOME/.zshrc"
        echo "# Claude Code 모델 설정" >> "$HOME/.zshrc"
        echo "export ANTHROPIC_MODEL=\"$NEW_MODEL\"" >> "$HOME/.zshrc"
        echo "export CLAUDE_MODEL=\"$NEW_MODEL\"" >> "$HOME/.zshrc"
        echo "   ✅ .zshrc에 환경 변수 추가됨"
    fi
fi

echo ""

# 4. npm global 설정 확인 (npm으로 설치한 경우)
echo "📂 [4단계] npm 글로벌 설정 확인..."
echo ""

NPM_GLOBAL=$(npm root -g 2>/dev/null)
if [ -n "$NPM_GLOBAL" ]; then
    CLAUDE_NPM="$NPM_GLOBAL/@anthropic-ai/claude-code"
    if [ -d "$CLAUDE_NPM" ]; then
        echo "   📦 npm 패키지 발견: $CLAUDE_NPM"
        FOUND_IN_NPM=$(grep -rl "claude-sonnet-4-20250514" "$CLAUDE_NPM" 2>/dev/null)
        if [ -n "$FOUND_IN_NPM" ]; then
            for FILE in $FOUND_IN_NPM; do
                echo "   🔧 수정 중: $FILE"
                sudo sed -i 's/claude-sonnet-4-20250514/'"$NEW_MODEL"'/g' "$FILE" 2>/dev/null || \
                sed -i 's/claude-sonnet-4-20250514/'"$NEW_MODEL"'/g' "$FILE"
            done
        fi
    fi
fi

echo ""
echo "==========================================="
echo "🎉 설정 완료!"
echo ""
echo "📌 다음 단계:"
echo "   1. 새 터미널을 열거나 다음 명령어 실행:"
echo "      source ~/.bashrc  (또는 source ~/.zshrc)"
echo ""
echo "   2. Claude Code 다시 실행:"
echo "      claude"
echo ""
echo "   3. 그래도 안 되면 모델을 직접 지정:"
echo "      claude --model $NEW_MODEL"
echo ""
echo "==========================================="
echo ""
echo "📋 현재 환경 변수 상태:"
echo "   ANTHROPIC_MODEL=$ANTHROPIC_MODEL"
echo "   CLAUDE_MODEL=$CLAUDE_MODEL"
