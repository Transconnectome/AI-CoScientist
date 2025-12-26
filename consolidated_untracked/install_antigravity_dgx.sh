#!/bin/bash
#
# Google Antigravity IDE 설치 스크립트
# Target: dgx-spark (Ubuntu 24.04 ARM64)
# Created: 2025-11-30
#

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Google Antigravity IDE 설치 스크립트 (dgx-spark)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 시스템 정보 확인
ARCH=$(uname -m)
OS=$(cat /etc/os-release | grep "^ID=" | cut -d= -f2)

echo "📋 시스템 정보:"
echo "   - Architecture: $ARCH"
echo "   - OS: $OS"
echo ""

if [[ "$ARCH" != "aarch64" && "$ARCH" != "x86_64" ]]; then
    echo "❌ 지원되지 않는 아키텍처: $ARCH"
    exit 1
fi

# 설치 방법 선택
echo "설치 방법을 선택하세요:"
echo "  1) APT 패키지 (sudo 필요, 자동 업데이트)"
echo "  2) tar.gz 직접 다운로드 (sudo 불필요)"
echo ""
read -p "선택 [1/2]: " INSTALL_METHOD

case $INSTALL_METHOD in
    1)
        echo ""
        echo "🔧 APT 패키지 설치 시작..."
        echo ""
        
        # Repository 키 추가
        echo ">>> GPG 키 추가 중..."
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://us-central1-apt.pkg.dev/doc/repo-signing-key.gpg | \
            sudo gpg --dearmor -o /etc/apt/keyrings/antigravity-repo-key.gpg
        
        # Repository 추가
        echo ">>> Repository 추가 중..."
        echo "deb [signed-by=/etc/apt/keyrings/antigravity-repo-key.gpg] https://us-central1-apt.pkg.dev/projects/antigravity-auto-updater-dev/ antigravity-debian main" | \
            sudo tee /etc/apt/sources.list.d/antigravity.list > /dev/null
        
        # 설치
        echo ">>> 패키지 설치 중..."
        sudo apt update
        sudo apt install -y antigravity
        
        echo ""
        echo "✅ APT 설치 완료!"
        echo "   실행: antigravity"
        ;;
        
    2)
        echo ""
        echo "🔧 tar.gz 직접 다운로드 설치 시작..."
        echo ""
        
        INSTALL_DIR="$HOME/.local/share/antigravity"
        BIN_DIR="$HOME/.local/bin"
        
        mkdir -p "$INSTALL_DIR"
        mkdir -p "$BIN_DIR"
        
        # 아키텍처별 다운로드 URL
        if [[ "$ARCH" == "aarch64" ]]; then
            DOWNLOAD_URL="https://antigravity.google/download/linux/antigravity-linux-arm64.tar.gz"
        else
            DOWNLOAD_URL="https://antigravity.google/download/linux/antigravity-linux-x64.tar.gz"
        fi
        
        echo ">>> 다운로드 중: $DOWNLOAD_URL"
        cd /tmp
        curl -L -o antigravity.tar.gz "$DOWNLOAD_URL"
        
        echo ">>> 압축 해제 중..."
        tar -xzf antigravity.tar.gz -C "$INSTALL_DIR" --strip-components=1
        rm antigravity.tar.gz
        
        # 심볼릭 링크 생성
        echo ">>> 심볼릭 링크 생성..."
        ln -sf "$INSTALL_DIR/antigravity" "$BIN_DIR/antigravity"
        
        # PATH 확인 및 추가
        if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
            echo ""
            echo "⚠️  PATH에 $BIN_DIR 추가 필요"
            echo "   다음 명령어를 ~/.bashrc에 추가하세요:"
            echo ""
            echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
            echo ""
            echo "   또는 지금 실행:"
            echo "   source ~/.bashrc"
        fi
        
        echo ""
        echo "✅ tar.gz 설치 완료!"
        echo "   설치 위치: $INSTALL_DIR"
        echo "   실행: $BIN_DIR/antigravity"
        ;;
        
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    설치 완료! 🚀                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  실행 방법:                                                   ║"
echo "║    antigravity                                                ║"
echo "║                                                               ║"
echo "║  참고:                                                        ║"
echo "║    - Google 계정 로그인 필요                                  ║"
echo "║    - Gemini 3 Pro 기능 사용 가능                              ║"
echo "║    - 문서: https://antigravity.codes/                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

