# Cursor 설치 가이드 - DGX Spark (ARM64)

**시스템**: DGX Spark (spark-3a4c)  
**아키텍처**: ARM64 (aarch64)  
**OS**: Ubuntu (DGX OS)  
**현재 상태**: Cursor Remote Server 2.1.39 설치됨 ✅

---

## 현재 상태

### ✅ 이미 설치된 것
- **Cursor Remote Server**: 버전 2.1.39
- **위치**: `~/.cursor-server/`
- **상태**: 실행 중 (Remote SSH를 통해 접속 가능)

### 확인된 정보
```bash
$ cursor --version
2.1.39
60d42bed27e5775c43ec0428d8c653c49e260
arm64
```

---

## 설치 옵션

### 옵션 1: Remote Server 사용 (현재 상태) ✅

**장점**:
- 이미 설치되어 있음
- 로컬 Cursor 클라이언트에서 SSH로 연결 가능
- 리소스 효율적

**사용 방법**:
1. 로컬 컴퓨터에 Cursor 설치
2. Cursor에서 `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
3. `juke@spark-3a4c` 또는 IP 주소 입력
4. 연결 완료

### 옵션 2: 데스크톱 버전 설치 (GUI 필요)

**전제 조건**:
- X11 또는 Wayland 디스플레이 서버
- GUI 환경 (GNOME, KDE 등)

**ARM64용 설치 방법**:

#### 방법 A: AppImage (권장)

```bash
# 1. 다운로드 (ARM64용)
cd /tmp
wget https://downloader.cursor.sh/linux/arm64 -O cursor.AppImage

# 2. 실행 권한 부여
chmod +x cursor.AppImage

# 3. 실행
./cursor.AppImage

# 또는 특정 위치에 설치
mkdir -p ~/Applications
mv cursor.AppImage ~/Applications/
~/Applications/cursor.AppImage
```

#### 방법 B: .deb 패키지 (가능한 경우)

```bash
# 1. 다운로드
wget https://downloader.cursor.sh/linux/arm64.deb -O cursor.deb

# 2. 설치
sudo dpkg -i cursor.deb
sudo apt-get install -f  # 의존성 해결

# 3. 실행
cursor
```

#### 방법 C: Snap (가능한 경우)

```bash
sudo snap install cursor --classic
```

---

## 현재 시스템 확인

### 디스플레이 확인
```bash
echo $DISPLAY
# 비어있으면 GUI 환경 없음
```

### X11 서버 확인
```bash
which Xorg
xhost
```

### Wayland 확인
```bash
echo $WAYLAND_DISPLAY
```

---

## 권장 방법

### 시나리오 1: SSH로 접속하는 경우
→ **Remote Server 사용** (현재 상태 유지)
- 로컬 Cursor에서 SSH 연결
- 가장 효율적이고 안정적

### 시나리오 2: 직접 GUI 접속하는 경우
→ **AppImage 설치**
```bash
# 다운로드 및 실행
cd ~/Downloads
wget https://downloader.cursor.sh/linux/arm64 -O cursor.AppImage
chmod +x cursor.AppImage
./cursor.AppImage
```

### 시나리오 3: 시스템 패키지로 설치
→ **공식 저장소 확인 필요**
- Cursor가 ARM64용 .deb 패키지를 제공하는지 확인
- 제공하지 않으면 AppImage 사용

---

## 설치 스크립트

### 자동 설치 스크립트

```bash
#!/bin/bash
# cursor_install.sh

set -e

ARCH=$(dpkg --print-architecture)
INSTALL_DIR="$HOME/Applications"

echo "Installing Cursor for $ARCH..."

# AppImage 다운로드
cd /tmp
if [ "$ARCH" = "arm64" ]; then
    URL="https://downloader.cursor.sh/linux/arm64"
else
    URL="https://downloader.cursor.sh/linux/x86_64"
fi

echo "Downloading from $URL..."
wget "$URL" -O cursor.AppImage

# 실행 권한 부여
chmod +x cursor.AppImage

# 설치 디렉토리 생성
mkdir -p "$INSTALL_DIR"

# 이동
mv cursor.AppImage "$INSTALL_DIR/cursor.AppImage"

# 심볼릭 링크 생성
ln -sf "$INSTALL_DIR/cursor.AppImage" ~/.local/bin/cursor

echo "✅ Cursor installed to $INSTALL_DIR/cursor.AppImage"
echo "Run with: ~/Applications/cursor.AppImage"
echo "Or: cursor (if ~/.local/bin is in PATH)"
```

**실행**:
```bash
chmod +x cursor_install.sh
./cursor_install.sh
```

---

## 문제 해결

### 문제 1: DISPLAY 환경 변수 없음
```bash
# X11 포워딩 활성화 (SSH 연결 시)
ssh -X juke@spark-3a4c

# 또는 X11 소켓 마운트
export DISPLAY=:0
```

### 문제 2: 권한 문제
```bash
# AppImage 실행 권한 확인
chmod +x cursor.AppImage

# 또는 FUSE 필요
sudo apt-get install fuse
```

### 문제 3: ARM64 지원 확인
```bash
# 현재 아키텍처 확인
uname -m
# aarch64 = ARM64

# Cursor가 ARM64를 지원하는지 확인
curl -I https://downloader.cursor.sh/linux/arm64
```

---

## 현재 Remote Server 활용

### Remote Server는 이미 작동 중:
- 프로세스 확인: `ps aux | grep cursor-server`
- 연결: 로컬 Cursor에서 SSH로 연결
- 포트: 자동 할당 (로컬호스트)

### Remote Server 업데이트:
```bash
# Cursor 클라이언트가 자동으로 업데이트
# 또는 수동으로 재설치
rm -rf ~/.cursor-server
# 다음 SSH 연결 시 자동 재설치
```

---

## 결론

**현재 상태**: Cursor Remote Server가 이미 설치되어 작동 중입니다.

**추가 설치 필요 시**:
1. GUI 환경이 있다면 → AppImage 설치
2. SSH 접속만 사용한다면 → 현재 상태 유지 (Remote Server)

**권장**: Remote Server 사용 (이미 작동 중, 효율적)

---

**참고**: Cursor 공식 사이트에서 최신 ARM64 다운로드 링크 확인
- https://cursor.sh/
- https://downloader.cursor.sh/




