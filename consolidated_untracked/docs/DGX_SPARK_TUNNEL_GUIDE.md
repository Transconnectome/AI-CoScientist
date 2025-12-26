# DGX Spark 원격 접속 가이드 (SSH + Tunnel + Antigravity)

**작성일**: 2025-11-30  
**대상 서버**: dgx-spark (spark-3a4c, Ubuntu 24.04 ARM64)  
**IP**: 192.168.0.79

---

## 📋 목차

1. [사전 준비](#1-사전-준비)
2. [SSH 기본 접속](#2-ssh-기본-접속)
3. [Tmux 세션 관리](#3-tmux-세션-관리)
4. [VS Code Tunnel 설정](#4-vs-code-tunnel-설정)
5. [Google Antigravity 실행](#5-google-antigravity-실행)
6. [문제 해결](#6-문제-해결)

---

## 1. 사전 준비

### 서버 정보

| 항목 | 값 |
|------|-----|
| 호스트명 | spark-3a4c |
| IP 주소 | 192.168.0.79 |
| 사용자 | juke |
| OS | Ubuntu 24.04.3 LTS |
| 아키텍처 | ARM64 (aarch64) |

### SSH Config 설정 (선택사항)

`~/.ssh/config` 파일에 추가:

```ssh-config
Host dgx-spark
    HostName 192.168.0.79
    User juke
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

설정 후 `ssh dgx-spark`로 간편 접속 가능.

---

## 2. SSH 기본 접속

### 직접 접속

```bash
ssh juke@192.168.0.79
```

### Config 사용 시

```bash
ssh dgx-spark
```

### X11 포워딩 (GUI 앱 실행 시)

```bash
ssh -X juke@192.168.0.79
```

---

## 3. Tmux 세션 관리

### 세션 목록 확인

```bash
ssh juke@192.168.0.79 "tmux list-sessions"
```

### 새 세션 생성

```bash
# 로컬에서 원격 세션 생성
ssh juke@192.168.0.79 "tmux new-session -d -s tunnel -c ~/git/AI-CoScientist"

# 세션에 접속
ssh -t juke@192.168.0.79 "tmux attach -t tunnel"
```

### 주요 Tmux 명령어

| 명령어 | 설명 |
|--------|------|
| `tmux new -s 이름` | 새 세션 생성 |
| `tmux attach -t 이름` | 세션 접속 |
| `tmux detach` 또는 `Ctrl+b d` | 세션에서 분리 |
| `tmux kill-session -t 이름` | 세션 종료 |
| `tmux ls` | 세션 목록 |

### 현재 활성 세션들

```
agentreview  - AgentReview 3.0 작업
antigravity  - Antigravity IDE 전용
dd           - Agent Pool 2.0 작업
main         - CENTaUR 환경
tunnel       - VS Code Tunnel 전용
workspace    - RAG 시스템 작업
```

---

## 4. VS Code Tunnel 설정

### 4.1 Tunnel CLI 설치 (최초 1회)

```bash
ssh juke@192.168.0.79 "
mkdir -p ~/.local/bin
curl -L -o /tmp/code-cli.tar.gz 'https://update.code.visualstudio.com/latest/cli-alpine-arm64/stable'
tar -xzf /tmp/code-cli.tar.gz -C ~/.local/bin
mv ~/.local/bin/code ~/.local/bin/code-tunnel
chmod +x ~/.local/bin/code-tunnel
"
```

### 4.2 Tunnel 세션 생성 및 시작

```bash
# tunnel 전용 tmux 세션 생성
ssh juke@192.168.0.79 "tmux new-session -d -s tunnel -c ~/git/AI-CoScientist"

# 터널 시작
ssh juke@192.168.0.79 "tmux send-keys -t tunnel '~/.local/bin/code-tunnel tunnel --accept-server-license-terms' Enter"
```

### 4.3 GitHub 인증 (최초 1회)

터널 출력 확인:

```bash
ssh juke@192.168.0.79 "tmux capture-pane -t tunnel -p | tail -20"
```

출력에서 인증 코드 확인 후:
1. 브라우저에서 https://github.com/login/device 접속
2. 표시된 코드 입력 (예: `BB63-83D1`)
3. "Authorize" 클릭

### 4.4 머신 이름 설정

인증 완료 후 머신 이름 묻는 프롬프트에서 Enter (기본값 `spark-3a4c` 사용)

```bash
ssh juke@192.168.0.79 "tmux send-keys -t tunnel Enter"
```

### 4.5 터널 URL 확인

```bash
ssh juke@192.168.0.79 "tmux capture-pane -t tunnel -p | grep 'vscode.dev'"
```

**접속 URL**: https://vscode.dev/tunnel/spark-3a4c

### 4.6 터널 관리 명령어

```bash
# 상태 확인
ssh juke@192.168.0.79 "~/.local/bin/code-tunnel tunnel status"

# 터널 중지
ssh juke@192.168.0.79 "~/.local/bin/code-tunnel tunnel kill"

# 터널 재시작
ssh juke@192.168.0.79 "~/.local/bin/code-tunnel tunnel restart"
```

### 4.7 원라이너: 터널 빠른 시작

```bash
ssh juke@192.168.0.79 "
tmux kill-session -t tunnel 2>/dev/null || true
tmux new-session -d -s tunnel -c ~/git/AI-CoScientist
tmux send-keys -t tunnel '~/.local/bin/code-tunnel tunnel --accept-server-license-terms' Enter
echo 'Tunnel started in tmux session: tunnel'
"
```

---

## 5. Google Antigravity 실행

### 5.1 설치 확인

```bash
ssh juke@192.168.0.79 "antigravity --version"
# 출력: 1.104.0
```

### 5.2 실행 방법

#### 방법 A: X11 포워딩 (GUI)

```bash
ssh -X juke@192.168.0.79 "antigravity"
```

#### 방법 B: Tmux 세션에서 실행

```bash
# antigravity 세션 생성
ssh juke@192.168.0.79 "tmux new-session -d -s antigravity -c ~/git/AI-CoScientist"

# 접속
ssh -t juke@192.168.0.79 "tmux attach -t antigravity"

# 세션 내에서 실행
antigravity
```

#### 방법 C: VS Code Tunnel 사용 (권장)

VS Code Tunnel이 실행 중이면 브라우저에서:
- https://vscode.dev/tunnel/spark-3a4c

### 5.3 Antigravity 주요 기능

- **Gemini 3 Pro** 기반 AI 코딩 에이전트
- 자율적 코드 작성, 터미널 실행, 브라우저 테스트
- 다중 에이전트 협업 지원

---

## 6. 문제 해결

### SSH 접속 실패

```bash
# 호스트 키 문제 시
ssh -o StrictHostKeyChecking=no juke@192.168.0.79

# known_hosts에서 기존 키 제거
ssh-keygen -R 192.168.0.79
```

### Tunnel이 시작되지 않음

```bash
# 기존 터널 프로세스 강제 종료
ssh juke@192.168.0.79 "pkill -f code-tunnel; ~/.local/bin/code-tunnel tunnel kill"

# 다시 시작
ssh juke@192.168.0.79 "tmux send-keys -t tunnel '~/.local/bin/code-tunnel tunnel' Enter"
```

### Tmux 세션 깨짐

```bash
# 세션 강제 종료 후 재생성
ssh juke@192.168.0.79 "tmux kill-session -t tunnel; tmux new-session -d -s tunnel"
```

### Antigravity tunnel 명령어 오류

Antigravity 자체 tunnel은 바이너리 누락 문제가 있음.  
**VS Code Tunnel (`code-tunnel`)을 대신 사용**:

```bash
~/.local/bin/code-tunnel tunnel --accept-server-license-terms
```

---

## 📎 빠른 참조

### 전체 프로세스 원라이너

```bash
# 1. Tunnel 세션 생성 및 시작
ssh juke@192.168.0.79 "
tmux kill-session -t tunnel 2>/dev/null
tmux new-session -d -s tunnel -c ~/git/AI-CoScientist
tmux send-keys -t tunnel '~/.local/bin/code-tunnel tunnel --accept-server-license-terms' Enter
"

# 2. 인증 코드 확인 (최초 1회)
ssh juke@192.168.0.79 "sleep 5; tmux capture-pane -t tunnel -p | grep -E 'code|vscode.dev'"

# 3. 브라우저에서 접속
# https://vscode.dev/tunnel/spark-3a4c
```

### 상태 확인 스크립트

```bash
ssh juke@192.168.0.79 "
echo '=== Tmux Sessions ==='
tmux ls
echo ''
echo '=== Tunnel Status ==='
~/.local/bin/code-tunnel tunnel status 2>/dev/null || echo 'Not running'
"
```

---

## 📚 참고 링크

- [VS Code Remote Tunnels](https://code.visualstudio.com/docs/remote/tunnels)
- [Google Antigravity](https://antigravity.google/)
- [Tmux Cheat Sheet](https://tmuxcheatsheet.com/)

