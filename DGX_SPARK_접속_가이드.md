# DGX Spark 접속 가이드

**서버 정보**:
- 호스트명: `spark-3a4c`
- IP 주소: `192.168.0.79` (로컬 네트워크)
- 사용자: `juke`
- SSH 포트: `22`

---

## 방법 1: 기본 SSH 접속

### Windows (PowerShell 또는 CMD)

```powershell
# 기본 접속
ssh juke@192.168.0.79

# 또는 호스트명 사용 (같은 네트워크에서)
ssh juke@spark-3a4c
```

### macOS / Linux

```bash
# 기본 접속
ssh juke@192.168.0.79

# 또는 호스트명 사용
ssh juke@spark-3a4c
```

---

## 방법 2: SSH 키 설정 (권장)

### 1단계: SSH 키 생성 (로컬 컴퓨터에서)

```bash
# SSH 키가 없으면 생성
ssh-keygen -t ed25519 -C "your_email@example.com"

# 또는 RSA 키
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

### 2단계: 공개키 복사

```bash
# 방법 A: ssh-copy-id 사용 (가장 쉬움)
ssh-copy-id juke@192.168.0.79

# 방법 B: 수동 복사
cat ~/.ssh/id_ed25519.pub | ssh juke@192.168.0.79 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### 3단계: 비밀번호 없이 접속

```bash
ssh juke@192.168.0.79
# 이제 비밀번호 없이 접속됩니다!
```

---

## 방법 3: SSH Config 설정 (편리함)

### 로컬 컴퓨터의 `~/.ssh/config` 파일 편집

```bash
# macOS/Linux
nano ~/.ssh/config

# Windows (Git Bash 또는 WSL)
notepad ~/.ssh/config
```

### Config 내용 추가

```ssh-config
Host dgx-spark
    HostName 192.168.0.79
    User juke
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3

# 또는 호스트명 사용 (같은 네트워크)
Host spark
    HostName spark-3a4c
    User juke
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

### 사용 방법

```bash
# 이제 간단하게 접속 가능
ssh dgx-spark
# 또는
ssh spark
```

---

## 방법 4: Cursor에서 Remote SSH 연결

### 1단계: Cursor 설치 (로컬 컴퓨터)

- https://cursor.sh/ 에서 다운로드
- Windows/macOS/Linux 버전 설치

### 2단계: Remote SSH 확장 설치

1. Cursor 열기
2. `Ctrl+Shift+X` (또는 `Cmd+Shift+X` on Mac) - 확장 마켓플레이스
3. "Remote - SSH" 검색 및 설치

### 3단계: 서버 연결

1. `Ctrl+Shift+P` (또는 `Cmd+Shift+P` on Mac)
2. "Remote-SSH: Connect to Host" 입력
3. 다음 중 선택:
   - `juke@192.168.0.79` (직접 입력)
   - `dgx-spark` (config 설정한 경우)
   - `spark` (config 설정한 경우)

### 4단계: 연결 완료

- 비밀번호 입력 (또는 SSH 키 사용)
- 연결 후 Cursor가 서버에 접속됨
- 파일 탐색기에서 서버 파일 접근 가능

---

## 방법 5: VS Code에서 접속

### VS Code Remote SSH 사용

1. VS Code 설치
2. "Remote - SSH" 확장 설치
3. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
4. `juke@192.168.0.79` 입력
5. 연결 완료

---

## 네트워크 설정

### 같은 네트워크에 있는 경우

```bash
# 직접 IP 또는 호스트명 사용
ssh juke@192.168.0.79
ssh juke@spark-3a4c
```

### 다른 네트워크에서 접속 (VPN 필요)

```bash
# VPN 연결 후
ssh juke@192.168.0.79

# 또는 Tailscale 사용 (이미 설정되어 있음)
ssh juke@<tailscale-ip>
```

### 포트 포워딩 (필요한 경우)

```bash
# SSH 포트 포워딩
ssh -L 8080:localhost:8080 juke@192.168.0.79

# Jupyter 등 다른 서비스 포트 포워딩
ssh -L 8888:localhost:8888 juke@192.168.0.79
```

---

## 접속 확인

### 서버 정보 확인

```bash
# 접속 후
hostname
# 출력: spark-3a4c

whoami
# 출력: juke

pwd
# 출력: /home/juke
```

### tmux 세션 확인

```bash
# 접속 후
tmux ls

# 세션 연결
tmux attach -t workspace
```

---

## 문제 해결

### 문제 1: "Connection refused"

```bash
# SSH 서비스 확인
# 서버에서:
sudo systemctl status ssh
sudo systemctl start ssh
```

### 문제 2: "Host key verification failed"

```bash
# 로컬에서 known_hosts에서 제거
ssh-keygen -R 192.168.0.79
ssh-keygen -R spark-3a4c
```

### 문제 3: "Permission denied"

```bash
# SSH 키 권한 확인 (서버에서)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 문제 4: 네트워크 연결 안 됨

```bash
# ping 테스트
ping 192.168.0.79

# 포트 확인
telnet 192.168.0.79 22
# 또는
nc -zv 192.168.0.79 22
```

---

## 보안 설정 (선택사항)

### SSH 포트 변경

```bash
# 서버에서 /etc/ssh/sshd_config 편집
sudo nano /etc/ssh/sshd_config

# Port 22를 다른 포트로 변경 (예: 2222)
Port 2222

# SSH 재시작
sudo systemctl restart sshd
```

### 접속 시 포트 지정

```bash
ssh -p 2222 juke@192.168.0.79
```

---

## 빠른 참조

### 기본 접속 명령어

```bash
# IP 주소 사용
ssh juke@192.168.0.79

# 호스트명 사용 (같은 네트워크)
ssh juke@spark-3a4c

# 포트 지정
ssh -p 22 juke@192.168.0.79

# X11 포워딩 (GUI 앱 실행 시)
ssh -X juke@192.168.0.79

# 포트 포워딩
ssh -L 8080:localhost:8080 juke@192.168.0.79
```

### SSH Config 예시

```ssh-config
Host dgx-spark
    HostName 192.168.0.79
    User juke
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ForwardX11 yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    LocalForward 8080 localhost:8080
```

---

## Cursor Remote SSH 설정 완료 후

1. **파일 탐색**: 왼쪽 파일 탐색기에서 서버 파일 접근
2. **터미널**: `Ctrl+`` (백틱)으로 통합 터미널 열기
3. **확장 설치**: 서버에 확장 설치 가능
4. **Git 작업**: 서버의 Git 저장소 직접 작업

---

## 요약

**가장 간단한 방법**:
```bash
ssh juke@192.168.0.79
```

**가장 편리한 방법**:
1. SSH config 설정 (`~/.ssh/config`)
2. `ssh dgx-spark`로 접속

**Cursor 사용 시**:
1. Remote-SSH 확장 설치
2. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
3. `juke@192.168.0.79` 입력

---

**서버 정보**:
- IP: `192.168.0.79`
- 호스트명: `spark-3a4c`
- 사용자: `juke`
- 포트: `22`




