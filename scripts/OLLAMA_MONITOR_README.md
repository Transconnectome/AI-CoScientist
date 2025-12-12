# Ollama Download Monitor

자동으로 Ollama 모델 다운로드를 모니터링하고 완료 시 이메일로 알려주는 스크립트입니다.

## 기능

- ✅ DGX 서버의 Ollama 다운로드 상태 자동 모니터링
- 📧 다운로드 완료 시 이메일 알림
- ⏱️  다운로드 시간 및 모델 크기 추적
- 🔄 백그라운드 실행 가능

## 기본 사용법

### 1. 이메일 알림 없이 모니터링만

```bash
# 30초마다 체크 (기본값)
python scripts/monitor_ollama_download.py deepseek-r1:32b

# 10초마다 체크
python scripts/monitor_ollama_download.py deepseek-r1:32b --interval 10
```

### 2. 이메일 알림과 함께 모니터링

```bash
python scripts/monitor_ollama_download.py deepseek-r1:32b \
  --email-to your.email@gmail.com \
  --smtp-server smtp.gmail.com \
  --smtp-user your.email@gmail.com \
  --smtp-password "your-app-password"
```

### 3. 백그라운드에서 실행

```bash
# nohup으로 백그라운드 실행
nohup python scripts/monitor_ollama_download.py deepseek-r1:32b \
  --email-to your.email@gmail.com \
  --smtp-server smtp.gmail.com \
  --smtp-user your.email@gmail.com \
  --smtp-password "your-app-password" \
  > /tmp/ollama_monitor.log 2>&1 &

# 프로세스 확인
ps aux | grep monitor_ollama

# 로그 확인
tail -f /tmp/ollama_monitor.log
```

## Gmail 설정 방법

Gmail을 사용하는 경우, 2단계 인증을 활성화하고 앱 비밀번호를 생성해야 합니다:

1. Google 계정 설정 → 보안 → 2단계 인증 활성화
2. 앱 비밀번호 생성: https://myaccount.google.com/apppasswords
3. 생성된 16자리 비밀번호를 `--smtp-password`에 사용

```bash
# Gmail 사용 예시
python scripts/monitor_ollama_download.py deepseek-r1:32b \
  --email-to jiook.cha@gmail.com \
  --smtp-server smtp.gmail.com \
  --smtp-user jiook.cha@gmail.com \
  --smtp-password "xxxx xxxx xxxx xxxx"
```

## 환경 변수로 설정

매번 SMTP 정보를 입력하기 번거롭다면 환경 변수로 설정:

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export OLLAMA_MONITOR_EMAIL="your.email@gmail.com"
export OLLAMA_MONITOR_SMTP_SERVER="smtp.gmail.com"
export OLLAMA_MONITOR_SMTP_USER="your.email@gmail.com"
export OLLAMA_MONITOR_SMTP_PASSWORD="your-app-password"

# 환경 변수 사용하는 wrapper 스크립트
cat > ~/monitor_ollama.sh << 'EOF'
#!/bin/bash
python /Users/jiookcha/Documents/git/AI-CoScientist/scripts/monitor_ollama_download.py "$1" \
  --email-to "$OLLAMA_MONITOR_EMAIL" \
  --smtp-server "$OLLAMA_MONITOR_SMTP_SERVER" \
  --smtp-user "$OLLAMA_MONITOR_SMTP_USER" \
  --smtp-password "$OLLAMA_MONITOR_SMTP_PASSWORD" \
  "${@:2}"
EOF

chmod +x ~/monitor_ollama.sh

# 간단하게 사용
~/monitor_ollama.sh deepseek-r1:32b
```

## 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--server` | SSH 서버 호스트명 | dgx-spark |
| `--interval` | 체크 간격 (초) | 30 |
| `--email-to` | 수신 이메일 | None |
| `--email-from` | 발신 이메일 | --email-to와 동일 |
| `--smtp-server` | SMTP 서버 | None |
| `--smtp-port` | SMTP 포트 | 587 |
| `--smtp-user` | SMTP 사용자명 | None |
| `--smtp-password` | SMTP 비밀번호 | None |
| `--max-checks` | 최대 체크 횟수 | 무제한 |

## 현재 다운로드 확인

```bash
# 수동으로 상태 확인
ssh dgx-spark "curl -s http://localhost:11434/api/tags | jq '.models[] | select(.name==\"deepseek-r1:32b\")'"

# 모델 목록 확인
ssh dgx-spark "curl -s http://localhost:11434/api/tags | jq -r '.models[] | .name'"
```

## 예제

### 현재 DeepSeek-R1 다운로드 모니터링

```bash
# 이메일 알림과 함께 모니터링 시작
python scripts/monitor_ollama_download.py deepseek-r1:32b \
  --email-to jiook.cha@gmail.com \
  --smtp-server smtp.gmail.com \
  --smtp-user jiook.cha@gmail.com \
  --smtp-password "your-gmail-app-password" \
  --interval 60
```

출력 예시:
```
🔍 Starting monitor for model: deepseek-r1:32b
📡 Server: dgx-spark
⏱️  Check interval: 60s
📧 Email notifications: jiook.cha@gmail.com
------------------------------------------------------------
⏳ Check #1: Model not ready yet (elapsed: 0s)
⏳ Check #2: Model not ready yet (elapsed: 1m)
⏳ Check #3: Model not ready yet (elapsed: 2m)
...
✅ Download complete!
📦 Model: deepseek-r1:32b
💾 Size: 18.95 GB
⏱️  Duration: 5m 30s
✅ Email sent to jiook.cha@gmail.com
============================================================
🏁 Monitoring complete
```

## 문제 해결

### SSH 연결 오류
```bash
# SSH 키가 설정되어 있는지 확인
ssh dgx-spark "echo Connection OK"

# SSH 설정 파일 확인
cat ~/.ssh/config
```

### 이메일 발송 실패
```bash
# SMTP 연결 테스트
python -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your.email@gmail.com', 'your-app-password')
print('✅ SMTP connection successful')
server.quit()
"
```

### 권한 오류
```bash
# 스크립트 실행 권한 확인
ls -la scripts/monitor_ollama_download.py

# 실행 권한 부여
chmod +x scripts/monitor_ollama_download.py
```

## 기술 세부사항

- **모니터링 방식**: SSH를 통해 Ollama API (`/api/tags`)를 주기적으로 호출
- **완료 판정**: 모델이 `/api/tags` 목록에 나타나면 다운로드 완료로 간주
- **이메일**: SMTP/TLS를 사용하여 안전하게 전송
- **에러 처리**: 네트워크 오류, API 오류 등을 자동으로 감지하고 재시도

## 개선 아이디어

향후 추가 가능한 기능:
- Slack, Discord 알림 지원
- 다운로드 진행률 표시 (가능한 경우)
- 웹 대시보드
- 여러 모델 동시 모니터링
- systemd 서비스로 등록

## 라이센스

MIT License - AI-CoScientist 프로젝트의 일부
