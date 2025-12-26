# 실시간 회의 음성 분석 시스템

## 📋 개요

Microsoft Teams, Zoom 등의 화상 회의 플랫폼에서 실시간으로 각 참여자의 음성을 분석하여:
- 화자별 회의 노트 자동 생성
- 발언 시간 분석 및 통계
- 회의 내용 요약 및 액션 아이템 추출

을 수행하는 시스템 구축 가이드 및 프로토타입입니다.

## 📚 문서

1. **[상세 기술 문서](./meeting_voice_analysis_system.md)** (영문)
   - 전체 시스템 아키텍처
   - 단계별 구현 가이드
   - 코드 예제 포함

2. **[한국어 요약](./meeting_voice_analysis_system_kr.md)**
   - 핵심 내용 요약
   - 빠른 참조용

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install -r scripts/requirements_meeting_analysis.txt
```

### 2. 프로토타입 실행

```bash
# 기본 사용 (Whisper만)
python scripts/meeting_analysis_prototype.py audio.wav

# 큰 모델 사용
python scripts/meeting_analysis_prototype.py audio.wav --whisper-model large

# 화자 분리 포함 (Hugging Face 토큰 필요)
python scripts/meeting_analysis_prototype.py audio.wav \
    --pyannote-token YOUR_HUGGINGFACE_TOKEN

# 결과 저장
python scripts/meeting_analysis_prototype.py audio.wav \
    --output results.json
```

### 3. Hugging Face 토큰 발급 (pyannote.audio 사용 시)

1. [Hugging Face](https://huggingface.co/) 계정 생성
2. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 접근 권한 요청
3. Settings > Access Tokens에서 토큰 생성

## 📊 기존 솔루션 비교

| 솔루션 | 실시간 분석 | 화자별 노트 | 발언 시간 | 커스터마이징 | 비용 |
|--------|-----------|------------|----------|------------|------|
| **Fireflies.ai** | ✅ | ✅ | ✅ | ❌ | 유료 |
| **Otter.ai** | ✅ | ✅ | ⚠️ | ❌ | 유료 |
| **Teams 내장** | ⚠️ | ❌ | ❌ | ❌ | 일부 플랜 |
| **Zoom 내장** | ⚠️ | ⚠️ | ❌ | ❌ | 일부 플랜 |
| **커스텀 개발** | ✅ | ✅ | ✅ | ✅ | 개발 비용 |

## 🏗️ 시스템 구성 요소

### 핵심 기술 스택

1. **오디오 수집**
   - Microsoft Graph API (Teams)
   - Zoom SDK/API
   - Webhook / Real-time Stream

2. **화자 분리 (Speaker Diarization)**
   - pyannote.audio (권장)
   - Resemblyzer (경량)

3. **음성 인식 (ASR)**
   - OpenAI Whisper (오프라인)
   - Azure Speech Service (실시간, 권장)
   - Google Cloud Speech-to-Text

4. **자연어 처리**
   - OpenAI GPT-4 (요약)
   - LangChain (체인 처리)
   - KeyBERT (키워드)

5. **백엔드**
   - FastAPI
   - WebSocket
   - PostgreSQL / MongoDB

## 📈 개발 일정

| Phase | 기간 | 난이도 |
|-------|------|--------|
| 인프라 구축 | 1-2주 | ⭐⭐ |
| 오디오 수집 | 2-3주 | ⭐⭐ |
| 화자 분리 | 2-3주 | ⭐⭐⭐⭐ |
| 음성 인식 | 2주 | ⭐⭐⭐ |
| NLP 및 요약 | 2주 | ⭐⭐ |
| 발언 시간 분석 | 1주 | ⭐ |
| 실시간 통합 | 3-4주 | ⭐⭐⭐⭐⭐ |
| **총계** | **13-17주** | **3-4개월** |

## 💰 예상 비용

### 개발 비용
- **기간**: 3-4개월
- **인원**: 3-4명
- **비용**: $50,000-100,000

### 운영 비용 (월간)
- 클라우드 서버 (GPU): $500-2,000
- Azure Speech Service: $1/시간
- 데이터베이스: $100-500
- 스토리지: $50-200
- OpenAI API: $100-500
- **총계**: $750-3,200/월

## ⚠️ 주요 도전 과제

1. **실시간 처리 지연 시간**
   - 목표: 1-3초 이내
   - 해결: 스트리밍 ASR, 병렬 처리

2. **화자 분리 정확도**
   - 비슷한 목소리 구분
   - 해결: 앙상블, 사전 등록

3. **한국어 인식 정확도**
   - 전문 용어 처리
   - 해결: Fine-tuning, 커스텀 어휘

4. **동시 발언 처리**
   - 여러 화자 동시 발언
   - 해결: Source Separation

## 🔒 보안 고려사항

- **데이터 암호화**: TLS 1.3, AES-256
- **접근 제어**: OAuth 2.0, RBAC
- **데이터 보존**: 자동 삭제 정책
- **규정 준수**: GDPR, CCPA, 개인정보보호법

## 📝 사용 예제

### 프로토타입 사용

```python
from scripts.meeting_analysis_prototype import MeetingAnalyzer

# 분석기 초기화
analyzer = MeetingAnalyzer(
    whisper_model="base",
    pyannote_token="YOUR_TOKEN"
)

# 회의 분석
results = analyzer.analyze_meeting("meeting.wav", language="ko")

# 결과 확인
analyzer.print_summary(results)

# 결과 저장
analyzer.save_results(results, "results.json")
```

### 결과 구조

```json
{
  "audio_file": "meeting.wav",
  "analysis_time": "2025-12-02T12:00:00",
  "transcription": {
    "text": "전체 전사 텍스트...",
    "segments": [...]
  },
  "speaker_diarization": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2,
      "duration": 5.2
    }
  ],
  "talk_time_stats": {
    "speakers": {
      "SPEAKER_00": {
        "total_seconds": 300.5,
        "total_minutes": 5.01,
        "number_of_turns": 15,
        "participation_rate": 60.5
      }
    },
    "total_meeting_time_minutes": 10.0,
    "number_of_speakers": 2
  },
  "combined_results": [
    {
      "speaker": "SPEAKER_00",
      "text": "안녕하세요...",
      "start": 0.0,
      "end": 2.5
    }
  ]
}
```

## 🛠️ 문제 해결

### Whisper 설치 오류
```bash
# FFmpeg 필요
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### pyannote.audio 권한 오류
- Hugging Face에서 모델 접근 권한 요청 필요
- 토큰이 올바른지 확인

### 메모리 부족
- 작은 Whisper 모델 사용 (`tiny`, `base`)
- GPU 사용 시 CUDA 메모리 확인

## 📚 참고 자료

### 공식 문서
- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Microsoft Graph API](https://docs.microsoft.com/en-us/graph/api/resources/communications-api-overview)
- [Zoom SDK](https://marketplace.zoom.us/docs/sdk/native-sdks/introduction)
- [Azure Speech Service](https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/)

### 연구 논문
- Speaker Diarization: A Review of Recent Research
- Real-time Speech Recognition: Challenges and Solutions

## 🤝 기여

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.
개선 사항이나 버그 리포트는 이슈로 등록해주세요.

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.
상업적 사용 시 각 라이브러리의 라이선스를 확인하세요.

---

**작성일**: 2025-12-02  
**버전**: 1.0  
**작성자**: AI Co-Scientist System




