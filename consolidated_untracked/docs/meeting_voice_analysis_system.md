# 실시간 회의 음성 분석 시스템 구축 가이드

## 📋 목차
1. [기존 솔루션 조사](#기존-솔루션-조사)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [필요한 기술 스택](#필요한-기술-스택)
4. [구현 단계별 가이드](#구현-단계별-가이드)
5. [기술적 도전 과제](#기술적-도전-과제)
6. [보안 및 프라이버시](#보안-및-프라이버시)

---

## 기존 솔루션 조사

### 상용 솔루션

#### 1. **Fireflies.ai** ⭐
- **기능**: 
  - 회의 자동 참여 및 녹음
  - 실시간 전사(Transcription)
  - 화자별 회의 노트 자동 생성
  - 발언 시간 분석
  - 감정 분석
  - 액션 아이템 추출
- **지원 플랫폼**: Zoom, Teams, Google Meet, Webex 등
- **가격**: 무료 플랜 + 유료 플랜
- **장점**: 즉시 사용 가능, 다양한 플랫폼 지원
- **단점**: 커스터마이징 제한, 데이터 프라이버시 우려

#### 2. **Otter.ai**
- **기능**: 실시간 전사, 화자 식별, 회의 요약
- **특징**: 모바일 앱 지원, 실시간 협업 기능

#### 3. **Microsoft Teams 내장 기능**
- **기능**: 
  - 자동 전사 (일부 플랜)
  - 회의 녹음
- **제한사항**: 
  - 화자별 상세 분석 제한적
  - 발언 시간 통계 부족
  - 실시간 분석 제한

#### 4. **Zoom 내장 기능**
- **기능**: 
  - 클라우드 녹음 및 전사
  - 화자 식별 (제한적)
- **제한사항**: 
  - 실시간 분석 부족
  - 상세 통계 제한

### 오픈소스 솔루션

#### 1. **Whisper + pyannote.audio**
- OpenAI Whisper: 음성 인식
- pyannote.audio: 화자 분리 (Speaker Diarization)
- **장점**: 무료, 오픈소스, 커스터마이징 가능
- **단점**: 실시간 처리 어려움, 인프라 필요

---

## 시스템 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────┐
│                    회의 플랫폼                          │
│  (Microsoft Teams / Zoom / Google Meet)                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Audio Stream / API
                   ▼
┌─────────────────────────────────────────────────────────┐
│              오디오 수집 레이어                          │
│  - Teams Graph API / Zoom SDK                           │
│  - Webhook / Real-time Stream                           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Raw Audio Data
                   ▼
┌─────────────────────────────────────────────────────────┐
│            전처리 및 오디오 처리                         │
│  - 노이즈 제거 (Noise Reduction)                        │
│  - 오디오 정규화                                        │
│  - 청크 분할 (Streaming Chunks)                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Processed Audio
                   ▼
┌─────────────────────────────────────────────────────────┐
│          화자 분리 (Speaker Diarization)                 │
│  - pyannote.audio / Resemblyzer                         │
│  - 화자 임베딩 추출                                      │
│  - 화자 클러스터링                                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Speaker-Segmented Audio
                   ▼
┌─────────────────────────────────────────────────────────┐
│        음성 인식 (ASR - Automatic Speech Recognition)   │
│  - OpenAI Whisper (Streaming)                           │
│  - Azure Speech Service                                 │
│  - Google Cloud Speech-to-Text                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Transcribed Text + Timestamps
                   ▼
┌─────────────────────────────────────────────────────────┐
│            자연어 처리 및 분석                            │
│  - 텍스트 요약 (Summarization)                          │
│  - 키워드 추출                                          │
│  - 액션 아이템 추출                                     │
│  - 감정 분석 (선택적)                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Analyzed Data
                   ▼
┌─────────────────────────────────────────────────────────┐
│            발언 시간 분석 엔진                           │
│  - 타임스탬프 추적                                      │
│  - 발언 시간 계산                                       │
│  - 참여도 통계                                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Statistics + Notes
                   ▼
┌─────────────────────────────────────────────────────────┐
│              데이터 저장 및 API                          │
│  - PostgreSQL / MongoDB                                 │
│  - REST API / WebSocket                                 │
│  - 대시보드 (Dashboard)                                 │
└─────────────────────────────────────────────────────────┘
```

### 실시간 처리 파이프라인

```
Real-time Audio Stream
    │
    ├─> Buffer (5-10초 청크)
    │
    ├─> Speaker Diarization (비동기)
    │   └─> 화자 임베딩 추출
    │   └─> 화자 클러스터링
    │
    ├─> ASR (스트리밍)
    │   └─> 실시간 텍스트 변환
    │
    ├─> NLP 처리 (배치, 30초마다)
    │   └─> 요약 생성
    │   └─> 키워드 추출
    │
    └─> 통계 업데이트 (실시간)
        └─> 발언 시간 누적
        └─> 대시보드 업데이트
```

---

## 필요한 기술 스택

### 1. 오디오 수집 및 스트리밍

#### Microsoft Teams
- **Microsoft Graph API**
  - `GET /communications/onlineMeetings/{id}/recordings`
  - `GET /communications/onlineMeetings/{id}/transcripts`
  - Real-time notifications via webhooks
- **Teams SDK**
  - Bot Framework
  - Meeting extensions

#### Zoom
- **Zoom SDK**
  - Meeting SDK for web/app
  - Cloud Recording API
  - Webhook events
- **OAuth 2.0** 인증 필요

#### Google Meet
- **Google Meet API** (제한적)
- **Chrome Extension** 방식으로 접근
- **Google Cloud Speech-to-Text** 통합

### 2. 화자 분리 (Speaker Diarization)

#### pyannote.audio
```python
# 설치
pip install pyannote.audio

# 사용 예시
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_TOKEN"
)

diarization = pipeline("audio.wav")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
```

**특징**:
- 사전 학습된 모델 사용 가능
- Hugging Face에서 모델 제공
- 실시간 처리 가능 (최적화 필요)

#### Resemblyzer
```python
# 화자 임베딩 추출
from resemblyzer import VoiceEncoder

encoder = VoiceEncoder()
embeddings = encoder.embed_utterance(audio)
```

**특징**:
- 경량 모델
- 실시간 처리에 적합
- 화자 식별에 특화

#### NVIDIA NeMo
- 엔터프라이즈급 솔루션
- 고성능 GPU 필요
- 상업적 라이선스 필요

### 3. 음성 인식 (ASR)

#### OpenAI Whisper
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.wav", language="ko")
```

**모델 크기**:
- `tiny`: 가장 빠름, 낮은 정확도
- `base`: 균형잡힌 성능
- `small`: 좋은 정확도
- `medium`: 높은 정확도
- `large`: 최고 정확도

**스트리밍 Whisper**:
- `whisper-streaming` 라이브러리
- 실시간 전사 지원

#### Azure Speech Service
```python
import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(
    subscription=subscription_key,
    region=region
)

speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config
)

result = speech_recognizer.recognize_once()
```

**특징**:
- 실시간 스트리밍 지원
- 화자 분리 내장 (Speaker Recognition)
- 다국어 지원
- 상업적 사용 가능

#### Google Cloud Speech-to-Text
```python
from google.cloud import speech

client = speech.SpeechClient()
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="ko-KR",
    enable_speaker_diarization=True,
    diarization_speaker_count=2,
)

response = client.recognize(config=config, audio=audio)
```

**특징**:
- 화자 분리 내장
- 실시간 스트리밍
- 높은 정확도

### 4. 자연어 처리 (NLP)

#### 텍스트 요약
- **OpenAI GPT-4 / GPT-3.5-turbo**
  ```python
  import openai
  
  response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": "회의 내용을 요약해주세요."},
          {"role": "user", "content": transcribed_text}
      ]
  )
  ```

- **LangChain**
  - 체인 기반 요약
  - 문서 분할 및 처리

- **Hugging Face Transformers**
  - `facebook/bart-large-cnn`
  - `google/pegasus-xsum`

#### 키워드 추출
- **spaCy** (한국어 지원)
- **Yake** (언어 독립적)
- **KeyBERT** (BERT 기반)

#### 액션 아이템 추출
- **GPT-4** 프롬프트 엔지니어링
- **NER (Named Entity Recognition)**
- **규칙 기반 패턴 매칭**

### 5. 데이터베이스 및 저장소

#### PostgreSQL
- 회의 메타데이터
- 전사 텍스트
- 통계 데이터
- 관계형 데이터 관리

#### MongoDB
- 유연한 스키마
- 대용량 텍스트 저장
- JSON 형태 데이터

#### Redis
- 실시간 캐싱
- 세션 관리
- 실시간 통계

### 6. 백엔드 프레임워크

#### FastAPI
```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.websocket("/ws/meeting/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    # 실시간 데이터 전송
```

**특징**:
- 비동기 처리
- WebSocket 지원
- 자동 API 문서화
- 높은 성능

#### Django Channels
- WebSocket 지원
- 실시간 기능
- Django 생태계 활용

### 7. 프론트엔드

#### React + TypeScript
- 실시간 대시보드
- WebSocket 클라이언트
- 차트 라이브러리 (Chart.js, Recharts)

#### Vue.js / Next.js
- 대안 프레임워크
- SSR 지원

---

## 구현 단계별 가이드

### Phase 1: 기본 인프라 구축 (1-2주)

#### 1.1 개발 환경 설정
```bash
# Python 가상환경
python -m venv venv
source venv/bin/activate

# 필수 패키지 설치
pip install fastapi uvicorn websockets
pip install openai-whisper
pip install pyannote.audio
pip install azure-cognitiveservices-speech
pip install openai langchain
pip install psycopg2-binary pymongo redis
```

#### 1.2 데이터베이스 스키마 설계
```sql
-- PostgreSQL 스키마 예시
CREATE TABLE meetings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50) NOT NULL,  -- 'teams', 'zoom', 'meet'
    meeting_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id),
    name VARCHAR(255),
    email VARCHAR(255),
    speaker_id VARCHAR(100),  -- 화자 분리에서 할당된 ID
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE transcriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id),
    speaker_id VARCHAR(100),
    text TEXT NOT NULL,
    start_time FLOAT,  -- 초 단위
    end_time FLOAT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE talk_time_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id),
    speaker_id VARCHAR(100),
    total_seconds FLOAT,
    number_of_turns INTEGER,
    average_turn_length FLOAT,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE meeting_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID REFERENCES meetings(id),
    summary TEXT,
    keywords TEXT[],  -- 배열
    action_items JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 1.3 기본 API 서버 구축
```python
# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Meeting Analysis API"}

@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    # 회의 정보 조회
    pass

@app.websocket("/ws/meeting/{meeting_id}")
async def websocket_endpoint(websocket: WebSocket, meeting_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # 실시간 데이터 처리
            await websocket.send_json({"status": "received"})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Phase 2: 오디오 수집 및 전처리 (2-3주)

#### 2.1 Teams 통합
```python
# teams_integration.py
import requests
from msal import ConfidentialClientApplication

class TeamsAudioCollector:
    def __init__(self, client_id, client_secret, tenant_id):
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.token = self.get_access_token()
    
    def get_access_token(self):
        app = ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
            client_credential=self.client_secret
        )
        result = app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )
        return result["access_token"]
    
    def get_meeting_recordings(self, meeting_id):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        url = f"https://graph.microsoft.com/v1.0/communications/onlineMeetings/{meeting_id}/recordings"
        response = requests.get(url, headers=headers)
        return response.json()
    
    def download_recording(self, recording_url):
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(recording_url, headers=headers, stream=True)
        return response.content
```

#### 2.2 Zoom 통합
```python
# zoom_integration.py
import requests
import base64

class ZoomAudioCollector:
    def __init__(self, account_id, client_id, client_secret):
        self.account_id = account_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_access_token()
    
    def get_access_token(self):
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "account_credentials", "account_id": self.account_id}
        
        response = requests.post(
            "https://zoom.us/oauth/token",
            headers=headers,
            data=data
        )
        return response.json()["access_token"]
    
    def get_meeting_recordings(self, meeting_id):
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"https://api.zoom.us/v2/meetings/{meeting_id}/recordings"
        response = requests.get(url, headers=headers)
        return response.json()
```

#### 2.3 오디오 전처리
```python
# audio_processor.py
import numpy as np
import librosa
from scipy import signal

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path):
        """오디오 파일 로드"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def remove_noise(self, audio):
        """노이즈 제거"""
        # Spectral gating 기반 노이즈 제거
        # 또는 noisereduce 라이브러리 사용
        import noisereduce as nr
        reduced_noise = nr.reduce_noise(audio, sr=self.sample_rate)
        return reduced_noise
    
    def normalize_audio(self, audio):
        """오디오 정규화"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def split_into_chunks(self, audio, chunk_duration=10):
        """오디오를 청크로 분할 (스트리밍용)"""
        chunk_size = int(self.sample_rate * chunk_duration)
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) == chunk_size:  # 완전한 청크만
                chunks.append(chunk)
        return chunks
```

### Phase 3: 화자 분리 구현 (2-3주)

#### 3.1 pyannote.audio 기반 화자 분리
```python
# speaker_diarization.py
from pyannote.audio import Pipeline
import torch

class SpeakerDiarizer:
    def __init__(self, auth_token):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        # GPU 사용 가능 시
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))
    
    def diarize(self, audio_path):
        """화자 분리 수행"""
        diarization = self.pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start
            })
        
        return segments
    
    def diarize_streaming(self, audio_chunk, previous_embeddings=None):
        """스트리밍 화자 분리 (최적화 필요)"""
        # 실시간 처리를 위한 경량 모델 사용
        # 또는 Resemblyzer 사용
        pass
```

#### 3.2 Resemblyzer 기반 실시간 화자 식별
```python
# realtime_speaker_id.py
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

class RealtimeSpeakerIdentifier:
    def __init__(self):
        self.encoder = VoiceEncoder()
        self.speaker_embeddings = {}  # speaker_id -> embedding
    
    def register_speaker(self, speaker_id, audio_samples):
        """화자 등록 (사전 등록된 화자)"""
        wav = preprocess_wav(audio_samples)
        embedding = self.encoder.embed_utterance(wav)
        self.speaker_embeddings[speaker_id] = embedding
    
    def identify_speaker(self, audio_chunk):
        """화자 식별"""
        wav = preprocess_wav(audio_chunk)
        embedding = self.encoder.embed_utterance(wav)
        
        # 기존 화자와 비교
        similarities = {}
        for speaker_id, known_embedding in self.speaker_embeddings.items():
            similarity = np.dot(embedding, known_embedding)
            similarities[speaker_id] = similarity
        
        if similarities:
            best_match = max(similarities.items(), key=lambda x: x[1])
            if best_match[1] > 0.7:  # 임계값
                return best_match[0]
        
        # 새로운 화자
        new_speaker_id = f"speaker_{len(self.speaker_embeddings)}"
        self.speaker_embeddings[new_speaker_id] = embedding
        return new_speaker_id
```

### Phase 4: 음성 인식 구현 (2주)

#### 4.1 Whisper 기반 전사
```python
# transcription_service.py
import whisper
import torch

class TranscriptionService:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def transcribe(self, audio_path, language="ko"):
        """오디오 파일 전사"""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe"
        )
        return result
    
    def transcribe_segment(self, audio_segment, start_time, end_time):
        """특정 세그먼트 전사"""
        result = self.model.transcribe(
            audio_segment,
            language="ko",
            initial_prompt="회의 내용입니다."
        )
        
        return {
            "text": result["text"],
            "start": start_time,
            "end": end_time,
            "segments": [
                {
                    "text": seg["text"],
                    "start": start_time + seg["start"],
                    "end": start_time + seg["end"]
                }
                for seg in result["segments"]
            ]
        }
```

#### 4.2 Azure Speech Service 통합
```python
# azure_transcription.py
import azure.cognitiveservices.speech as speechsdk

class AzureTranscriptionService:
    def __init__(self, subscription_key, region):
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key,
            region=region
        )
        self.speech_config.speech_recognition_language = "ko-KR"
        # 화자 분리 활성화
        self.speech_config.request_word_level_timestamps()
    
    def transcribe_streaming(self, audio_stream):
        """스트리밍 전사"""
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        speech_recognizer = speechsdk.SpeechRecognizer(
            self.speech_config,
            audio_config
        )
        
        results = []
        done = False
        
        def recognized_cb(evt):
            results.append({
                "text": evt.result.text,
                "offset": evt.result.offset / 10000000,  # 초 단위
                "duration": evt.result.duration / 10000000
            })
        
        def canceled_cb(evt):
            nonlocal done
            done = True
        
        speech_recognizer.recognized.connect(recognized_cb)
        speech_recognizer.canceled.connect(canceled_cb)
        
        speech_recognizer.start_continuous_recognition()
        
        while not done:
            time.sleep(0.5)
        
        speech_recognizer.stop_continuous_recognition()
        return results
```

### Phase 5: NLP 및 요약 (2주)

#### 5.1 회의 요약 생성
```python
# meeting_summarizer.py
import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

class MeetingSummarizer:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key
        self.llm = OpenAI(temperature=0.7)
    
    def summarize(self, transcript_text):
        """회의 내용 요약"""
        prompt = PromptTemplate(
            input_variables=["transcript"],
            template="""
            다음은 회의 전사 내용입니다. 주요 내용을 요약해주세요.
            
            전사 내용:
            {transcript}
            
            요약:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        summary = chain.run(transcript=transcript_text)
        return summary
    
    def extract_action_items(self, transcript_text):
        """액션 아이템 추출"""
        prompt = PromptTemplate(
            input_variables=["transcript"],
            template="""
            다음 회의 전사에서 액션 아이템(할 일)을 추출해주세요.
            JSON 형식으로 반환해주세요:
            {{
                "action_items": [
                    {{
                        "task": "작업 내용",
                        "assignee": "담당자",
                        "due_date": "마감일"
                    }}
                ]
            }}
            
            전사 내용:
            {transcript}
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(transcript=transcript_text)
        return result
    
    def extract_keywords(self, transcript_text, top_k=10):
        """키워드 추출"""
        from keybert import KeyBERT
        
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            transcript_text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_k=top_k
        )
        return [kw[0] for kw in keywords]
```

### Phase 6: 발언 시간 분석 (1주)

#### 6.1 통계 계산 엔진
```python
# talk_time_analyzer.py
from collections import defaultdict
from datetime import timedelta

class TalkTimeAnalyzer:
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "total_seconds": 0.0,
            "number_of_turns": 0,
            "turns": []
        })
    
    def add_transcription(self, speaker_id, start_time, end_time, text):
        """전사 데이터 추가"""
        duration = end_time - start_time
        
        self.stats[speaker_id]["total_seconds"] += duration
        self.stats[speaker_id]["number_of_turns"] += 1
        self.stats[speaker_id]["turns"].append({
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "text": text
        })
    
    def calculate_statistics(self, meeting_duration):
        """통계 계산"""
        results = {}
        
        for speaker_id, data in self.stats.items():
            total_seconds = data["total_seconds"]
            participation_rate = (total_seconds / meeting_duration) * 100 if meeting_duration > 0 else 0
            
            avg_turn_length = (
                total_seconds / data["number_of_turns"]
                if data["number_of_turns"] > 0
                else 0
            )
            
            results[speaker_id] = {
                "total_seconds": total_seconds,
                "total_minutes": total_seconds / 60,
                "participation_rate": participation_rate,
                "number_of_turns": data["number_of_turns"],
                "average_turn_length": avg_turn_length,
                "longest_turn": max(
                    [turn["duration"] for turn in data["turns"]],
                    default=0
                ),
                "shortest_turn": min(
                    [turn["duration"] for turn in data["turns"]],
                    default=0
                )
            }
        
        return results
    
    def get_timeline(self):
        """시간대별 발언 타임라인"""
        timeline = []
        for speaker_id, data in self.stats.items():
            for turn in data["turns"]:
                timeline.append({
                    "speaker": speaker_id,
                    "start": turn["start"],
                    "end": turn["end"],
                    "text": turn["text"]
                })
        
        timeline.sort(key=lambda x: x["start"])
        return timeline
```

### Phase 7: 통합 및 실시간 처리 (3-4주)

#### 7.1 실시간 파이프라인 통합
```python
# realtime_pipeline.py
import asyncio
from queue import Queue
import threading

class RealtimeMeetingPipeline:
    def __init__(self, meeting_id):
        self.meeting_id = meeting_id
        self.audio_queue = Queue()
        self.transcription_queue = Queue()
        self.speaker_diarizer = SpeakerDiarizer(auth_token="...")
        self.transcriber = TranscriptionService()
        self.analyzer = TalkTimeAnalyzer()
        self.running = False
    
    async def process_audio_stream(self, audio_chunk):
        """오디오 스트림 처리"""
        # 1. 화자 분리
        speaker_segments = await self.speaker_diarizer.diarize_streaming(
            audio_chunk
        )
        
        # 2. 각 화자 세그먼트 전사
        transcriptions = []
        for segment in speaker_segments:
            transcription = await self.transcriber.transcribe_segment(
                segment["audio"],
                segment["start"],
                segment["end"]
            )
            transcription["speaker"] = segment["speaker"]
            transcriptions.append(transcription)
            
            # 3. 통계 업데이트
            self.analyzer.add_transcription(
                transcription["speaker"],
                transcription["start"],
                transcription["end"],
                transcription["text"]
            )
        
        return transcriptions
    
    async def start_processing(self):
        """처리 시작"""
        self.running = True
        
        while self.running:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                transcriptions = await self.process_audio_stream(audio_chunk)
                
                # 결과 전송 (WebSocket 등)
                await self.send_results(transcriptions)
            
            await asyncio.sleep(0.1)  # CPU 부하 조절
    
    async def send_results(self, transcriptions):
        """결과 전송"""
        # WebSocket을 통해 클라이언트에 전송
        pass
```

#### 7.2 WebSocket 통합
```python
# websocket_handler.py
from fastapi import WebSocket
import json

class MeetingWebSocketManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, meeting_id: str):
        await websocket.accept()
        self.active_connections[meeting_id] = websocket
    
    def disconnect(self, meeting_id: str):
        if meeting_id in self.active_connections:
            del self.active_connections[meeting_id]
    
    async def send_transcription(self, meeting_id: str, data: dict):
        if meeting_id in self.active_connections:
            websocket = self.active_connections[meeting_id]
            await websocket.send_json({
                "type": "transcription",
                "data": data
            })
    
    async def send_statistics(self, meeting_id: str, stats: dict):
        if meeting_id in self.active_connections:
            websocket = self.active_connections[meeting_id]
            await websocket.send_json({
                "type": "statistics",
                "data": stats
            })
```

---

## 기술적 도전 과제

### 1. 실시간 처리 지연 시간 (Latency)

**문제**: 
- 화자 분리 + 전사 + NLP 처리 시간
- 실시간성 요구사항 (1-3초 이내)

**해결 방안**:
- **스트리밍 ASR 사용**: Azure Speech Service, Google Cloud Speech-to-Text
- **병렬 처리**: 화자 분리와 전사를 동시에 수행
- **청크 크기 최적화**: 5-10초 단위로 처리
- **GPU 가속**: CUDA를 활용한 모델 추론 가속

### 2. 화자 분리 정확도

**문제**:
- 비슷한 목소리 구분
- 화자 수 사전 알림 불가
- 배경 소음 및 음질 문제

**해결 방안**:
- **앙상블 방법**: 여러 모델 조합
- **사전 등록**: 참여자 음성 샘플 수집
- **오디오 전처리 강화**: 노이즈 제거, 음질 향상
- **후처리**: 문맥 기반 화자 보정

### 3. 한국어 음성 인식 정확도

**문제**:
- 한국어 특수성 (조사, 어미 변화)
- 전문 용어 인식
- 방언 및 억양

**해결 방안**:
- **한국어 특화 모델**: Whisper 한국어 fine-tuning
- **Custom Vocabulary**: 전문 용어 사전 추가
- **Language Model**: 후처리 언어 모델 적용

### 4. 동시 발언 (Overlapping Speech)

**문제**:
- 여러 화자가 동시에 말할 때
- 화자 분리 실패
- 전사 정확도 저하

**해결 방안**:
- **Source Separation**: 음원 분리 기술
- **Beamforming**: 마이크 배열 활용
- **후처리 보정**: 문맥 기반 복원

### 5. 확장성 및 성능

**문제**:
- 다중 회의 동시 처리
- 리소스 사용량
- 비용 관리

**해결 방안**:
- **마이크로서비스 아키텍처**: 독립적 스케일링
- **큐 시스템**: RabbitMQ, Kafka
- **캐싱**: Redis 활용
- **클라우드 오토스케일링**: AWS, Azure, GCP

---

## 보안 및 프라이버시

### 1. 데이터 암호화

- **전송 중 암호화**: TLS 1.3
- **저장 시 암호화**: AES-256
- **데이터베이스 암호화**: PostgreSQL TDE

### 2. 접근 제어

- **인증**: OAuth 2.0, JWT
- **권한 관리**: RBAC (Role-Based Access Control)
- **감사 로그**: 모든 접근 기록

### 3. 데이터 보존 정책

- **자동 삭제**: 설정된 기간 후 자동 삭제
- **사용자 요청 삭제**: GDPR 준수
- **백업 및 복구**: 정기 백업

### 4. 규정 준수

- **GDPR**: 유럽 개인정보 보호 규정
- **CCPA**: 캘리포니아 개인정보 보호법
- **개인정보 보호법**: 한국 법규 준수

---

## 예상 비용 및 리소스

### 인프라 비용 (월간)

| 항목 | 비용 (USD) | 비고 |
|------|-----------|------|
| 클라우드 서버 (GPU) | $500-2000 | AWS p3.2xlarge 등 |
| Azure Speech Service | $1/시간 | 사용량 기반 |
| 데이터베이스 | $100-500 | PostgreSQL, MongoDB |
| 스토리지 | $50-200 | 오디오 파일 저장 |
| API 호출 (OpenAI) | $100-500 | 요약, NLP 처리 |
| **총계** | **$750-3200** | 회의 수에 따라 변동 |

### 개발 시간 추정

| Phase | 기간 | 인원 |
|-------|------|------|
| Phase 1: 인프라 | 1-2주 | 1-2명 |
| Phase 2: 오디오 수집 | 2-3주 | 1-2명 |
| Phase 3: 화자 분리 | 2-3주 | 1명 |
| Phase 4: 음성 인식 | 2주 | 1명 |
| Phase 5: NLP | 2주 | 1명 |
| Phase 6: 통계 분석 | 1주 | 1명 |
| Phase 7: 통합 | 3-4주 | 2-3명 |
| **총계** | **13-17주** | **3-4개월** |

---

## 결론 및 권장사항

### 기존 솔루션 활용
- **Fireflies.ai** 같은 상용 솔루션을 먼저 검토
- 요구사항이 단순한 경우 상용 솔루션이 더 경제적

### 커스텀 개발이 필요한 경우
1. **MVP (Minimum Viable Product) 먼저 구축**
   - 기본 전사 기능
   - 간단한 화자 분리
   - 기본 통계

2. **점진적 개선**
   - 실시간 처리 추가
   - 정확도 향상
   - 고급 기능 추가

3. **오픈소스 활용**
   - Whisper, pyannote.audio 등
   - 커뮤니티 지원 활용

4. **클라우드 서비스 통합**
   - Azure Speech Service
   - Google Cloud Speech-to-Text
   - 관리형 서비스 활용

### 기술 스택 권장사항

**빠른 프로토타입**:
- Whisper (오프라인)
- pyannote.audio
- FastAPI
- PostgreSQL

**프로덕션**:
- Azure Speech Service (실시간)
- 커스텀 화자 분리 모델
- 마이크로서비스 아키텍처
- Kubernetes 오케스트레이션

---

## 참고 자료

### 공식 문서
- [Microsoft Graph API](https://docs.microsoft.com/en-us/graph/api/resources/communications-api-overview)
- [Zoom SDK](https://marketplace.zoom.us/docs/sdk/native-sdks/introduction)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Azure Speech Service](https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/)

### 연구 논문
- Speaker Diarization: A Review of Recent Research
- Real-time Speech Recognition: Challenges and Solutions
- Meeting Summarization: State of the Art

### 오픈소스 프로젝트
- [whisper-streaming](https://github.com/ufal/whisper_streaming)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- [NeMo](https://github.com/NVIDIA/NeMo)

---

**작성일**: 2025-12-02  
**버전**: 1.0  
**작성자**: AI Co-Scientist System




