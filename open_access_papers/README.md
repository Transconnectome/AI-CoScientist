# Open Access Papers Collection

이 폴더는 Scientific Reports, PLOS ONE, MDPI 저널의 오픈 액세스 논문들을 수집합니다.

## 대상 저널

1. **Scientific Reports** - Nature Publishing Group의 오픈 액세스 저널
2. **PLOS ONE** - 다학제 오픈 액세스 저널
3. **MDPI** - 여러 오픈 액세스 저널 출판사
   - Sensors
   - Applied Sciences
   - Brain Sciences
   - Diagnostics
   - Entropy
   - Healthcare
   - 등 다수

## 검색 키워드

- brain imaging machine learning
- fMRI deep learning
- neuroscience artificial intelligence
- neuroimaging
- brain decoding
- computational neuroscience deep learning
- neural networks brain
- EEG machine learning
- MEG neuroimaging

## 사용 방법

```bash
python3 download_open_access.py
```

## 특징

- 모든 논문이 오픈 액세스로 무료 다운로드 가능
- 2018년 이후 발표된 최신 논문들
- PDF 파일 유효성 검증
- 상세한 메타데이터 저장 (JSON 및 텍스트 형식)

## 출력 파일

- `*.pdf` - 다운로드된 논문 PDF 파일들
- `papers_summary.txt` - 논문 목록 및 상세 정보
- `papers_metadata.json` - 기계가 읽을 수 있는 메타데이터
