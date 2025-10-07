# Neuroscience & Machine Learning Papers Collection

이 폴더는 Brain Imaging, Neuroscience, Machine Learning 관련 논문들을 수집하기 위한 것입니다.

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

스크립트를 실행하여 arXiv에서 관련 논문 50개를 다운로드합니다:

```bash
python3 download_papers.py
```

## 검색 주제

스크립트는 다음 주제로 논문을 검색합니다:
- Brain imaging + Machine learning
- fMRI + Deep learning
- Neuroscience + Artificial intelligence
- Neural networks + Brain
- Neuroimaging + AI
- Brain decoding + Machine learning
- Computational neuroscience + Deep learning
- Brain connectivity + Neural networks

## 주의사항

- 이 스크립트는 arXiv의 공개 논문만 다운로드합니다
- 서버에 부담을 주지 않기 위해 다운로드 사이에 3초 딜레이가 있습니다
- Nature, Science 등의 상업 저널은 페이월이 있어 자동 다운로드가 불가능합니다

## 출력

- 다운로드된 PDF 파일들
- `papers_summary.txt`: 다운로드된 논문들의 상세 정보

## 추가 논문 수집 방법

상업 저널(Nature, Science 등)의 논문을 얻으려면:
1. 기관의 도서관 액세스를 통해 다운로드
2. Google Scholar에서 검색하여 저자가 제공하는 preprint 버전 찾기
3. ResearchGate나 저자의 개인 웹사이트에서 찾기
