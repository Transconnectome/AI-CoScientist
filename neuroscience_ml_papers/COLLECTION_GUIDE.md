# Top-5 Venue 논문 수집 가이드

## 현실적인 문제점
- Nature/Science: 페이월로 자동 다운로드 불가능
- NeurIPS/ICLR/AAAI: 대규모 크롤링 필요, API rate limit 존재

## 추천하는 실용적 방법

### 1. 기관 도서관 접근 사용 (가장 확실)
대학/연구기관의 도서관 VPN을 통해 접속하면:
- Nature, Science 논문 전문 다운로드 가능
- 모든 페이월 논문 접근 가능

### 2. Google Scholar 활용
```
검색: "brain imaging" OR "fMRI" OR "neuroimaging" 
       (NeurIPS OR ICLR OR AAAI OR Nature OR Science)
       filetype:pdf
```
- 저자가 업로드한 preprint/postprint 찾기
- ResearchGate, arxiv 버전 찾기

### 3. 각 Venue의 공식 소스

#### NeurIPS
- https://papers.nips.cc/
- 년도별로 검색: brain, fMRI, neuroscience 등
- PDF 직접 다운로드 가능

#### ICLR
- https://openreview.net/group?id=ICLR.cc/2024/Conference
- Accept된 논문만 필터링
- PDF 버튼 클릭으로 다운로드

#### AAAI
- https://aaai.org/Library/AAAI/aaai-library.php
- 년도별 proceedings 접근
- 일부만 open access

### 4. 추천 검색 전략

**PubMed (Nature/Science 논문 많음)**
```
("brain imaging"[Title/Abstract] OR "fMRI"[Title/Abstract])
AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract])
AND ("Nature"[Journal] OR "Science"[Journal])
```

**Semantic Scholar**
- https://www.semanticscholar.org/
- Venue 필터: Nature, Science, NeurIPS, ICLR, AAAI
- "Open Access" 필터 적용
- PDF available만 선택

## 반자동화 스크립트 사용법

아래 스크립트는 찾은 PDF URL 리스트를 받아서 다운로드합니다:
```python
python3 url_downloader.py
```

스크립트는 urls.txt 파일의 URL 목록을 읽어서 PDF를 다운로드합니다.

## URL 수집 예시

`urls.txt` 파일 형식:
```
# NeurIPS Papers
https://papers.nips.cc/paper_files/paper/2023/file/xxx-Paper-Conference.pdf | Brain Imaging Paper | NeurIPS 2023
https://papers.nips.cc/paper_files/paper/2024/file/yyy-Paper-Conference.pdf | fMRI Analysis | NeurIPS 2024

# ICLR Papers  
https://openreview.net/pdf?id=xxxxx | Neuroscience ML | ICLR 2024

# Nature (open access만)
https://www.nature.com/articles/sxxxxx.pdf | Brain Decoding | Nature 2023
```

## 실제 작업 권장 순서

1. **NeurIPS**: papers.nips.cc 검색 → PDF 링크 수집 (15-20개)
2. **ICLR**: openreview.net 검색 → PDF 링크 수집 (10-15개)
3. **AAAI**: aaai.org proceedings 확인 (5-10개)
4. **Nature/Science**: 
   - 기관 도서관으로 접속하여 직접 다운로드, 또는
   - Google Scholar에서 open access 버전 찾기 (5-10개)

이렇게 하면 50개 목표를 달성할 수 있습니다.

## 도움이 필요하시면

특정 venue나 topic에 대해 URL 리스트를 찾아드릴 수 있습니다.
예: "NeurIPS 2023-2024 brain imaging 논문 URL 찾아줘"
