# 중견 폴더 (Mid-sized Company Data)

이 폴더는 중견기업 관련 데이터를 저장하기 위한 공간입니다.

## 폴더 구조

```
data/중견 폴더/
├── README.md          # 이 파일
├── raw/               # 원본 데이터
├── processed/         # 전처리된 데이터
└── analysis/          # 분석 결과
```

## 사용 방법

1. 원본 데이터는 `raw/` 폴더에 저장
2. 전처리된 데이터는 `processed/` 폴더에 저장
3. 분석 결과는 `analysis/` 폴더에 저장

## 데이터 형식

지원하는 데이터 형식:
- CSV, Excel (.xlsx, .xls)
- JSON
- PDF 문서
- 텍스트 파일 (.txt)

## 주의사항

- 민감한 정보가 포함된 파일은 `.gitignore`에 추가하세요
- 대용량 파일은 Git LFS를 사용하거나 별도 저장소에 보관하세요
