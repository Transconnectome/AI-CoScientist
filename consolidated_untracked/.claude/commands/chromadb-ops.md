# ChromaDB 운영 스킬 (chromadb-ops)

## 개요
AI-CoScientist 프로젝트의 ChromaDB RAG 시스템 운영 및 관리 스킬입니다.

---

## ⚠️ 중요 데이터 (삭제 금지!)

### 발달장애 연구 DB
```
chromadb_data_dd/
├── dd_papers_L0: 1,387개 (원본 청크)
├── dd_papers_L1:   112개 (섹션 요약)  
└── dd_papers_L2:    26개 (논문 요약)
총: 1,525개 | 재생성 시간: 3시간 | API 비용: $50-100
```

### 메인 RAG 시스템
```
chromadb_data/
├── scientific_papers:     355개
├── improvement_patterns:   70개
├── research_documents: 10,000+개
└── successful_papers, user_history 등
총: ~10,400개 | 재생성 시간: 2-3시간
```

### 백업 위치
```
chromadb_backups/
└── chromadb_backup_YYYYMMDD_HHMMSS.tar.gz (최근 5개 유지)
```

---

## 🔧 빠른 명령어 참조

### 백업
```bash
# 전체 백업 (자동 회전, 5개 유지)
./scripts/backup_chromadb.sh

# 수동 백업
tar -czf chromadb_backup_$(date +%Y%m%d_%H%M%S).tar.gz chromadb_data_dd/ chromadb_data/
```

### 상태 확인
```bash
# 컬렉션 목록 및 개수 확인
poetry run python scripts/investigate_rag_history.py

# 빠른 상태 체크
poetry run python -c "
import chromadb
client = chromadb.PersistentClient(path='chromadb_data_dd')
for col in client.list_collections():
    print(f'{col.name}: {col.count()}개')
"
```

### 검색 테스트
```bash
# 발달장애 DB 검색
poetry run python scripts/query_dd_rag.py "autism diagnosis deep learning" -n 5

# 빠른 검색 (리랭킹 없이)
poetry run python scripts/query_dd_rag.py "brain development" --no-rerank
```

### 데이터 관리
```bash
# VectorDB만 재생성 (JSON에서, 2초)
poetry run python scripts/load_json_to_chromadb_dd.py

# 전체 재생성 (PDF부터, 3시간)
poetry run python scripts/ingest_golden_references_advanced.py --dir "data/발달장애/dd_papers" --all

# 새 논문 증분 추가
poetry run python scripts/ingest_golden_references_advanced.py --dir "data/발달장애/dd_papers" --incremental
```

---

## 📊 상태 확인 스크립트

사용자가 "ChromaDB 상태", "RAG 상태" 요청 시 실행:

```python
import chromadb
import os
from datetime import datetime

def check_chromadb_status():
    print("🗄️ ChromaDB 상태 대시보드")
    print("━" * 50)
    
    # 발달장애 DB
    if os.path.exists("chromadb_data_dd"):
        client_dd = chromadb.PersistentClient(path="chromadb_data_dd")
        print("\n📁 chromadb_data_dd/ (발달장애 전용)")
        total_dd = 0
        for col in client_dd.list_collections():
            count = col.count()
            total_dd += count
            print(f"   ├── {col.name}: {count:,}개")
        print(f"   총: {total_dd:,}개")
    
    # 메인 DB
    if os.path.exists("chromadb_data"):
        client_main = chromadb.PersistentClient(path="chromadb_data")
        print("\n📁 chromadb_data/ (메인 시스템)")
        total_main = 0
        for col in client_main.list_collections():
            count = col.count()
            total_main += count
            print(f"   ├── {col.name}: {count:,}개")
        print(f"   총: {total_main:,}개")
    
    # 백업 상태
    print("\n💾 백업 상태")
    backup_dir = "chromadb_backups"
    if os.path.exists(backup_dir):
        backups = sorted(os.listdir(backup_dir), reverse=True)
        if backups:
            latest = backups[0]
            # 날짜 추출 시도
            print(f"   ├── 최근: {latest}")
            print(f"   └── 총 {len(backups)}개 백업")
        else:
            print("   ⚠️ 백업 없음!")
    else:
        print("   ⚠️ 백업 폴더 없음!")

if __name__ == "__main__":
    check_chromadb_status()
```

---

## 🔄 복구 절차

### 백업에서 복구
```bash
# 1. 백업 목록 확인
ls -lh chromadb_backups/

# 2. 복구할 백업 선택 및 압축 해제
tar -xzf chromadb_backups/chromadb_backup_YYYYMMDD_HHMMSS.tar.gz

# 3. 복구 검증
poetry run python -c "
import chromadb
client = chromadb.PersistentClient(path='chromadb_data_dd')
print('컬렉션:', [c.name for c in client.list_collections()])
print('dd_papers_L0:', client.get_collection('dd_papers_L0').count(), '개')
"
```

### JSON에서 재생성 (백업 없을 때)
```bash
# 처리된 JSON이 있으면 2초만에 재생성 가능
ls data/발달장애/reference_papers/processed_json/

# 재생성
poetry run python scripts/load_json_to_chromadb_dd.py
```

### 완전 재생성 (모든 데이터 손실 시)
```bash
# PDF에서 다시 처리 (3시간 소요, API 비용 발생)
poetry run python scripts/ingest_golden_references_advanced.py \
  --dir "data/발달장애/dd_papers" \
  --all
```

---

## 🚨 트러블슈팅

### "Could not connect to a Chroma server"
```python
# HttpClient 대신 PersistentClient 사용
# ❌ 잘못된 방법
client = chromadb.HttpClient(host="localhost", port=8001)

# ✅ 올바른 방법
client = chromadb.PersistentClient(path="./chromadb_data_dd")
```

### 검색 결과가 없음
```bash
# 컬렉션 확인
poetry run python -c "
import chromadb
client = chromadb.PersistentClient(path='chromadb_data_dd')
col = client.get_collection('dd_papers_L0')
print(f'항목 수: {col.count()}')
# 샘플 확인
sample = col.peek(1)
print(f'샘플: {sample}')
"
```

### 쿼리 속도 느림
```python
# 성능 벤치마크
import time
import chromadb

client = chromadb.PersistentClient(path="chromadb_data_dd")
collection = client.get_collection("dd_papers_L0")

start = time.time()
results = collection.query(
    query_texts=["autism diagnosis"],
    n_results=50
)
elapsed = time.time() - start

print(f"쿼리 시간: {elapsed*1000:.1f}ms")
print(f"상태: {'✅ 정상' if elapsed < 0.5 else '⚠️ 최적화 필요'}")
```

---

## 📅 유지보수 체크리스트

### 주간
- [ ] `./scripts/backup_chromadb.sh` 실행
- [ ] 백업 파일 크기 확인 (비정상적 변화 없는지)

### 월간  
- [ ] 복구 테스트 (백업에서 임시 폴더로 복원 테스트)
- [ ] 오래된 백업 외부 저장소로 아카이브
- [ ] 컬렉션 무결성 확인

### 새 논문 추가 시
- [ ] PDF를 `data/발달장애/dd_papers/`에 추가
- [ ] 증분 인덱싱 실행
- [ ] 검색 테스트로 확인
- [ ] 백업 생성

---

## 🎯 자주 사용하는 시나리오

### "백업해줘"
→ `./scripts/backup_chromadb.sh` 실행

### "상태 확인해줘"  
→ 위의 상태 확인 스크립트 실행

### "새 논문 추가했어"
→ 증분 인덱싱 + 검색 테스트 + 백업

### "검색이 안 돼"
→ 컬렉션 존재 확인 → 항목 수 확인 → 샘플 쿼리 테스트

### "데이터 날아갔어"
→ 백업 확인 → 복구 → 검증

---

## 📂 관련 파일 위치

```
AI-CoScientist/
├── chromadb_data_dd/          # 발달장애 VectorDB
├── chromadb_data/             # 메인 VectorDB  
├── chromadb_backups/          # 백업 저장소
├── data/발달장애/
│   ├── dd_papers/             # 원본 PDF 26개
│   └── reference_papers/
│       └── processed_json/    # 처리된 JSON (재생성용)
└── scripts/
    ├── backup_chromadb.sh     # 백업 스크립트
    ├── load_json_to_chromadb_dd.py  # VectorDB 생성
    ├── query_dd_rag.py        # 검색 CLI
    └── investigate_rag_history.py   # 상태 확인
```

