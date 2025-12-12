#!/usr/bin/env python3
"""
로컬 문서를 RAG 시스템에 인제스트하는 스크립트.

지원 파일 형식:
- PDF (.pdf)
- Word 문서 (.docx)
- 텍스트 파일 (.txt)
- Markdown (.md)

사용법:
    python scripts/ingest_local_documents.py [폴더경로] [--collection 컬렉션명]

예시:
    python scripts/ingest_local_documents.py "data/중견 폴더"
    python scripts/ingest_local_documents.py "data/중견 폴더" --collection my_docs
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sentence_transformers import SentenceTransformer

console = Console()


class LocalDocumentIngestor:
    """로컬 문서를 ChromaDB에 인제스트하는 클래스."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def __init__(
        self,
        chromadb_host: str = "localhost",
        chromadb_port: int = 8001,
        collection_name: str = "local_documents",
        embedding_model: str = "allenai/scibert_scivocab_uncased",
    ):
        """인제스터 초기화.

        Args:
            chromadb_host: ChromaDB 호스트
            chromadb_port: ChromaDB 포트
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델 이름
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # 임베딩 모델 로드
        console.print(f"[cyan]임베딩 모델 로드 중: {embedding_model}[/cyan]")
        self._model: Optional[SentenceTransformer] = None

        # ChromaDB 연결 시도
        try:
            self.client = chromadb.HttpClient(
                host=chromadb_host,
                port=chromadb_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.client.heartbeat()
            console.print(
                f"[green]✓ ChromaDB 연결 성공: {chromadb_host}:{chromadb_port}[/green]"
            )
            self._use_http_client = True
        except Exception as e:
            console.print(
                f"[yellow]ChromaDB HTTP 연결 실패 ({e}), 로컬 저장소 사용[/yellow]"
            )
            # 로컬 영속 저장소 사용
            persist_dir = project_root / "chroma_local"
            persist_dir.mkdir(exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_dir))
            self._use_http_client = False

        # 컬렉션 생성/가져오기
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        console.print(f"[green]✓ 컬렉션 준비 완료: {collection_name}[/green]")

    @property
    def model(self) -> SentenceTransformer:
        """임베딩 모델 (lazy loading)."""
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def extract_text(self, file_path: Path) -> str:
        """파일에서 텍스트 추출.

        Args:
            file_path: 파일 경로

        Returns:
            추출된 텍스트
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix == ".docx":
            return self._extract_docx(file_path)
        elif suffix in {".txt", ".md"}:
            return self._extract_text_file(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

    def _extract_pdf(self, file_path: Path) -> str:
        """PDF에서 텍스트 추출."""
        try:
            import PyPDF2

            text_parts = []
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            console.print("[red]PyPDF2가 설치되지 않았습니다: pip install PyPDF2[/red]")
            return ""
        except Exception as e:
            console.print(f"[red]PDF 추출 오류: {e}[/red]")
            return ""

    def _extract_docx(self, file_path: Path) -> str:
        """DOCX에서 텍스트 추출."""
        try:
            from docx import Document

            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            console.print(
                "[red]python-docx가 설치되지 않았습니다: pip install python-docx[/red]"
            )
            return ""
        except Exception as e:
            console.print(f"[red]DOCX 추출 오류: {e}[/red]")
            return ""

    def _extract_text_file(self, file_path: Path) -> str:
        """텍스트 파일에서 내용 읽기."""
        encodings = ["utf-8", "cp949", "euc-kr", "latin-1"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        console.print(f"[red]텍스트 파일 인코딩 오류: {file_path}[/red]")
        return ""

    def generate_document_id(self, file_path: Path, content: str) -> str:
        """문서 ID 생성 (내용 기반 해시)."""
        hash_input = f"{file_path.name}:{content[:1000]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """텍스트를 청크로 분할.

        Args:
            text: 원본 텍스트
            chunk_size: 청크 크기 (문자 수)
            overlap: 오버랩 크기

        Returns:
            청크 리스트
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # 문장 경계에서 자르기 시도
            if end < len(text):
                # 마침표, 느낌표, 물음표 찾기
                for sep in [".\n", "。\n", "\n\n", ". ", "。", "!", "?", "\n"]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def ingest_file(self, file_path: Path) -> dict:
        """단일 파일 인제스트.

        Args:
            file_path: 파일 경로

        Returns:
            인제스트 결과 정보
        """
        # 텍스트 추출
        text = self.extract_text(file_path)
        if not text.strip():
            return {"status": "skipped", "reason": "빈 내용", "file": str(file_path)}

        # 청크 분할
        chunks = self.chunk_text(text)

        # 각 청크에 대해 임베딩 생성 및 저장
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{self.generate_document_id(file_path, text)}_{i}"
            chunk_ids.append(doc_id)

            # 임베딩 생성
            embedding = self.model.encode(chunk, convert_to_numpy=True).tolist()

            # 메타데이터
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": datetime.utcnow().isoformat(),
                "folder": str(file_path.parent.name),
            }

            # ChromaDB에 추가 (upsert로 중복 방지)
            self.collection.upsert(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata],
            )

        return {
            "status": "success",
            "file": str(file_path),
            "chunks": len(chunks),
            "text_length": len(text),
        }

    def ingest_folder(self, folder_path: str | Path) -> list[dict]:
        """폴더 내 모든 문서 인제스트.

        Args:
            folder_path: 폴더 경로

        Returns:
            각 파일의 인제스트 결과 리스트
        """
        folder = Path(folder_path)
        if not folder.exists():
            console.print(f"[red]폴더가 존재하지 않습니다: {folder}[/red]")
            return []

        # 지원 파일 찾기
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(folder.rglob(f"*{ext}"))

        if not files:
            console.print(f"[yellow]인제스트할 파일이 없습니다: {folder}[/yellow]")
            return []

        console.print(f"\n[cyan]총 {len(files)}개 파일 발견[/cyan]\n")

        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("인제스트 중...", total=len(files))

            for file_path in files:
                progress.update(task, description=f"처리 중: {file_path.name}")
                result = self.ingest_file(file_path)
                results.append(result)
                progress.advance(task)

        return results

    def show_collection_stats(self):
        """컬렉션 통계 출력."""
        count = self.collection.count()
        console.print(f"\n[green]컬렉션 '{self.collection_name}' 통계:[/green]")
        console.print(f"  - 총 문서(청크) 수: {count}")

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """의미 검색 수행.

        Args:
            query: 검색 쿼리
            n_results: 결과 수

        Returns:
            검색 결과 리스트
        """
        embedding = self.model.encode(query, convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=[embedding], n_results=n_results
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i]
                        if results["documents"]
                        else None,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else None,
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else None,
                    }
                )
        return output


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="로컬 문서를 RAG 시스템에 인제스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder", nargs="?", default="data/중견 폴더", help="인제스트할 폴더 경로"
    )
    parser.add_argument(
        "--collection",
        "-c",
        default="local_documents",
        help="ChromaDB 컬렉션 이름 (기본값: local_documents)",
    )
    parser.add_argument(
        "--host", default="localhost", help="ChromaDB 호스트 (기본값: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="ChromaDB 포트 (기본값: 8001)"
    )
    parser.add_argument(
        "--model",
        default="allenai/scibert_scivocab_uncased",
        help="임베딩 모델 (기본값: allenai/scibert_scivocab_uncased)",
    )
    parser.add_argument(
        "--search", "-s", help="인제스트 후 테스트 검색 쿼리 실행"
    )

    args = parser.parse_args()

    console.print("[bold blue]═══════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]     로컬 문서 RAG 인제스터 (AI-CoScientist)[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════[/bold blue]\n")

    # 인제스터 초기화
    ingestor = LocalDocumentIngestor(
        chromadb_host=args.host,
        chromadb_port=args.port,
        collection_name=args.collection,
        embedding_model=args.model,
    )

    # 폴더 인제스트
    folder_path = project_root / args.folder
    console.print(f"[cyan]대상 폴더: {folder_path}[/cyan]")

    results = ingestor.ingest_folder(folder_path)

    # 결과 출력
    if results:
        table = Table(title="인제스트 결과")
        table.add_column("파일", style="cyan")
        table.add_column("상태", style="green")
        table.add_column("청크 수", justify="right")
        table.add_column("텍스트 길이", justify="right")

        success_count = 0
        for r in results:
            status = r.get("status", "unknown")
            if status == "success":
                success_count += 1
                table.add_row(
                    Path(r["file"]).name,
                    "✓ 성공",
                    str(r.get("chunks", "-")),
                    str(r.get("text_length", "-")),
                )
            else:
                table.add_row(
                    Path(r["file"]).name,
                    f"⊘ {r.get('reason', '실패')}",
                    "-",
                    "-",
                )

        console.print(table)
        console.print(f"\n[green]✓ {success_count}/{len(results)} 파일 인제스트 완료[/green]")

    # 통계 출력
    ingestor.show_collection_stats()

    # 테스트 검색
    if args.search:
        console.print(f"\n[cyan]테스트 검색: '{args.search}'[/cyan]")
        search_results = ingestor.search(args.search)

        if search_results:
            for i, r in enumerate(search_results, 1):
                console.print(f"\n[yellow]결과 {i}:[/yellow]")
                console.print(f"  파일: {r['metadata'].get('filename', 'N/A')}")
                console.print(f"  거리: {r['distance']:.4f}")
                snippet = r["document"][:200] + "..." if len(r["document"]) > 200 else r["document"]
                console.print(f"  내용: {snippet}")
        else:
            console.print("[yellow]검색 결과 없음[/yellow]")


if __name__ == "__main__":
    main()
