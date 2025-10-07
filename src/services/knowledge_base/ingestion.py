"""Literature ingestion service."""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.literature import Literature, Author, FieldOfStudy
from src.services.external import SemanticScholarClient, CrossRefClient
from src.services.knowledge_base.vector_store import VectorStore
from src.services.knowledge_base.embedding import EmbeddingService


class LiteratureIngestion:
    """Ingest papers from external sources."""

    def __init__(
        self,
        db: AsyncSession,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        semantic_scholar_client: SemanticScholarClient,
        crossref_client: CrossRefClient
    ):
        """Initialize ingestion service."""
        self.db = db
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.ss_client = semantic_scholar_client
        self.crossref_client = crossref_client

    async def ingest_by_doi(self, doi: str) -> UUID:
        """Ingest paper by DOI."""
        # Check if already exists
        query = select(Literature).where(Literature.doi == doi)
        result = await self.db.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            return existing.id

        # Try Semantic Scholar first
        paper_data = await self.ss_client.get_paper(f"DOI:{doi}")

        # Fallback to CrossRef
        if not paper_data:
            paper_data = await self.crossref_client.get_work(doi)

        if not paper_data:
            raise ValueError(f"Could not find paper with DOI: {doi}")

        # Store paper
        paper_id = await self._store_paper(paper_data)

        # Generate and store embedding
        await self._embed_paper(paper_id, paper_data)

        return paper_id

    async def ingest_by_query(
        self,
        query: str,
        max_results: int = 50
    ) -> List[UUID]:
        """Ingest papers by search query."""
        # Search Semantic Scholar
        results = await self.ss_client.search_papers(query, limit=max_results)

        paper_ids = []
        for paper_data in results:
            # Check if exists
            doi = paper_data.get("paperId")
            if doi:
                query_stmt = select(Literature).where(Literature.doi == doi)
                result = await self.db.execute(query_stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    paper_ids.append(existing.id)
                    continue

            # Store new paper
            try:
                paper_id = await self._store_paper(paper_data)
                await self._embed_paper(paper_id, paper_data)
                paper_ids.append(paper_id)
            except Exception as e:
                print(f"Error ingesting paper: {e}")
                continue

        return paper_ids

    async def _store_paper(self, paper_data: dict) -> UUID:
        """Store paper in PostgreSQL."""
        # Parse publication date
        pub_date = None
        if paper_data.get("year"):
            try:
                pub_date = date(paper_data["year"], 1, 1)
            except (ValueError, TypeError):
                pass

        # Create paper
        paper = Literature(
            doi=paper_data.get("doi") or paper_data.get("paperId"),
            title=paper_data.get("title", ""),
            abstract=paper_data.get("abstract"),
            publication_date=pub_date,
            journal=paper_data.get("journal", {}).get("name") if isinstance(
                paper_data.get("journal"), dict
            ) else paper_data.get("journal"),
            citations_count=paper_data.get("citationCount", 0),
            url=paper_data.get("url")
        )

        # Extract PDF URL if available
        if "openAccessPdf" in paper_data and paper_data["openAccessPdf"]:
            paper.pdf_url = paper_data["openAccessPdf"].get("url")

        self.db.add(paper)
        await self.db.flush()  # Get paper ID without committing

        # Insert authors
        for author_data in paper_data.get("authors", []):
            author_name = author_data.get("name", "")
            if not author_name:
                continue

            # Check if author exists
            author_query = select(Author).where(Author.name == author_name)
            author_result = await self.db.execute(author_query)
            author = author_result.scalar_one_or_none()

            if not author:
                author = Author(name=author_name)
                self.db.add(author)
                await self.db.flush()

            # Add to paper
            paper.authors.append(author)

        # Insert fields of study
        for field_name in paper_data.get("fieldsOfStudy", []):
            # Check if field exists
            field_query = select(FieldOfStudy).where(FieldOfStudy.name == field_name)
            field_result = await self.db.execute(field_query)
            field = field_result.scalar_one_or_none()

            if not field:
                field = FieldOfStudy(name=field_name)
                self.db.add(field)
                await self.db.flush()

            # Add to paper
            paper.fields.append(field)

        await self.db.commit()
        await self.db.refresh(paper)

        return paper.id

    async def _embed_paper(self, paper_id: UUID, paper_data: dict) -> None:
        """Generate and store embeddings."""
        # Combine title and abstract
        text = f"{paper_data.get('title', '')}. {paper_data.get('abstract', '')}"

        # Generate embedding
        embedding = await self.embedding_service.encode_async(text)

        # Extract metadata
        metadata = {
            "type": "paper",
            "doi": paper_data.get("doi", ""),
            "title": paper_data.get("title", ""),
            "year": paper_data.get("year", 0),
            "citations_count": paper_data.get("citationCount", 0),
            "journal": paper_data.get("journal", {}).get("name", "") if isinstance(
                paper_data.get("journal"), dict
            ) else paper_data.get("journal", ""),
            "created_at": datetime.utcnow().isoformat()
        }

        # Add fields of study
        if "fieldsOfStudy" in paper_data and paper_data["fieldsOfStudy"]:
            metadata["field_of_study"] = paper_data["fieldsOfStudy"][0]

        # Store in ChromaDB
        self.vector_store.add_documents(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[str(paper_id)]
        )
