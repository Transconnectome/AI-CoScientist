# Knowledge Base Service Design

## 🎯 Purpose
과학 문헌 관리 및 검색을 위한 하이브리드 데이터베이스 시스템으로, 의미론적 검색과 구조화된 메타데이터 관리를 결합합니다.

## 🏗️ Component Architecture

```
┌────────────────────────────────────────────┐
│      Knowledge Base API Layer              │
│  - Search Interface                        │
│  - Literature Ingestion                    │
│  - Citation Management                     │
└────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Vector  │   │Metadata │   │  Graph  │
│  Store  │   │  Store  │   │Database │
│(Chroma) │   │(Postgres│   │(optional│
└─────────┘   └─────────┘   └─────────┘
    │               │               │
    └───────────────┼───────────────┘
                    ▼
        ┌───────────────────────┐
        │   Embedding Engine    │
        │   - SciBERT           │
        │   - Sentence-BERT     │
        │   - Caching           │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  External Sources     │
        │  - Semantic Scholar   │
        │  - PubMed             │
        │  - arXiv              │
        └───────────────────────┘
```

## 📋 Data Models

### Vector Store Schema (ChromaDB)

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class DocumentType(Enum):
    """Types of documents in knowledge base"""
    PAPER = "paper"
    ABSTRACT = "abstract"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    CONCEPT = "concept"

@dataclass
class Document:
    """Document in knowledge base"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict
    doc_type: DocumentType
    created_at: datetime

@dataclass
class PaperMetadata:
    """Metadata for research papers"""
    doi: str
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    journal: str
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    citations_count: int
    fields_of_study: List[str]
    keywords: List[str]
    url: str
    pdf_url: Optional[str]

@dataclass
class SearchResult:
    """Search result from knowledge base"""
    document: Document
    score: float
    highlights: List[str]
    context: Optional[str]
```

### PostgreSQL Schema

```sql
-- Papers table
CREATE TABLE papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doi VARCHAR(255) UNIQUE,
    title TEXT NOT NULL,
    abstract TEXT,
    publication_date DATE,
    journal VARCHAR(500),
    volume VARCHAR(50),
    issue VARCHAR(50),
    pages VARCHAR(50),
    citations_count INTEGER DEFAULT 0,
    url TEXT,
    pdf_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP
);

CREATE INDEX idx_papers_doi ON papers(doi);
CREATE INDEX idx_papers_publication_date ON papers(publication_date);
CREATE INDEX idx_papers_citations ON papers(citations_count);

-- Authors table
CREATE TABLE authors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    affiliation VARCHAR(500),
    orcid VARCHAR(50),
    h_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_authors_name ON authors(name);
CREATE INDEX idx_authors_orcid ON authors(orcid);

-- Paper-Author relationship
CREATE TABLE paper_authors (
    paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    author_id UUID REFERENCES authors(id) ON DELETE CASCADE,
    author_order INTEGER,
    is_corresponding BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX idx_paper_authors_paper ON paper_authors(paper_id);
CREATE INDEX idx_paper_authors_author ON paper_authors(author_id);

-- Fields of study
CREATE TABLE fields_of_study (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    parent_id UUID REFERENCES fields_of_study(id),
    level INTEGER DEFAULT 0
);

-- Paper-Field relationship
CREATE TABLE paper_fields (
    paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    field_id UUID REFERENCES fields_of_study(id) ON DELETE CASCADE,
    relevance_score FLOAT,
    PRIMARY KEY (paper_id, field_id)
);

-- Citations network
CREATE TABLE citations (
    citing_paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    cited_paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

CREATE INDEX idx_citations_citing ON citations(citing_paper_id);
CREATE INDEX idx_citations_cited ON citations(cited_paper_id);

-- Research projects
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    domain VARCHAR(255),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Project-Paper relationship (saved/relevant papers)
CREATE TABLE project_papers (
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    relevance_score FLOAT,
    notes TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (project_id, paper_id)
);

-- Search history for learning
CREATE TABLE search_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    query TEXT NOT NULL,
    filters JSONB,
    results_count INTEGER,
    top_result_ids UUID[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_search_history_project ON search_history(project_id);
CREATE INDEX idx_search_history_created ON search_history(created_at);
```

## 🔍 Search Interface

```python
from chromadb import Client
from chromadb.config import Settings
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer

class KnowledgeBaseSearch:
    """Search interface for knowledge base"""

    def __init__(
        self,
        chroma_client: Client,
        pg_connection: Any,
        embedding_model: str = "allenai/scibert_scivocab_uncased"
    ):
        self.chroma = chroma_client
        self.pg = pg_connection
        self.encoder = SentenceTransformer(embedding_model)

        # Get or create collection
        self.collection = self.chroma.get_or_create_collection(
            name="scientific_papers",
            metadata={"hnsw:space": "cosine"}
        )

    async def semantic_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        min_score: float = 0.5
    ) -> List[SearchResult]:
        """Semantic search using embeddings"""

        # Generate query embedding
        query_embedding = self.encoder.encode(query).tolist()

        # Build metadata filter
        where_filter = self._build_filter(filters) if filters else None

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        # Parse results
        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            score = 1 - results['distances'][0][i]  # Convert distance to similarity

            if score < min_score:
                continue

            document = Document(
                id=doc_id,
                content=results['documents'][0][i],
                embedding=results['embeddings'][0][i] if results['embeddings'] else [],
                metadata=results['metadatas'][0][i],
                doc_type=DocumentType(results['metadatas'][0][i]['type']),
                created_at=datetime.fromisoformat(
                    results['metadatas'][0][i]['created_at']
                )
            )

            # Get highlights
            highlights = self._extract_highlights(
                query,
                results['documents'][0][i]
            )

            search_results.append(SearchResult(
                document=document,
                score=score,
                highlights=highlights,
                context=None
            ))

        return search_results

    async def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword search"""

        # Semantic search
        semantic_results = await self.semantic_search(
            query,
            filters,
            top_k * 2
        )

        # Keyword search from PostgreSQL
        keyword_results = await self._keyword_search(
            query,
            filters,
            top_k * 2
        )

        # Combine and re-rank
        combined = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight
        )

        return combined[:top_k]

    async def find_similar_papers(
        self,
        paper_id: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Find papers similar to given paper"""

        # Get paper embedding
        paper_doc = self.collection.get(ids=[paper_id])

        if not paper_doc['embeddings']:
            raise ValueError(f"Paper {paper_id} not found")

        # Search for similar
        results = self.collection.query(
            query_embeddings=paper_doc['embeddings'],
            n_results=top_k + 1  # +1 to exclude self
        )

        # Parse and filter out the query paper itself
        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id == paper_id:
                continue

            score = 1 - results['distances'][0][i]

            document = Document(
                id=doc_id,
                content=results['documents'][0][i],
                embedding=results['embeddings'][0][i] if results['embeddings'] else [],
                metadata=results['metadatas'][0][i],
                doc_type=DocumentType(results['metadatas'][0][i]['type']),
                created_at=datetime.fromisoformat(
                    results['metadatas'][0][i]['created_at']
                )
            )

            search_results.append(SearchResult(
                document=document,
                score=score,
                highlights=[],
                context=None
            ))

        return search_results[:top_k]

    async def citation_based_search(
        self,
        paper_id: str,
        depth: int = 2,
        direction: str = "both"  # "citing", "cited", "both"
    ) -> List[str]:
        """Find papers through citation network"""

        query = """
        WITH RECURSIVE citation_tree AS (
            -- Base case
            SELECT
                CASE
                    WHEN :direction IN ('citing', 'both')
                    THEN citing_paper_id
                    ELSE cited_paper_id
                END as paper_id,
                1 as depth
            FROM citations
            WHERE
                CASE
                    WHEN :direction = 'citing' THEN cited_paper_id = :paper_id
                    WHEN :direction = 'cited' THEN citing_paper_id = :paper_id
                    ELSE citing_paper_id = :paper_id OR cited_paper_id = :paper_id
                END

            UNION ALL

            -- Recursive case
            SELECT
                CASE
                    WHEN :direction IN ('citing', 'both')
                    THEN c.citing_paper_id
                    ELSE c.cited_paper_id
                END as paper_id,
                ct.depth + 1
            FROM citations c
            INNER JOIN citation_tree ct ON
                CASE
                    WHEN :direction = 'citing' THEN c.cited_paper_id = ct.paper_id
                    WHEN :direction = 'cited' THEN c.citing_paper_id = ct.paper_id
                    ELSE c.citing_paper_id = ct.paper_id OR c.cited_paper_id = ct.paper_id
                END
            WHERE ct.depth < :max_depth
        )
        SELECT DISTINCT paper_id
        FROM citation_tree
        WHERE paper_id != :paper_id;
        """

        result = await self.pg.fetch(
            query,
            paper_id=paper_id,
            direction=direction,
            max_depth=depth
        )

        return [row['paper_id'] for row in result]

    async def concept_search(
        self,
        concept: str,
        top_k: int = 20
    ) -> List[SearchResult]:
        """Search by scientific concept"""

        # Expand concept using related terms
        expanded_query = await self._expand_concept(concept)

        # Search with expanded query
        return await self.semantic_search(
            expanded_query,
            filters={"type": "paper"},
            top_k=top_k
        )

    def _build_filter(self, filters: Dict[str, Any]) -> Dict:
        """Build ChromaDB metadata filter"""
        where_filter = {}

        if "year_min" in filters:
            where_filter["year"] = {"$gte": filters["year_min"]}

        if "year_max" in filters:
            if "year" in where_filter:
                where_filter["year"]["$lte"] = filters["year_max"]
            else:
                where_filter["year"] = {"$lte": filters["year_max"]}

        if "fields" in filters:
            where_filter["field_of_study"] = {"$in": filters["fields"]}

        if "journal" in filters:
            where_filter["journal"] = filters["journal"]

        if "min_citations" in filters:
            where_filter["citations_count"] = {"$gte": filters["min_citations"]}

        return where_filter

    def _extract_highlights(
        self,
        query: str,
        text: str,
        window: int = 100
    ) -> List[str]:
        """Extract relevant text snippets"""
        query_terms = query.lower().split()
        text_lower = text.lower()

        highlights = []
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                start = max(0, pos - window)
                end = min(len(text), pos + len(term) + window)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."
                highlights.append(snippet)

        return highlights[:3]  # Top 3 highlights

    async def _keyword_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[SearchResult]:
        """Full-text search using PostgreSQL"""

        # Build WHERE clause
        where_clauses = ["to_tsvector('english', title || ' ' || abstract) @@ plainto_tsquery('english', $1)"]
        params = [query]
        param_count = 1

        if filters:
            if "year_min" in filters:
                param_count += 1
                where_clauses.append(f"EXTRACT(YEAR FROM publication_date) >= ${param_count}")
                params.append(filters["year_min"])

            if "year_max" in filters:
                param_count += 1
                where_clauses.append(f"EXTRACT(YEAR FROM publication_date) <= ${param_count}")
                params.append(filters["year_max"])

        where_clause = " AND ".join(where_clauses)

        query_sql = f"""
        SELECT
            id,
            title,
            abstract,
            ts_rank(
                to_tsvector('english', title || ' ' || abstract),
                plainto_tsquery('english', $1)
            ) as rank
        FROM papers
        WHERE {where_clause}
        ORDER BY rank DESC
        LIMIT ${ param_count + 1};
        """

        params.append(top_k)

        results = await self.pg.fetch(query_sql, *params)

        # Convert to SearchResult
        search_results = []
        for row in results:
            document = Document(
                id=str(row['id']),
                content=row['abstract'],
                embedding=[],
                metadata={"title": row['title']},
                doc_type=DocumentType.PAPER,
                created_at=datetime.utcnow()
            )

            search_results.append(SearchResult(
                document=document,
                score=float(row['rank']),
                highlights=[],
                context=None
            ))

        return search_results

    def _combine_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float
    ) -> List[SearchResult]:
        """Combine and re-rank results from multiple sources"""

        # Create score dictionary
        scores = {}

        # Add semantic scores
        for result in semantic_results:
            scores[result.document.id] = {
                "semantic": result.score,
                "keyword": 0.0,
                "document": result.document,
                "highlights": result.highlights
            }

        # Add keyword scores
        for result in keyword_results:
            if result.document.id in scores:
                scores[result.document.id]["keyword"] = result.score
            else:
                scores[result.document.id] = {
                    "semantic": 0.0,
                    "keyword": result.score,
                    "document": result.document,
                    "highlights": result.highlights
                }

        # Compute combined scores
        combined = []
        for doc_id, data in scores.items():
            combined_score = (
                semantic_weight * data["semantic"] +
                (1 - semantic_weight) * data["keyword"]
            )

            combined.append(SearchResult(
                document=data["document"],
                score=combined_score,
                highlights=data["highlights"],
                context=None
            ))

        # Sort by combined score
        combined.sort(key=lambda x: x.score, reverse=True)

        return combined

    async def _expand_concept(self, concept: str) -> str:
        """Expand concept with related terms"""
        # This could use WordNet, domain ontologies, or LLM
        # For now, simple implementation
        return concept
```

## 📥 Literature Ingestion

```python
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup

class LiteratureIngestion:
    """Ingest papers from external sources"""

    def __init__(
        self,
        kb_search: KnowledgeBaseSearch,
        semantic_scholar_api_key: Optional[str] = None
    ):
        self.kb = kb_search
        self.ss_api_key = semantic_scholar_api_key

    async def ingest_by_doi(self, doi: str) -> str:
        """Ingest paper by DOI"""

        # Check if already exists
        existing = await self.kb.pg.fetchrow(
            "SELECT id FROM papers WHERE doi = $1",
            doi
        )

        if existing:
            return str(existing['id'])

        # Fetch from Semantic Scholar
        paper_data = await self._fetch_semantic_scholar(doi=doi)

        if not paper_data:
            # Try other sources
            paper_data = await self._fetch_crossref(doi)

        if not paper_data:
            raise ValueError(f"Could not find paper with DOI: {doi}")

        # Insert into database
        paper_id = await self._store_paper(paper_data)

        # Generate and store embedding
        await self._embed_paper(paper_id, paper_data)

        return paper_id

    async def ingest_by_query(
        self,
        query: str,
        max_results: int = 50
    ) -> List[str]:
        """Ingest papers by search query"""

        # Search Semantic Scholar
        results = await self._search_semantic_scholar(query, max_results)

        paper_ids = []
        for paper_data in results:
            # Check if exists
            existing = await self.kb.pg.fetchrow(
                "SELECT id FROM papers WHERE doi = $1",
                paper_data.get('doi')
            )

            if existing:
                paper_ids.append(str(existing['id']))
                continue

            # Store new paper
            paper_id = await self._store_paper(paper_data)
            await self._embed_paper(paper_id, paper_data)

            paper_ids.append(paper_id)

        return paper_ids

    async def _fetch_semantic_scholar(
        self,
        doi: Optional[str] = None,
        paper_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Fetch paper from Semantic Scholar API"""

        base_url = "https://api.semanticscholar.org/graph/v1/paper"

        if doi:
            url = f"{base_url}/DOI:{doi}"
        elif paper_id:
            url = f"{base_url}/{paper_id}"
        else:
            return None

        params = {
            "fields": "title,authors,abstract,year,citationCount,journal,fieldsOfStudy,url,openAccessPdf"
        }

        headers = {}
        if self.ss_api_key:
            headers["x-api-key"] = self.ss_api_key

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)

            if response.status_code != 200:
                return None

            return response.json()

    async def _search_semantic_scholar(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict]:
        """Search Semantic Scholar"""

        url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,citationCount,journal,fieldsOfStudy,url,openAccessPdf"
        }

        headers = {}
        if self.ss_api_key:
            headers["x-api-key"] = self.ss_api_key

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)

            if response.status_code != 200:
                return []

            data = response.json()
            return data.get('data', [])

    async def _fetch_crossref(self, doi: str) -> Optional[Dict]:
        """Fetch paper from Crossref API"""

        url = f"https://api.crossref.org/works/{doi}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url)

            if response.status_code != 200:
                return None

            data = response.json()
            message = data.get('message', {})

            # Convert to standard format
            return {
                "doi": doi,
                "title": message.get('title', [''])[0],
                "authors": [
                    {"name": f"{a.get('given', '')} {a.get('family', '')}".strip()}
                    for a in message.get('author', [])
                ],
                "abstract": message.get('abstract', ''),
                "year": message.get('published-print', {}).get('date-parts', [[None]])[0][0],
                "journal": message.get('container-title', [''])[0],
                "citationCount": message.get('is-referenced-by-count', 0),
                "url": message.get('URL', '')
            }

    async def _store_paper(self, paper_data: Dict) -> str:
        """Store paper in PostgreSQL"""

        # Insert paper
        paper_id = await self.kb.pg.fetchval(
            """
            INSERT INTO papers (
                doi, title, abstract, publication_date,
                journal, citations_count, url, pdf_url
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """,
            paper_data.get('doi'),
            paper_data.get('title'),
            paper_data.get('abstract'),
            f"{paper_data.get('year')}-01-01" if paper_data.get('year') else None,
            paper_data.get('journal'),
            paper_data.get('citationCount', 0),
            paper_data.get('url'),
            paper_data.get('openAccessPdf', {}).get('url') if paper_data.get('openAccessPdf') else None
        )

        # Insert authors
        for author_data in paper_data.get('authors', []):
            author_id = await self.kb.pg.fetchval(
                """
                INSERT INTO authors (name)
                VALUES ($1)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                author_data.get('name')
            )

            await self.kb.pg.execute(
                """
                INSERT INTO paper_authors (paper_id, author_id, author_order)
                VALUES ($1, $2, $3)
                """,
                paper_id,
                author_id,
                author_data.get('order', 0)
            )

        # Insert fields of study
        for field in paper_data.get('fieldsOfStudy', []):
            field_id = await self.kb.pg.fetchval(
                """
                INSERT INTO fields_of_study (name)
                VALUES ($1)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                field
            )

            await self.kb.pg.execute(
                """
                INSERT INTO paper_fields (paper_id, field_id)
                VALUES ($1, $2)
                """,
                paper_id,
                field_id
            )

        return str(paper_id)

    async def _embed_paper(self, paper_id: str, paper_data: Dict):
        """Generate and store embeddings"""

        # Combine title and abstract
        text = f"{paper_data.get('title', '')}. {paper_data.get('abstract', '')}"

        # Generate embedding
        embedding = self.kb.encoder.encode(text).tolist()

        # Store in ChromaDB
        self.kb.collection.add(
            ids=[paper_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "type": "paper",
                "doi": paper_data.get('doi', ''),
                "title": paper_data.get('title', ''),
                "year": paper_data.get('year', 0),
                "citations_count": paper_data.get('citationCount', 0),
                "journal": paper_data.get('journal', ''),
                "created_at": datetime.utcnow().isoformat()
            }]
        )
```

## 📊 Analytics & Insights

```python
class KnowledgeBaseAnalytics:
    """Analytics and insights from knowledge base"""

    def __init__(self, kb_search: KnowledgeBaseSearch):
        self.kb = kb_search

    async def get_trending_topics(
        self,
        time_window_days: int = 90,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify trending research topics"""

        query = """
        SELECT
            f.name as field,
            COUNT(*) as paper_count,
            AVG(p.citations_count) as avg_citations
        FROM papers p
        JOIN paper_fields pf ON p.id = pf.paper_id
        JOIN fields_of_study f ON pf.field_id = f.id
        WHERE p.publication_date >= CURRENT_DATE - INTERVAL ':days days'
        GROUP BY f.name
        ORDER BY paper_count DESC
        LIMIT :limit;
        """

        results = await self.kb.pg.fetch(
            query,
            days=time_window_days,
            limit=top_k
        )

        return [dict(row) for row in results]

    async def get_influential_papers(
        self,
        field: Optional[str] = None,
        min_year: Optional[int] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find most influential papers by citations"""

        where_clauses = []
        params = []
        param_count = 0

        if field:
            param_count += 1
            where_clauses.append(
                f"f.name = ${param_count}"
            )
            params.append(field)

        if min_year:
            param_count += 1
            where_clauses.append(
                f"EXTRACT(YEAR FROM p.publication_date) >= ${param_count}"
            )
            params.append(min_year)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
        SELECT
            p.id,
            p.title,
            p.citations_count,
            p.publication_date,
            p.journal,
            ARRAY_AGG(DISTINCT a.name) as authors
        FROM papers p
        LEFT JOIN paper_authors pa ON p.id = pa.paper_id
        LEFT JOIN authors a ON pa.author_id = a.id
        LEFT JOIN paper_fields pf ON p.id = pf.paper_id
        LEFT JOIN fields_of_study f ON pf.field_id = f.id
        WHERE {where_clause}
        GROUP BY p.id
        ORDER BY p.citations_count DESC
        LIMIT ${param_count + 1};
        """

        params.append(top_k)

        results = await self.kb.pg.fetch(query, *params)

        return [dict(row) for row in results]

    async def analyze_research_gaps(
        self,
        domain: str
    ) -> List[str]:
        """Identify potential research gaps"""

        # Get papers in domain
        papers = await self.kb.semantic_search(
            query=domain,
            filters={"fields": [domain]},
            top_k=100
        )

        # Extract concepts
        all_concepts = []
        for paper in papers:
            # This would use NLP to extract concepts
            # For now, simplified
            concepts = paper.document.metadata.get('keywords', [])
            all_concepts.extend(concepts)

        # Find underexplored concepts (low frequency)
        from collections import Counter
        concept_counts = Counter(all_concepts)

        gaps = [
            concept for concept, count in concept_counts.items()
            if count < 5  # Threshold for "underexplored"
        ]

        return gaps[:20]

    async def get_citation_network(
        self,
        paper_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Build citation network for visualization"""

        query = """
        WITH RECURSIVE citation_tree AS (
            SELECT
                citing_paper_id as source,
                cited_paper_id as target,
                1 as depth
            FROM citations
            WHERE citing_paper_id = $1 OR cited_paper_id = $1

            UNION ALL

            SELECT
                c.citing_paper_id,
                c.cited_paper_id,
                ct.depth + 1
            FROM citations c
            INNER JOIN citation_tree ct ON
                c.citing_paper_id = ct.target OR c.cited_paper_id = ct.source
            WHERE ct.depth < $2
        )
        SELECT DISTINCT
            source,
            target,
            depth
        FROM citation_tree;
        """

        edges = await self.kb.pg.fetch(query, paper_id, depth)

        # Get node details
        node_ids = set()
        for edge in edges:
            node_ids.add(edge['source'])
            node_ids.add(edge['target'])

        nodes = await self.kb.pg.fetch(
            """
            SELECT id, title, citations_count
            FROM papers
            WHERE id = ANY($1)
            """,
            list(node_ids)
        )

        return {
            "nodes": [dict(node) for node in nodes],
            "edges": [dict(edge) for edge in edges]
        }
```

## 🔐 Security & Privacy

```python
class AccessControl:
    """Access control for knowledge base"""

    def __init__(self, pg_connection: Any):
        self.pg = pg_connection

    async def check_access(
        self,
        user_id: str,
        paper_id: str,
        action: str  # "read", "annotate", "share"
    ) -> bool:
        """Check if user has access to paper"""

        # Public papers are readable by all
        is_public = await self.pg.fetchval(
            "SELECT is_public FROM papers WHERE id = $1",
            paper_id
        )

        if is_public and action == "read":
            return True

        # Check user permissions
        has_permission = await self.pg.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM paper_permissions
                WHERE user_id = $1 AND paper_id = $2 AND permission >= $3
            )
            """,
            user_id,
            paper_id,
            self._permission_level(action)
        )

        return has_permission

    def _permission_level(self, action: str) -> int:
        """Map action to permission level"""
        levels = {
            "read": 1,
            "annotate": 2,
            "share": 3
        }
        return levels.get(action, 0)
```

## 📈 Performance Optimization

### Caching Strategy
```python
from functools import lru_cache
import redis.asyncio as redis

class KnowledgeBaseCache:
    """Cache layer for knowledge base"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get_search_results(
        self,
        query_hash: str
    ) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        cached = await self.redis.get(f"search:{query_hash}")
        if cached:
            import pickle
            return pickle.loads(cached)
        return None

    async def cache_search_results(
        self,
        query_hash: str,
        results: List[SearchResult],
        ttl: int = 3600
    ):
        """Cache search results"""
        import pickle
        await self.redis.setex(
            f"search:{query_hash}",
            ttl,
            pickle.dumps(results)
        )
```

### Index Optimization
```sql
-- Full-text search index
CREATE INDEX idx_papers_fts ON papers
USING GIN(to_tsvector('english', title || ' ' || abstract));

-- Composite indexes for common queries
CREATE INDEX idx_papers_field_year ON papers(field_id, publication_date DESC);
CREATE INDEX idx_papers_citations_year ON papers(citations_count DESC, publication_date DESC);
```

## 🧪 Testing

```python
@pytest.mark.asyncio
async def test_semantic_search():
    """Test semantic search functionality"""
    kb = KnowledgeBaseSearch(...)

    results = await kb.semantic_search(
        query="machine learning in drug discovery",
        top_k=10
    )

    assert len(results) > 0
    assert all(r.score > 0.5 for r in results)

@pytest.mark.asyncio
async def test_citation_network():
    """Test citation network generation"""
    analytics = KnowledgeBaseAnalytics(...)

    network = await analytics.get_citation_network(
        paper_id="test-paper-id",
        depth=2
    )

    assert "nodes" in network
    assert "edges" in network
    assert len(network["nodes"]) > 0
```
