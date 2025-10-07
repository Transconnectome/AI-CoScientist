"""Literature-related Pydantic schemas."""

from datetime import date, datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class LiteratureBase(BaseModel):
    """Base literature schema."""
    doi: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    publication_date: Optional[date] = None
    journal: Optional[str] = None
    citations_count: int = 0
    url: Optional[str] = None


class Literature(LiteratureBase):
    """Literature response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime


class LiteratureSearchRequest(BaseModel):
    """Request for literature search."""
    query: str
    top_k: int = 10
    search_type: str = "hybrid"  # semantic, keyword, hybrid
    filters: Optional[dict] = None


class SearchResultSchema(BaseModel):
    """Search result schema."""
    document_id: str
    title: str
    abstract: str
    score: float
    metadata: dict
    highlights: List[str]


class LiteratureIngestRequest(BaseModel):
    """Request to ingest literature."""
    source_type: str  # doi, query
    source_value: str
    max_results: Optional[int] = 50
