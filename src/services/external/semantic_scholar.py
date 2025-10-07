"""Semantic Scholar API client."""

from typing import Any, Dict, List, Optional
import httpx

from src.core.config import settings


class SemanticScholarClient:
    """Client for Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Semantic Scholar client."""
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    async def get_paper(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get paper by ID or DOI."""
        if not fields:
            fields = [
                "title", "authors", "abstract", "year",
                "citationCount", "journal", "fieldsOfStudy",
                "url", "openAccessPdf"
            ]

        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": ",".join(fields)}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params=params,
                headers=self.headers,
                timeout=30.0
            )

            if response.status_code != 200:
                return None

            return response.json()

    async def search_papers(
        self,
        query: str,
        limit: int = 50,
        fields: Optional[List[str]] = None,
        year: Optional[str] = None,
        publication_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search papers by query."""
        if not fields:
            fields = [
                "title", "authors", "abstract", "year",
                "citationCount", "journal", "fieldsOfStudy",
                "url", "openAccessPdf"
            ]

        url = f"{self.BASE_URL}/paper/search"
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100
            "fields": ",".join(fields)
        }

        if year:
            params["year"] = year

        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params=params,
                headers=self.headers,
                timeout=30.0
            )

            if response.status_code != 200:
                return []

            data = response.json()
            return data.get("data", [])

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get papers that cite this paper."""
        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        params = {
            "limit": min(limit, 1000),
            "fields": "title,authors,year,citationCount"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params=params,
                headers=self.headers,
                timeout=30.0
            )

            if response.status_code != 200:
                return []

            data = response.json()
            return data.get("data", [])

    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get papers referenced by this paper."""
        url = f"{self.BASE_URL}/paper/{paper_id}/references"
        params = {
            "limit": min(limit, 1000),
            "fields": "title,authors,year,citationCount"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params=params,
                headers=self.headers,
                timeout=30.0
            )

            if response.status_code != 200:
                return []

            data = response.json()
            return data.get("data", [])
