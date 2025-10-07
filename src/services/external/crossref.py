"""CrossRef API client."""

from typing import Any, Dict, Optional
import httpx

from src.core.config import settings


class CrossRefClient:
    """Client for CrossRef API."""

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: Optional[str] = None):
        """Initialize CrossRef client."""
        self.email = email or settings.crossref_email
        self.headers = {}
        if self.email:
            self.headers["User-Agent"] = f"AI-CoScientist/0.1 (mailto:{self.email})"

    async def get_work(self, doi: str) -> Optional[Dict[str, Any]]:
        """Get work (paper) by DOI."""
        url = f"{self.BASE_URL}/works/{doi}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=self.headers,
                timeout=30.0
            )

            if response.status_code != 200:
                return None

            data = response.json()
            return self._parse_work(data.get("message", {}))

    def _parse_work(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CrossRef work to standard format."""
        # Extract authors
        authors = []
        for author in work.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            authors.append({"name": name})

        # Extract publication date
        published = work.get("published-print", work.get("published-online", {}))
        date_parts = published.get("date-parts", [[None]])[0]
        year = date_parts[0] if date_parts else None

        # Extract journal
        journal = ""
        container_title = work.get("container-title", [])
        if container_title:
            journal = container_title[0]

        return {
            "doi": work.get("DOI"),
            "title": work.get("title", [""])[0],
            "authors": authors,
            "abstract": work.get("abstract", ""),
            "year": year,
            "journal": journal,
            "citationCount": work.get("is-referenced-by-count", 0),
            "url": work.get("URL", ""),
            "volume": work.get("volume"),
            "issue": work.get("issue"),
            "pages": work.get("page")
        }
