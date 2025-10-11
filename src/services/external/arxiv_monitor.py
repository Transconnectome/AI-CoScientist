"""ArXiv API Monitor Service.

Real-time monitoring of ArXiv papers with RSS feed parsing.
Supports automated paper ingestion from multiple categories.
"""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlencode

import httpx
from structlog import get_logger

logger = get_logger(__name__)


class ArXivPaper:
    """ArXiv paper metadata."""

    def __init__(
        self,
        arxiv_id: str,
        title: str,
        abstract: str,
        authors: List[str],
        categories: List[str],
        published: datetime,
        updated: datetime,
        pdf_url: str,
        doi: Optional[str] = None,
    ):
        self.arxiv_id = arxiv_id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.categories = categories
        self.published = published
        self.updated = updated
        self.pdf_url = pdf_url
        self.doi = doi

    def __repr__(self) -> str:
        return f"ArXivPaper(id={self.arxiv_id}, title={self.title[:50]}...)"


class ArXivMonitor:
    """Real-time ArXiv paper monitoring service.

    Monitors ArXiv RSS feeds and API for new papers in specified categories.
    Supports rate limiting and error handling.

    Example:
        monitor = ArXivMonitor()
        papers = await monitor.fetch_recent_papers(
            categories=["cs.AI", "cs.LG"],
            days_back=1,
            max_results=100
        )
    """

    BASE_URL = "http://export.arxiv.org/api/query"
    MAX_RESULTS_PER_REQUEST = 100  # ArXiv API limit

    def __init__(self, timeout: int = 30):
        """Initialize ArXiv monitor.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ArXivMonitor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: tuple) -> None:
        """Async context manager exit."""
        await self.close()

    def _build_query(
        self,
        categories: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """Build ArXiv API search query.

        Args:
            categories: List of ArXiv categories (e.g., ["cs.AI", "cs.LG"])
            start_date: Filter papers after this date
            end_date: Filter papers before this date

        Returns:
            Query string for ArXiv API
        """
        # Build category query: cat:cs.AI OR cat:cs.LG
        if len(categories) == 1:
            query = f"cat:{categories[0]}"
        else:
            query = " OR ".join(f"cat:{cat}" for cat in categories)

        # Add date filter if specified
        if start_date or end_date:
            date_query_parts = []
            if start_date:
                date_str = start_date.strftime("%Y%m%d")
                date_query_parts.append(f"submittedDate:[{date_str}0000 TO ")
            else:
                date_query_parts.append("submittedDate:[* TO ")

            if end_date:
                date_str = end_date.strftime("%Y%m%d")
                date_query_parts.append(f"{date_str}2359]")
            else:
                date_query_parts.append("*]")

            date_query = "".join(date_query_parts)
            query = f"({query}) AND {date_query}"

        return query

    def _parse_entry(self, entry: ET.Element) -> Optional[ArXivPaper]:
        """Parse ArXiv API XML entry.

        Args:
            entry: XML element representing a paper

        Returns:
            ArXivPaper object or None if parsing fails
        """
        try:
            # Namespace for ArXiv API
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            # Extract ArXiv ID from URL
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or id_elem.text is None:
                return None
            arxiv_id = id_elem.text.split("/")[-1]  # Extract ID from URL

            # Title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

            # Abstract
            summary_elem = entry.find("atom:summary", ns)
            abstract = (
                summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
            )

            # Authors
            author_elems = entry.findall("atom:author/atom:name", ns)
            authors = [
                author.text.strip() for author in author_elems if author.text is not None
            ]

            # Categories
            category_elems = entry.findall("atom:category", ns)
            categories = [cat.get("term", "") for cat in category_elems if cat.get("term")]

            # Dates
            published_elem = entry.find("atom:published", ns)
            published = (
                datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
                if published_elem is not None and published_elem.text
                else datetime.now()
            )

            updated_elem = entry.find("atom:updated", ns)
            updated = (
                datetime.fromisoformat(updated_elem.text.replace("Z", "+00:00"))
                if updated_elem is not None and updated_elem.text
                else published
            )

            # PDF URL
            pdf_link = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_link = link.get("href")
                    break

            if not pdf_link:
                # Fallback: construct PDF URL from ID
                pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # DOI (optional)
            doi_elem = entry.find("arxiv:doi", ns)
            doi = doi_elem.text if doi_elem is not None and doi_elem.text else None

            return ArXivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_link,
                doi=doi,
            )

        except Exception as e:
            logger.warning("failed_to_parse_arxiv_entry", error=str(e))
            return None

    async def fetch_recent_papers(
        self,
        categories: List[str],
        days_back: int = 1,
        max_results: int = 100,
    ) -> List[ArXivPaper]:
        """Fetch recent papers from ArXiv.

        Args:
            categories: List of ArXiv categories (e.g., ["cs.AI", "cs.LG", "q-bio.NC"])
            days_back: Number of days to look back
            max_results: Maximum number of results to return

        Returns:
            List of ArXivPaper objects

        Example:
            papers = await monitor.fetch_recent_papers(
                categories=["cs.AI", "cs.LG"],
                days_back=1,
                max_results=100
            )
        """
        if not categories:
            raise ValueError("At least one category must be specified")

        if max_results > 2000:
            logger.warning("max_results_capped", requested=max_results, max_allowed=2000)
            max_results = 2000

        start_date = datetime.now() - timedelta(days=days_back)
        query = self._build_query(categories, start_date=start_date)

        params = {
            "search_query": query,
            "start": 0,
            "max_results": min(max_results, self.MAX_RESULTS_PER_REQUEST),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        papers: List[ArXivPaper] = []
        client = await self._get_client()

        try:
            logger.info(
                "fetching_arxiv_papers",
                categories=categories,
                days_back=days_back,
                max_results=max_results,
            )

            # Fetch papers (may require multiple requests for pagination)
            offset = 0
            while len(papers) < max_results:
                params["start"] = offset

                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                entries = root.findall("atom:entry", ns)
                if not entries:
                    break  # No more results

                for entry in entries:
                    paper = self._parse_entry(entry)
                    if paper:
                        papers.append(paper)

                    if len(papers) >= max_results:
                        break

                # Check if there are more results
                if len(entries) < self.MAX_RESULTS_PER_REQUEST:
                    break  # Last page

                offset += self.MAX_RESULTS_PER_REQUEST

                # Rate limiting: ArXiv requests max 1 request per 3 seconds
                await asyncio.sleep(3)

            logger.info("arxiv_papers_fetched", count=len(papers), categories=categories)
            return papers[:max_results]

        except httpx.HTTPError as e:
            logger.error("arxiv_api_error", error=str(e), categories=categories)
            raise
        except ET.ParseError as e:
            logger.error("arxiv_xml_parse_error", error=str(e))
            raise
        except Exception as e:
            logger.error("arxiv_fetch_error", error=str(e), categories=categories)
            raise

    async def fetch_paper_by_id(self, arxiv_id: str) -> Optional[ArXivPaper]:
        """Fetch a single paper by ArXiv ID.

        Args:
            arxiv_id: ArXiv ID (e.g., "2301.12345")

        Returns:
            ArXivPaper object or None if not found
        """
        params = {"id_list": arxiv_id, "max_results": 1}

        client = await self._get_client()

        try:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entry = root.find("atom:entry", ns)
            if entry is not None:
                return self._parse_entry(entry)

            return None

        except Exception as e:
            logger.error("arxiv_fetch_by_id_error", arxiv_id=arxiv_id, error=str(e))
            return None
