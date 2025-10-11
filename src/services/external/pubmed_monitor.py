"""PubMed E-utilities API Monitor Service.

Real-time monitoring of PubMed papers using NCBI E-utilities API.
Supports automated paper ingestion with advanced search queries.
"""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlencode

import httpx
from structlog import get_logger

logger = get_logger(__name__)


class PubMedPaper:
    """PubMed paper metadata."""

    def __init__(
        self,
        pmid: str,
        title: str,
        abstract: str,
        authors: List[str],
        journal: str,
        pub_date: datetime,
        doi: Optional[str] = None,
        pmc_id: Optional[str] = None,
        mesh_terms: Optional[List[str]] = None,
    ):
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.journal = journal
        self.pub_date = pub_date
        self.doi = doi
        self.pmc_id = pmc_id
        self.mesh_terms = mesh_terms or []

    def __repr__(self) -> str:
        return f"PubMedPaper(pmid={self.pmid}, title={self.title[:50]}...)"


class PubMedMonitor:
    """Real-time PubMed paper monitoring service.

    Uses NCBI E-utilities API (ESearch + EFetch) for paper discovery and retrieval.
    Supports MeSH term queries and date filtering.

    Example:
        monitor = PubMedMonitor(api_key="your_ncbi_api_key")
        papers = await monitor.fetch_recent_papers(
            query="neuroscience[MeSH] AND machine learning",
            days_back=1,
            max_results=100
        )
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # Rate limits
    RATE_LIMIT_WITH_KEY = 10  # requests per second with API key
    RATE_LIMIT_WITHOUT_KEY = 3  # requests per second without API key
    MAX_RESULTS_PER_REQUEST = 500  # EFetch limit

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """Initialize PubMed monitor.

        Args:
            api_key: NCBI API key (optional but recommended for higher rate limits)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._request_delay = (
            1.0 / self.RATE_LIMIT_WITH_KEY if api_key else 1.0 / self.RATE_LIMIT_WITHOUT_KEY
        )

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

    async def __aenter__(self) -> "PubMedMonitor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: tuple) -> None:
        """Async context manager exit."""
        await self.close()

    def _build_date_range(self, days_back: int) -> str:
        """Build PubMed date range query.

        Args:
            days_back: Number of days to look back

        Returns:
            Date range string for PubMed query
        """
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now()

        return f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[pdat]"

    async def _search_pubmed(
        self,
        query: str,
        max_results: int = 100,
        date_range: Optional[str] = None,
    ) -> List[str]:
        """Search PubMed and return list of PMIDs.

        Args:
            query: PubMed search query
            max_results: Maximum number of results
            date_range: Date range filter (e.g., "2025/01/01:2025/01/31[pdat]")

        Returns:
            List of PMIDs
        """
        # Build final query with date filter
        full_query = f"({query})"
        if date_range:
            full_query = f"{full_query} AND {date_range}"

        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub_date",
            "usehistory": "y",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        client = await self._get_client()

        try:
            logger.info("searching_pubmed", query=query, max_results=max_results)

            response = await client.get(self.ESEARCH_URL, params=params)
            response.raise_for_status()

            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

            logger.info("pubmed_search_complete", pmids_found=len(pmids))
            return pmids

        except httpx.HTTPError as e:
            logger.error("pubmed_search_error", error=str(e), query=query)
            raise
        except Exception as e:
            logger.error("pubmed_search_exception", error=str(e), query=query)
            raise

    def _parse_pubmed_article(self, article: ET.Element) -> Optional[PubMedPaper]:
        """Parse PubMed article XML.

        Args:
            article: XML element representing a PubMed article

        Returns:
            PubMedPaper object or None if parsing fails
        """
        try:
            # PMID
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or pmid_elem.text is None:
                return None
            pmid = pmid_elem.text

            # Title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            # Abstract
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                label = abstract_text.get("Label", "")
                text = abstract_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            # Authors
            authors = []
            for author in article.findall(".//Author"):
                last_name_elem = author.find("LastName")
                fore_name_elem = author.find("ForeName")

                last_name = last_name_elem.text if last_name_elem is not None and last_name_elem.text else ""
                fore_name = fore_name_elem.text if fore_name_elem is not None and fore_name_elem.text else ""

                if last_name:
                    full_name = f"{fore_name} {last_name}".strip()
                    authors.append(full_name)

            # Journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None and journal_elem.text else "Unknown"

            # Publication date
            pub_date_elem = article.find(".//PubDate")
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find("Year")
                month_elem = pub_date_elem.find("Month")
                day_elem = pub_date_elem.find("Day")

                year = int(year_elem.text) if year_elem is not None and year_elem.text else datetime.now().year

                # Parse month (can be numeric or name like "Jan")
                month_text = month_elem.text if month_elem is not None and month_elem.text else "1"
                try:
                    month = int(month_text)
                except ValueError:
                    # Month name to number mapping
                    month_map = {
                        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                    }
                    month = month_map.get(month_text[:3], 1)

                day = int(day_elem.text) if day_elem is not None and day_elem.text else 1

                try:
                    pub_date = datetime(year, month, day)
                except ValueError:
                    pub_date = datetime(year, 1, 1)
            else:
                pub_date = datetime.now()

            # DOI
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi" and article_id.text:
                    doi = article_id.text
                    break

            # PMC ID
            pmc_id = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "pmc" and article_id.text:
                    pmc_id = article_id.text
                    break

            # MeSH terms
            mesh_terms = []
            for mesh in article.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            return PubMedPaper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                pub_date=pub_date,
                doi=doi,
                pmc_id=pmc_id,
                mesh_terms=mesh_terms,
            )

        except Exception as e:
            logger.warning("failed_to_parse_pubmed_article", error=str(e))
            return None

    async def _fetch_pubmed_details(self, pmids: List[str]) -> List[PubMedPaper]:
        """Fetch detailed metadata for PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PubMedPaper objects
        """
        if not pmids:
            return []

        papers: List[PubMedPaper] = []
        client = await self._get_client()

        # Process in batches
        for i in range(0, len(pmids), self.MAX_RESULTS_PER_REQUEST):
            batch = pmids[i : i + self.MAX_RESULTS_PER_REQUEST]

            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }

            if self.api_key:
                params["api_key"] = self.api_key

            try:
                logger.info("fetching_pubmed_details", batch_size=len(batch))

                response = await client.get(self.EFETCH_URL, params=params)
                response.raise_for_status()

                # Parse XML
                root = ET.fromstring(response.text)

                for article in root.findall(".//PubmedArticle"):
                    paper = self._parse_pubmed_article(article)
                    if paper:
                        papers.append(paper)

                # Rate limiting
                await asyncio.sleep(self._request_delay)

            except httpx.HTTPError as e:
                logger.error("pubmed_fetch_error", error=str(e), batch_size=len(batch))
                # Continue with next batch
            except ET.ParseError as e:
                logger.error("pubmed_xml_parse_error", error=str(e))
            except Exception as e:
                logger.error("pubmed_fetch_exception", error=str(e))

        return papers

    async def fetch_recent_papers(
        self,
        query: str,
        days_back: int = 1,
        max_results: int = 100,
    ) -> List[PubMedPaper]:
        """Fetch recent papers from PubMed.

        Args:
            query: PubMed search query (supports MeSH terms, field tags, etc.)
                   Examples:
                   - "neuroscience[MeSH] AND machine learning"
                   - "brain imaging[Title/Abstract]"
                   - "deep learning[Title] AND (fMRI OR MRI)"
            days_back: Number of days to look back
            max_results: Maximum number of results to return

        Returns:
            List of PubMedPaper objects

        Example:
            papers = await monitor.fetch_recent_papers(
                query="neuroscience[MeSH] AND machine learning",
                days_back=1,
                max_results=100
            )
        """
        if not query:
            raise ValueError("Query must be specified")

        if max_results > 10000:
            logger.warning("max_results_capped", requested=max_results, max_allowed=10000)
            max_results = 10000

        # Build date range filter
        date_range = self._build_date_range(days_back)

        logger.info(
            "fetching_pubmed_papers",
            query=query,
            days_back=days_back,
            max_results=max_results,
        )

        try:
            # Step 1: Search for PMIDs
            pmids = await self._search_pubmed(query, max_results, date_range)

            if not pmids:
                logger.info("no_pubmed_results", query=query)
                return []

            # Step 2: Fetch detailed metadata
            papers = await self._fetch_pubmed_details(pmids)

            logger.info("pubmed_papers_fetched", count=len(papers), query=query)
            return papers

        except Exception as e:
            logger.error("pubmed_fetch_error", error=str(e), query=query)
            raise

    async def fetch_paper_by_pmid(self, pmid: str) -> Optional[PubMedPaper]:
        """Fetch a single paper by PMID.

        Args:
            pmid: PubMed ID

        Returns:
            PubMedPaper object or None if not found
        """
        papers = await self._fetch_pubmed_details([pmid])
        return papers[0] if papers else None
