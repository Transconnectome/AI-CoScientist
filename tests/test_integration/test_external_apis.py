"""Integration tests for external API integrations."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from src.services.llm.claude_service import ClaudeService
from src.services.literature.arxiv_fetcher import ArxivFetcher
from src.services.literature.pubmed_fetcher import PubMedFetcher
from src.services.knowledge_base.vector_store import VectorStore


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client."""
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[
                MagicMock(text="Test response from Claude")
            ],
            usage=MagicMock(
                input_tokens=100,
                output_tokens=50
            )
        )
    )
    return client


@pytest.fixture
def claude_service(mock_anthropic_client):
    """Create Claude service with mocked client."""
    service = ClaudeService()
    service.client = mock_anthropic_client
    return service


@pytest.mark.asyncio
class TestClaudeIntegration:
    """Integration tests for Claude API."""

    async def test_generate_text(self, claude_service):
        """Test text generation with Claude."""
        result = await claude_service.generate(
            prompt="Explain quantum computing",
            max_tokens=1000
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_generate_hypothesis(self, claude_service):
        """Test hypothesis generation."""
        literature_summary = """
        Recent studies show neural networks can learn complex patterns.
        Transfer learning improves model performance.
        """

        result = await claude_service.generate_hypothesis(
            literature_summary=literature_summary,
            research_domain="Machine Learning"
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "content" in result
        assert "rationale" in result

    async def test_validate_hypothesis(self, claude_service):
        """Test hypothesis validation."""
        hypothesis = "Transfer learning improves model performance on small datasets"
        literature_context = "Studies show transfer learning is effective"

        result = await claude_service.validate_hypothesis(
            hypothesis=hypothesis,
            literature_context=literature_context
        )

        assert result is not None
        assert "validation_score" in result
        assert 0 <= result["validation_score"] <= 1

    async def test_design_experiment_protocol(self, claude_service):
        """Test experiment protocol design."""
        result = await claude_service.design_experiment(
            research_question="Does transfer learning improve performance?",
            hypothesis="Transfer learning increases accuracy by 10%",
            methodology_context="Use pre-trained models"
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "protocol" in result
        assert "methods" in result

    async def test_error_handling(self, claude_service):
        """Test error handling for API failures."""
        # Mock API error
        claude_service.client.messages.create = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(Exception):
            await claude_service.generate("Test prompt")

    async def test_token_counting(self, claude_service):
        """Test token usage tracking."""
        result = await claude_service.generate(
            prompt="Short test",
            max_tokens=100
        )

        # Verify token usage is tracked
        assert claude_service.client.messages.create.called


@pytest.mark.asyncio
class TestArxivIntegration:
    """Integration tests for ArXiv API."""

    @pytest.fixture
    def arxiv_fetcher(self):
        """Create ArXiv fetcher."""
        return ArxivFetcher()

    async def test_search_papers(self, arxiv_fetcher):
        """Test searching papers on ArXiv."""
        with patch('arxiv.Search') as mock_search:
            # Mock ArXiv search results
            mock_result = MagicMock()
            mock_result.title = "Test Paper"
            mock_result.summary = "Test abstract"
            mock_result.authors = [MagicMock(name="Author 1")]
            mock_result.published = "2024-01-01"
            mock_result.entry_id = "http://arxiv.org/abs/2401.00001"

            mock_search.return_value.results.return_value = [mock_result]

            papers = await arxiv_fetcher.search(
                query="machine learning",
                max_results=5
            )

            assert len(papers) > 0
            assert papers[0]["title"] == "Test Paper"
            assert "abstract" in papers[0]
            assert "authors" in papers[0]

    async def test_get_paper_by_id(self, arxiv_fetcher):
        """Test retrieving specific paper by ArXiv ID."""
        with patch('arxiv.Search') as mock_search:
            mock_result = MagicMock()
            mock_result.title = "Specific Paper"
            mock_result.summary = "Specific abstract"
            mock_result.authors = [MagicMock(name="Author")]
            mock_result.published = "2024-01-01"
            mock_result.entry_id = "http://arxiv.org/abs/2401.00001"

            mock_search.return_value.results.return_value = [mock_result]

            paper = await arxiv_fetcher.get_paper("2401.00001")

            assert paper is not None
            assert paper["title"] == "Specific Paper"

    async def test_filter_by_date(self, arxiv_fetcher):
        """Test filtering papers by publication date."""
        with patch('arxiv.Search') as mock_search:
            # Mock recent and old papers
            recent = MagicMock()
            recent.title = "Recent Paper"
            recent.published = "2024-01-01"

            old = MagicMock()
            old.title = "Old Paper"
            old.published = "2020-01-01"

            mock_search.return_value.results.return_value = [recent, old]

            papers = await arxiv_fetcher.search(
                query="test",
                date_from="2023-01-01"
            )

            # Should only include recent papers
            assert len(papers) > 0

    async def test_handle_no_results(self, arxiv_fetcher):
        """Test handling when no papers are found."""
        with patch('arxiv.Search') as mock_search:
            mock_search.return_value.results.return_value = []

            papers = await arxiv_fetcher.search(
                query="nonexistent query xyz123",
                max_results=5
            )

            assert len(papers) == 0


@pytest.mark.asyncio
class TestPubMedIntegration:
    """Integration tests for PubMed API."""

    @pytest.fixture
    def pubmed_fetcher(self):
        """Create PubMed fetcher."""
        return PubMedFetcher()

    async def test_search_articles(self, pubmed_fetcher):
        """Test searching articles on PubMed."""
        with patch('Bio.Entrez.esearch') as mock_search, \
             patch('Bio.Entrez.efetch') as mock_fetch:

            # Mock search results
            mock_search.return_value = MagicMock()
            mock_search.return_value.read.return_value = {
                'IdList': ['12345678']
            }

            # Mock article fetch
            mock_fetch.return_value = MagicMock()
            mock_fetch.return_value.read.return_value = """
            <PubmedArticle>
                <MedlineCitation>
                    <Article>
                        <ArticleTitle>Test Article</ArticleTitle>
                        <Abstract>Test abstract</Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
            """

            articles = await pubmed_fetcher.search(
                query="cancer research",
                max_results=5
            )

            assert isinstance(articles, list)

    async def test_get_article_details(self, pubmed_fetcher):
        """Test retrieving article details."""
        with patch('Bio.Entrez.efetch') as mock_fetch:
            mock_fetch.return_value = MagicMock()
            mock_fetch.return_value.read.return_value = """
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Detailed Article</ArticleTitle>
                        <Abstract>Detailed abstract</Abstract>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                        </AuthorList>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
            """

            article = await pubmed_fetcher.get_article("12345678")

            assert article is not None

    async def test_filter_by_publication_type(self, pubmed_fetcher):
        """Test filtering by publication type."""
        with patch('Bio.Entrez.esearch') as mock_search:
            mock_search.return_value = MagicMock()
            mock_search.return_value.read.return_value = {
                'IdList': ['12345678', '87654321']
            }

            articles = await pubmed_fetcher.search(
                query="test",
                publication_type="Clinical Trial"
            )

            assert isinstance(articles, list)


@pytest.mark.asyncio
class TestVectorStoreIntegration:
    """Integration tests for vector store (ChromaDB)."""

    @pytest.fixture
    async def vector_store(self):
        """Create vector store instance."""
        store = VectorStore()
        await store.initialize()
        return store

    async def test_store_and_retrieve_embeddings(self, vector_store):
        """Test storing and retrieving embeddings."""
        # Store documents
        documents = [
            {
                "id": str(uuid4()),
                "text": "Neural networks are powerful machine learning models",
                "metadata": {"source": "arxiv", "year": 2024}
            },
            {
                "id": str(uuid4()),
                "text": "Deep learning has revolutionized computer vision",
                "metadata": {"source": "arxiv", "year": 2024}
            }
        ]

        await vector_store.add_documents(documents)

        # Search for similar documents
        results = await vector_store.search(
            query="machine learning models",
            n_results=2
        )

        assert len(results) > 0
        assert "neural networks" in results[0]["text"].lower()

    async def test_semantic_search(self, vector_store):
        """Test semantic similarity search."""
        # Add documents
        docs = [
            {"id": str(uuid4()), "text": "Python is a programming language"},
            {"id": str(uuid4()), "text": "Java is used for software development"},
            {"id": str(uuid4()), "text": "Apples are healthy fruits"}
        ]
        await vector_store.add_documents(docs)

        # Search for programming-related content
        results = await vector_store.search(
            query="coding and programming",
            n_results=3
        )

        # Should prioritize programming docs
        programming_results = [
            r for r in results
            if "python" in r["text"].lower() or "java" in r["text"].lower()
        ]
        assert len(programming_results) > 0

    async def test_filter_by_metadata(self, vector_store):
        """Test filtering search results by metadata."""
        # Add documents with different years
        docs = [
            {
                "id": str(uuid4()),
                "text": "Recent ML research",
                "metadata": {"year": 2024}
            },
            {
                "id": str(uuid4()),
                "text": "Old ML research",
                "metadata": {"year": 2020}
            }
        ]
        await vector_store.add_documents(docs)

        # Search with metadata filter
        results = await vector_store.search(
            query="machine learning",
            filter_metadata={"year": 2024},
            n_results=5
        )

        assert all(r["metadata"]["year"] == 2024 for r in results)

    async def test_update_document(self, vector_store):
        """Test updating existing document."""
        doc_id = str(uuid4())

        # Add document
        await vector_store.add_documents([{
            "id": doc_id,
            "text": "Original text",
            "metadata": {"version": 1}
        }])

        # Update document
        await vector_store.update_document(
            document_id=doc_id,
            text="Updated text",
            metadata={"version": 2}
        )

        # Verify update
        results = await vector_store.search(
            query="updated text",
            n_results=1
        )

        assert len(results) > 0
        assert results[0]["metadata"]["version"] == 2

    async def test_delete_document(self, vector_store):
        """Test deleting document."""
        doc_id = str(uuid4())

        # Add document
        await vector_store.add_documents([{
            "id": doc_id,
            "text": "To be deleted"
        }])

        # Delete document
        await vector_store.delete_document(doc_id)

        # Verify deletion
        results = await vector_store.search(
            query="to be deleted",
            n_results=10
        )

        assert not any(r["id"] == doc_id for r in results)


@pytest.mark.asyncio
class TestAPIRateLimiting:
    """Integration tests for API rate limiting."""

    async def test_claude_rate_limit_handling(self, claude_service):
        """Test handling Claude API rate limits."""
        # Mock rate limit error
        from anthropic import RateLimitError

        claude_service.client.messages.create = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded")
        )

        with pytest.raises(RateLimitError):
            await claude_service.generate("Test prompt")

    async def test_retry_mechanism(self, claude_service):
        """Test retry mechanism for transient failures."""
        call_count = 0

        async def mock_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return MagicMock(
                content=[MagicMock(text="Success after retry")],
                usage=MagicMock(input_tokens=100, output_tokens=50)
            )

        claude_service.client.messages.create = mock_with_retry

        # Should succeed after retries
        result = await claude_service.generate_with_retry(
            prompt="Test",
            max_retries=3
        )

        assert result is not None
        assert call_count == 3


@pytest.mark.asyncio
class TestConcurrentAPIRequests:
    """Integration tests for concurrent API requests."""

    async def test_concurrent_claude_requests(self, claude_service):
        """Test making concurrent Claude API requests."""
        import asyncio

        prompts = [
            "Explain quantum computing",
            "What is machine learning?",
            "Describe neural networks"
        ]

        results = await asyncio.gather(*[
            claude_service.generate(prompt)
            for prompt in prompts
        ])

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    async def test_concurrent_literature_searches(self):
        """Test concurrent literature searches."""
        import asyncio

        arxiv = ArxivFetcher()

        with patch('arxiv.Search') as mock_search:
            mock_result = MagicMock()
            mock_result.title = "Test Paper"
            mock_result.summary = "Abstract"
            mock_result.authors = [MagicMock(name="Author")]
            mock_result.published = "2024-01-01"
            mock_result.entry_id = "http://arxiv.org/abs/2401.00001"

            mock_search.return_value.results.return_value = [mock_result]

            queries = ["quantum", "neural", "optimization"]

            results = await asyncio.gather(*[
                arxiv.search(query, max_results=5)
                for query in queries
            ])

            assert len(results) == 3
            assert all(isinstance(r, list) for r in results)


@pytest.mark.asyncio
class TestErrorRecovery:
    """Integration tests for error recovery."""

    async def test_network_timeout_recovery(self, claude_service):
        """Test recovery from network timeouts."""
        import asyncio

        async def mock_timeout(*args, **kwargs):
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError("Request timeout")

        claude_service.client.messages.create = mock_timeout

        with pytest.raises(asyncio.TimeoutError):
            await claude_service.generate("Test", timeout=0.05)

    async def test_invalid_response_handling(self, claude_service):
        """Test handling of invalid API responses."""
        # Mock invalid response
        claude_service.client.messages.create = AsyncMock(
            return_value=MagicMock(content=None)
        )

        with pytest.raises(ValueError):
            await claude_service.generate("Test prompt")
