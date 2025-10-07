"""External API services."""

from src.services.external.semantic_scholar import SemanticScholarClient
from src.services.external.crossref import CrossRefClient

__all__ = [
    "SemanticScholarClient",
    "CrossRefClient"
]
