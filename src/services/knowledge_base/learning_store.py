"""ChromaDB collections for learning from improvements.

Phase 4: Intelligent learning system that stores and retrieves
successful improvement patterns, high-quality papers, and user preferences.
"""

from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings

from src.core.config import settings as app_settings


class LearningStore:
    """ChromaDB collections for learning from improvements.

    Manages three specialized collections:
    1. improvement_patterns: Successful improvement techniques
    2. successful_papers: High-quality papers for reference
    3. user_history: User interaction patterns and preferences
    """

    def __init__(self):
        """Initialize ChromaDB client and collections."""
        self.client = chromadb.HttpClient(
            host=app_settings.chromadb_host, port=app_settings.chromadb_port
        )

        # Collection 1: Improvement Patterns
        self.improvement_patterns = self.client.get_or_create_collection(
            name="improvement_patterns",
            metadata={
                "description": "Successful improvement patterns and techniques",
                "hnsw:space": "cosine",
            },
        )

        # Collection 2: Successful Papers
        self.successful_papers = self.client.get_or_create_collection(
            name="successful_papers",
            metadata={
                "description": "High-quality papers for reference and learning",
                "hnsw:space": "cosine",
            },
        )

        # Collection 3: User Interaction History
        self.user_history = self.client.get_or_create_collection(
            name="user_history",
            metadata={
                "description": "User preferences and feedback patterns",
                "hnsw:space": "cosine",
            },
        )

    async def store_improvement_pattern(
        self,
        improvement_id: str,
        pattern_type: str,
        original_text: str,
        improved_text: str,
        improvement_score: float,
        metadata: Dict,
    ):
        """Store successful improvement pattern for future learning.

        Args:
            improvement_id: Unique ID for this improvement
            pattern_type: Type of improvement (clarity, coherence, methodology, etc.)
            original_text: Original section content
            improved_text: Improved section content
            improvement_score: Quality score after improvement
            metadata: Additional context (section_name, changes, etc.)
        """
        # Combine original and improved for context-aware retrieval
        pattern_text = f"Original: {original_text}\n\nImproved: {improved_text}"

        self.improvement_patterns.add(
            documents=[pattern_text],
            metadatas=[
                {
                    "pattern_type": pattern_type,
                    "improvement_score": improvement_score,
                    "section_type": metadata.get("section_name", "unknown"),
                    "timestamp": datetime.utcnow().isoformat(),
                    **metadata,
                }
            ],
            ids=[improvement_id],
        )

    async def find_similar_improvements(
        self,
        query_text: str,
        pattern_type: Optional[str] = None,
        n_results: int = 5,
        min_score: float = 7.0,
    ) -> List[Dict]:
        """Find similar successful improvement patterns using RAG.

        Args:
            query_text: Text to find similar improvements for
            pattern_type: Optional filter for improvement type
            n_results: Number of results to return
            min_score: Minimum quality score threshold

        Returns:
            List of similar improvement patterns with metadata
        """
        where_filter = {"improvement_score": {"$gte": min_score}}
        if pattern_type:
            where_filter["pattern_type"] = pattern_type

        results = self.improvement_patterns.query(
            query_texts=[query_text], n_results=n_results, where=where_filter
        )

        return self._format_results(results)

    async def store_successful_paper(
        self, paper_id: str, content: str, quality_scores: Dict[str, float], metadata: Dict
    ):
        """Store high-quality paper for reference.

        Args:
            paper_id: Paper UUID
            content: Full paper content or abstract
            quality_scores: Quality scores (overall, novelty, methodology, clarity)
            metadata: Additional paper metadata
        """
        self.successful_papers.add(
            documents=[content],
            metadatas=[
                {
                    "overall_score": quality_scores.get("overall", 0.0),
                    "novelty_score": quality_scores.get("novelty", 0.0),
                    "methodology_score": quality_scores.get("methodology", 0.0),
                    "clarity_score": quality_scores.get("clarity", 0.0),
                    "timestamp": datetime.utcnow().isoformat(),
                    **metadata,
                }
            ],
            ids=[paper_id],
        )

    async def find_exemplar_papers(
        self, query_text: str, min_quality: float = 8.0, n_results: int = 3
    ) -> List[Dict]:
        """Find high-quality exemplar papers for guidance.

        Args:
            query_text: Text to find similar high-quality papers for
            min_quality: Minimum overall quality score
            n_results: Number of exemplars to return

        Returns:
            List of high-quality papers with metadata
        """
        results = self.successful_papers.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"overall_score": {"$gte": min_quality}},
        )

        return self._format_results(results)

    async def store_user_interaction(
        self,
        interaction_id: str,
        user_id: str,
        action: str,
        context: str,
        feedback: Optional[Dict] = None,
    ):
        """Store user interaction for preference learning.

        Args:
            interaction_id: Unique interaction ID
            user_id: User identifier
            action: Action taken (applied, rejected, modified, etc.)
            context: Context of the interaction
            feedback: Optional user feedback
        """
        self.user_history.add(
            documents=[context],
            metadatas=[
                {
                    "user_id": user_id,
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat(),
                    "feedback": str(feedback or {}),
                }
            ],
            ids=[interaction_id],
        )

    async def get_user_preferences(
        self, user_id: str, n_results: int = 10
    ) -> List[Dict]:
        """Get user's historical preferences and patterns.

        Args:
            user_id: User identifier
            n_results: Number of historical interactions to return

        Returns:
            List of user's past interactions and preferences
        """
        # ChromaDB doesn't support empty query, so use a generic query
        # and filter by user_id in metadata
        results = self.user_history.query(
            query_texts=["user interaction"],
            n_results=n_results,
            where={"user_id": user_id},
        )

        return self._format_results(results)

    async def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all learning collections.

        Returns:
            Dict with count of documents in each collection
        """
        return {
            "improvement_patterns": self.improvement_patterns.count(),
            "successful_papers": self.successful_papers.count(),
            "user_history": self.user_history.count(),
        }

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into clean dict list.

        Args:
            results: Raw ChromaDB query results

        Returns:
            Formatted list of results
        """
        if not results["ids"] or not results["ids"][0]:
            return []

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None
                    ),
                }
            )
        return formatted
