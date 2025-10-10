"""Improvement service for intelligent paper enhancement with learning.

Phase 4: Orchestrates version tracking, one-click improvements,
iterative optimization, and ChromaDB-based learning.
"""

from typing import Dict, List, Optional
from uuid import UUID
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from difflib import unified_diff

from src.models.project import Paper, PaperSection
from src.models.paper_version import (
    PaperVersion,
    ImprovementHistory,
    IterationSession,
    VersionType,
    ImprovementStatus,
)
from src.services.paper import PaperAnalyzer, PaperImprover
from src.services.knowledge_base.learning_store import LearningStore
from src.services.llm.service import LLMService
from src.core.config import settings


class ImprovementService:
    """Orchestrates intelligent paper improvement with learning.

    Integrates version tracking, one-click improvements, iterative optimization,
    and ChromaDB-based pattern learning for continuous quality improvement.
    """

    def __init__(self, db: AsyncSession):
        """Initialize improvement service.

        Args:
            db: Database session for persistence
        """
        self.db = db
        self.learning_store = LearningStore()
        self.llm = LLMService(
            primary_provider=settings.llm_primary_provider,
            fallback_provider=settings.llm_fallback_provider,
        )
        self.analyzer = PaperAnalyzer(self.llm, db)
        self.improver = PaperImprover(self.llm, db)

    async def apply_improvement(
        self,
        paper_id: UUID,
        section_name: str,
        improved_content: str,
        improvement_metadata: Dict,
    ) -> Dict:
        """Apply improvement and create new version.

        Workflow:
        1. Get current paper and section
        2. Assess quality before/after
        3. Create version snapshot BEFORE applying
        4. Apply improvement to section
        5. Increment paper version
        6. Record improvement history
        7. Store in ChromaDB for learning

        Args:
            paper_id: Paper UUID
            section_name: Section to improve
            improved_content: New improved content
            improvement_metadata: Dict with type, changes, summary

        Returns:
            Dict with success status, new version, quality improvement
        """
        # 1. Get current paper and section
        paper = await self._get_paper(paper_id)
        section = await self._get_section(paper_id, section_name)
        original_content = section.content

        # 2. Analyze quality before/after
        quality_before = await self._assess_section_quality(original_content)
        quality_after = await self._assess_section_quality(improved_content)

        # 3. Create version snapshot BEFORE applying changes
        version = await self._create_version_snapshot(
            paper=paper,
            version_type=VersionType.PATCH,
            change_summary=f"Improved {section_name}: {improvement_metadata.get('summary', 'AI improvement')}",
        )

        # 4. Apply improvement to section
        section.content = improved_content
        section.version += 1

        # 5. Increment paper version (patch level)
        paper.version_patch += 1
        paper.version += 1  # Legacy version field

        # 6. Record improvement history
        improvement_record = ImprovementHistory(
            paper_id=paper_id,
            version_id=version.id,
            section_name=section_name,
            original_content=original_content,
            improved_content=improved_content,
            improvement_type=improvement_metadata.get("type", "general"),
            changes_made=improvement_metadata.get("changes", []),
            status=ImprovementStatus.APPLIED.value,
            quality_before=quality_before,
            quality_after=quality_after,
            improvement_score=quality_after - quality_before,
        )
        self.db.add(improvement_record)

        # 7. Store in ChromaDB for learning (if successful)
        if quality_after > quality_before:
            await self.learning_store.store_improvement_pattern(
                improvement_id=str(improvement_record.id),
                pattern_type=improvement_metadata.get("type", "general"),
                original_text=original_content,
                improved_text=improved_content,
                improvement_score=quality_after,
                metadata={
                    "section_name": section_name,
                    "paper_id": str(paper_id),
                    "changes": improvement_metadata.get("changes", []),
                },
            )

        await self.db.commit()
        await self.db.refresh(improvement_record)

        return {
            "success": True,
            "new_version": version.version_string,
            "quality_improvement": quality_after - quality_before,
            "section_updated": section_name,
            "improvement_id": str(improvement_record.id),
        }

    async def rollback_to_version(
        self, paper_id: UUID, target_version: str, create_backup: bool = True
    ) -> Dict:
        """Rollback paper to a previous version.

        Creates new version (doesn't delete history) with content
        from specified previous version.

        Args:
            paper_id: Paper UUID
            target_version: Target version string (e.g., "1.2.0")
            create_backup: Whether to create backup of current version

        Returns:
            Dict with rollback status and new version info
        """
        paper = await self._get_paper(paper_id)

        # Parse target version
        try:
            major, minor, patch = map(int, target_version.split("."))
        except ValueError:
            raise ValueError(f"Invalid version format: {target_version}. Expected 'major.minor.patch'")

        # Get target version snapshot
        query = select(PaperVersion).where(
            PaperVersion.paper_id == paper_id,
            PaperVersion.major == major,
            PaperVersion.minor == minor,
            PaperVersion.patch == patch,
        )
        result = await self.db.execute(query)
        target_version_obj = result.scalar_one_or_none()

        if not target_version_obj:
            raise ValueError(f"Version {target_version} not found for paper {paper_id}")

        # Create backup of current state if requested
        if create_backup:
            await self._create_version_snapshot(
                paper=paper,
                version_type=VersionType.MAJOR,
                change_summary=f"Backup before rollback to {target_version}",
            )

        # Apply rollback: restore content and sections
        paper.content = target_version_obj.content_snapshot

        # Restore each section
        for section_name, content in target_version_obj.sections_snapshot.items():
            section = await self._get_section(paper_id, section_name)
            section.content = content
            section.version += 1

        # Increment version (major change for rollback)
        paper.version_major += 1
        paper.version_minor = 0
        paper.version_patch = 0
        paper.version += 1

        # Create version record for rollback
        rollback_version = await self._create_version_snapshot(
            paper=paper,
            version_type=VersionType.MAJOR,
            change_summary=f"Rolled back to version {target_version}",
        )

        await self.db.commit()

        return {
            "success": True,
            "rolled_back_from": paper.current_version,
            "rolled_back_to": target_version,
            "new_version": rollback_version.version_string,
            "backup_created": create_backup,
        }

    async def compare_versions(
        self, paper_id: UUID, version_a: str, version_b: str
    ) -> Dict:
        """Compare two versions with diff visualization.

        Args:
            paper_id: Paper UUID
            version_a: First version string (e.g., "1.0.0")
            version_b: Second version string (e.g., "1.2.0")

        Returns:
            Dict with comparison data and diffs
        """
        # Parse version strings
        major_a, minor_a, patch_a = map(int, version_a.split("."))
        major_b, minor_b, patch_b = map(int, version_b.split("."))

        # Get version snapshots
        query_a = select(PaperVersion).where(
            PaperVersion.paper_id == paper_id,
            PaperVersion.major == major_a,
            PaperVersion.minor == minor_a,
            PaperVersion.patch == patch_a,
        )
        query_b = select(PaperVersion).where(
            PaperVersion.paper_id == paper_id,
            PaperVersion.major == major_b,
            PaperVersion.minor == minor_b,
            PaperVersion.patch == patch_b,
        )

        result_a = await self.db.execute(query_a)
        result_b = await self.db.execute(query_b)

        version_obj_a = result_a.scalar_one_or_none()
        version_obj_b = result_b.scalar_one_or_none()

        if not version_obj_a or not version_obj_b:
            raise ValueError("One or both versions not found")

        # Generate diffs for each section
        section_diffs = []
        all_section_names = set(version_obj_a.sections_snapshot.keys()) | set(
            version_obj_b.sections_snapshot.keys()
        )

        for section_name in sorted(all_section_names):
            content_a = version_obj_a.sections_snapshot.get(section_name, "")
            content_b = version_obj_b.sections_snapshot.get(section_name, "")

            diff = list(
                unified_diff(
                    content_a.splitlines(keepends=True),
                    content_b.splitlines(keepends=True),
                    fromfile=f"{section_name} ({version_a})",
                    tofile=f"{section_name} ({version_b})",
                )
            )

            changes_count = len(
                [line for line in diff if line.startswith("+") or line.startswith("-")]
            )

            section_diffs.append(
                {
                    "section_name": section_name,
                    "diff": "".join(diff),
                    "changes_count": changes_count,
                }
            )

        return {
            "version_a": version_a,
            "version_b": version_b,
            "quality_score_a": version_obj_a.quality_score,
            "quality_score_b": version_obj_b.quality_score,
            "quality_change": (version_obj_b.quality_score or 0)
            - (version_obj_a.quality_score or 0),
            "section_diffs": section_diffs,
            "summary_a": version_obj_a.change_summary,
            "summary_b": version_obj_b.change_summary,
        }

    async def get_version_history(self, paper_id: UUID) -> Dict:
        """Get complete version history for a paper.

        Args:
            paper_id: Paper UUID

        Returns:
            Dict with version history and current version
        """
        paper = await self._get_paper(paper_id)

        # Get all versions ordered by creation time (newest first)
        query = (
            select(PaperVersion)
            .where(PaperVersion.paper_id == paper_id)
            .order_by(desc(PaperVersion.created_at))
        )
        result = await self.db.execute(query)
        versions = result.scalars().all()

        version_list = [
            {
                "version_string": v.version_string,
                "version_type": v.version_type,
                "change_summary": v.change_summary,
                "quality_score": v.quality_score,
                "created_at": v.created_at.isoformat(),
            }
            for v in versions
        ]

        return {
            "paper_id": str(paper_id),
            "current_version": paper.current_version,
            "versions": version_list,
            "total_versions": len(version_list),
        }

    async def generate_smart_suggestions(
        self, paper_id: UUID, section_name: Optional[str] = None
    ) -> Dict:
        """Generate RAG-powered suggestions using historical patterns.

        Uses ChromaDB to find similar successful improvements and generates
        context-aware suggestions enhanced with exemplar papers.

        Args:
            paper_id: Paper UUID
            section_name: Optional specific section (None = all sections)

        Returns:
            Dict with smart suggestions and RAG metadata
        """
        # Determine sections to process
        if section_name:
            sections = [await self._get_section(paper_id, section_name)]
        else:
            sections = await self._get_all_sections(paper_id)

        suggestions = []

        for section in sections:
            # Find similar successful improvements from ChromaDB
            similar_patterns = await self.learning_store.find_similar_improvements(
                query_text=section.content, n_results=5, min_score=7.0
            )

            # Find exemplar papers for context
            exemplars = await self.learning_store.find_exemplar_papers(
                query_text=section.content, min_quality=8.0, n_results=2
            )

            # Build RAG context from ChromaDB results
            rag_context = self._build_rag_context(similar_patterns, exemplars)

            # Generate improvement with RAG context
            improvement = await self.improver.improve_section(
                paper_id=paper_id, section_name=section.name, feedback=rag_context
            )

            suggestions.append(
                {
                    "section_name": section.name,
                    "improved_content": improvement["improved_content"],
                    "metadata": {
                        "type": improvement.get("type", "general"),
                        "changes": improvement.get("changes", []),
                        "summary": improvement["changes_summary"],
                        "similar_patterns_used": len(similar_patterns),
                        "exemplars_referenced": len(exemplars),
                    },
                    "expected_improvement": improvement["improvement_score"],
                }
            )

        return {
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "rag_enhanced": True,
        }

    async def run_iterative_improvement(
        self,
        paper_id: UUID,
        target_score: float,
        max_iterations: int = 5,
        focus_areas: Optional[List[str]] = None,
    ) -> Dict:
        """Run iterative improvement loop until target reached.

        Executes multiple rounds of:
        1. Analyze current quality
        2. Generate smart suggestions (RAG-enhanced)
        3. Apply top improvements
        4. Re-assess quality
        5. Continue until target or max iterations

        Args:
            paper_id: Paper UUID
            target_score: Target quality score (0-10)
            max_iterations: Maximum improvement rounds
            focus_areas: Optional focus areas (clarity, methodology, etc.)

        Returns:
            Dict with iteration results and final metrics
        """
        paper = await self._get_paper(paper_id)

        # Get initial quality score
        initial_score = await self._get_overall_quality(paper_id)

        # Create iteration session
        session = IterationSession(
            paper_id=paper_id,
            target_score=target_score,
            max_iterations=max_iterations,
            focus_areas=focus_areas or ["clarity", "coherence", "methodology"],
            current_score=initial_score,
            current_iteration=0,
            is_complete=False,
            improvements_applied=0,
            score_improvement=0.0,
        )
        self.db.add(session)
        await self.db.flush()

        # Store start version
        start_version = await self._create_version_snapshot(
            paper=paper,
            version_type=VersionType.MINOR,
            change_summary=f"Start iterative improvement session (target: {target_score})",
        )
        session.start_version_id = start_version.id

        iterations_completed = 0
        improvements_applied = 0
        current_score = initial_score

        while iterations_completed < max_iterations and current_score < target_score:
            iterations_completed += 1

            # 1. Analyze current state
            analysis = await self.analyzer.analyze_quality(paper_id)

            # 2. Generate smart suggestions using RAG
            suggestions_result = await self.generate_smart_suggestions(paper_id=paper_id)
            suggestions = suggestions_result["suggestions"]

            # 3. Apply top 3 improvements per iteration
            for suggestion in suggestions[:3]:
                try:
                    await self.apply_improvement(
                        paper_id=paper_id,
                        section_name=suggestion["section_name"],
                        improved_content=suggestion["improved_content"],
                        improvement_metadata=suggestion["metadata"],
                    )
                    improvements_applied += 1
                except Exception as e:
                    # Log but continue with other improvements
                    print(f"Failed to apply improvement: {e}")
                    continue

            # 4. Re-assess quality
            current_score = await self._get_overall_quality(paper_id)

            # 5. Update session progress
            session.current_iteration = iterations_completed
            session.current_score = current_score
            session.improvements_applied = improvements_applied
            session.score_improvement = current_score - initial_score

            # Store current version
            current_version = await self._create_version_snapshot(
                paper=paper,
                version_type=VersionType.MINOR,
                change_summary=f"Iteration {iterations_completed}: score={current_score:.2f}",
            )
            session.current_version_id = current_version.id

            await self.db.commit()

            # Early exit if target reached
            if current_score >= target_score:
                break

        # Mark session complete
        session.is_complete = True
        await self.db.commit()

        return {
            "session_id": str(session.id),
            "iterations_completed": iterations_completed,
            "improvements_applied": improvements_applied,
            "initial_score": initial_score,
            "final_score": current_score,
            "score_improvement": current_score - initial_score,
            "target_reached": current_score >= target_score,
            "start_version": start_version.version_string,
            "final_version": paper.current_version,
        }

    def _build_rag_context(
        self, similar_patterns: List[Dict], exemplars: List[Dict]
    ) -> str:
        """Build RAG context from ChromaDB results.

        Args:
            similar_patterns: Similar improvement patterns from ChromaDB
            exemplars: High-quality exemplar papers

        Returns:
            Formatted RAG context string for LLM
        """
        context = "Reference successful improvements:\n\n"

        for i, pattern in enumerate(similar_patterns[:3], 1):
            metadata = pattern.get("metadata", {})
            changes = metadata.get("changes", ["Improved quality"])
            context += f"{i}. {', '.join(changes)}\n"

        if exemplars:
            context += "\n\nHigh-quality examples for reference:\n"
            for i, exemplar in enumerate(exemplars[:2], 1):
                metadata = exemplar.get("metadata", {})
                score = metadata.get("overall_score", 0.0)
                context += f"{i}. Quality score: {score:.1f}/10\n"

        return context

    # ========== Helper Methods ==========

    async def _get_paper(self, paper_id: UUID) -> Paper:
        """Get paper by ID or raise error."""
        result = await self.db.execute(select(Paper).where(Paper.id == paper_id))
        paper = result.scalar_one_or_none()
        if not paper:
            raise ValueError(f"Paper {paper_id} not found")
        return paper

    async def _get_section(self, paper_id: UUID, section_name: str) -> PaperSection:
        """Get section by name or raise error."""
        query = select(PaperSection).where(
            PaperSection.paper_id == paper_id, PaperSection.name == section_name
        )
        result = await self.db.execute(query)
        section = result.scalar_one_or_none()
        if not section:
            raise ValueError(f"Section '{section_name}' not found in paper {paper_id}")
        return section

    async def _get_all_sections(self, paper_id: UUID) -> List[PaperSection]:
        """Get all sections for a paper."""
        query = (
            select(PaperSection)
            .where(PaperSection.paper_id == paper_id)
            .order_by(PaperSection.order)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def _create_version_snapshot(
        self, paper: Paper, version_type: VersionType, change_summary: str
    ) -> PaperVersion:
        """Create version snapshot before changes.

        Args:
            paper: Paper object
            version_type: Type of version change
            change_summary: Description of changes

        Returns:
            Created PaperVersion object
        """
        # Get all sections
        sections = await self._get_all_sections(paper.id)
        sections_snapshot = {section.name: section.content for section in sections}

        # Get overall quality score
        quality_score = await self._get_overall_quality(paper.id)

        version = PaperVersion(
            paper_id=paper.id,
            major=paper.version_major,
            minor=paper.version_minor,
            patch=paper.version_patch,
            content_snapshot=paper.content or "",
            sections_snapshot=sections_snapshot,
            version_type=version_type.value,
            change_summary=change_summary,
            quality_score=quality_score,
        )

        self.db.add(version)
        await self.db.flush()  # Get ID without committing
        return version

    async def _assess_section_quality(self, content: str) -> float:
        """Quick quality assessment for a section.

        Args:
            content: Section content

        Returns:
            Quality score (0-10)
        """
        # Simplified heuristic-based scoring
        # In production, could use actual LLM-based scoring
        score = 5.0  # Base score

        # Length-based adjustment (reasonable length is good)
        if 100 < len(content) < 2000:
            score += 1.0
        elif len(content) < 50:
            score -= 1.0

        # Structure indicators (has paragraphs)
        if "\n\n" in content:
            score += 0.5

        # Citation indicators
        if any(indicator in content for indicator in ["et al.", "Figure", "Table", "Equation"]):
            score += 0.5

        return min(10.0, max(0.0, score))

    async def _get_overall_quality(self, paper_id: UUID) -> float:
        """Get overall paper quality score.

        Args:
            paper_id: Paper UUID

        Returns:
            Overall quality score (0-10)
        """
        try:
            analysis = await self.analyzer.analyze_quality(paper_id)
            return analysis.get("quality_score", 5.0)
        except Exception:
            # Fallback if analyzer fails
            return 5.0
