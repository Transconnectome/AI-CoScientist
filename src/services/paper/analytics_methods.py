"""Analytics methods for ImprovementService.

These methods implement the analytics dashboard functionality
following TDD principles.
"""
from typing import Dict, List
from collections import defaultdict, Counter


class AnalyticsMethods:
    """Static analytics calculation methods."""

    @staticmethod
    def _calculate_quality_progression(versions: List) -> Dict:
        """Calculate quality progression metrics.

        Args:
            versions: List of PaperVersion objects sorted by creation time

        Returns:
            Dict with progression metrics
        """
        if not versions:
            return {
                'starting_score': 0.0,
                'current_score': 0.0,
                'total_improvement': 0.0,
                'improvement_percentage': 0.0,
                'iterations_count': 0
            }

        # Oldest to newest
        sorted_versions = sorted(versions, key=lambda v: v.created_at)

        starting_score = sorted_versions[0].quality_score or 0.0
        current_score = sorted_versions[-1].quality_score or 0.0
        total_improvement = current_score - starting_score

        # Calculate percentage improvement
        if starting_score > 0:
            improvement_percentage = (total_improvement / starting_score) * 100
        else:
            improvement_percentage = 0.0

        return {
            'starting_score': starting_score,
            'current_score': current_score,
            'total_improvement': total_improvement,
            'improvement_percentage': improvement_percentage,
            'iterations_count': len(sorted_versions)
        }

    @staticmethod
    def _aggregate_section_improvements(improvements: List) -> Dict:
        """Aggregate improvements by section.

        Args:
            improvements: List of ImprovementHistory objects

        Returns:
            Dict mapping section names to stats
        """
        section_stats = defaultdict(lambda: {'count': 0, 'total_impact': 0.0})

        for imp in improvements:
            section_name = imp.section_name
            section_stats[section_name]['count'] += 1
            section_stats[section_name]['total_impact'] += imp.improvement_score

        return dict(section_stats)

    @staticmethod
    def _calculate_type_effectiveness(improvements: List) -> Dict:
        """Calculate effectiveness by improvement type.

        Args:
            improvements: List of ImprovementHistory objects

        Returns:
            Dict mapping improvement types to effectiveness metrics
        """
        type_stats = defaultdict(lambda: {
            'scores': [],
            'count': 0,
            'successful': 0
        })

        for imp in improvements:
            imp_type = imp.improvement_type
            type_stats[imp_type]['count'] += 1
            type_stats[imp_type]['scores'].append(imp.improvement_score)

            # Consider successful if positive improvement
            if imp.improvement_score > 0:
                type_stats[imp_type]['successful'] += 1

        # Calculate averages and success rates
        result = {}
        for imp_type, stats in type_stats.items():
            avg_impact = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0.0
            success_rate = (stats['successful'] / stats['count'] * 100) if stats['count'] > 0 else 0.0

            result[imp_type] = {
                'count': stats['count'],
                'average_impact': avg_impact,
                'success_rate': success_rate
            }

        return result

    @staticmethod
    def _calculate_iteration_efficiency(sessions: List) -> Dict:
        """Calculate iteration efficiency metrics.

        Args:
            sessions: List of IterationSession objects

        Returns:
            Dict with efficiency metrics
        """
        if not sessions:
            return {
                'total_iterations': 0,
                'average_improvements_per_iteration': 0.0,
                'average_score_gain': 0.0,
                'diminishing_returns_detected': False
            }

        # Total iterations across all sessions
        total_iterations = len(sessions)
        total_improvements = sum(s.improvements_applied for s in sessions)
        total_score_gain = sum(s.score_improvement for s in sessions)

        avg_improvements = total_improvements / total_iterations if total_iterations > 0 else 0.0
        avg_score_gain = total_score_gain / total_iterations if total_iterations > 0 else 0.0

        # Detect diminishing returns: check if last iteration had no improvement
        diminishing_returns = False
        if sessions:
            last_session = sorted(sessions, key=lambda s: s.created_at)[-1]
            # Check if last iteration of last session had minimal gain
            if last_session.current_iteration > 1:
                # If score improvement is less than 0.1 in the last iteration, flag diminishing returns
                avg_per_iteration = last_session.score_improvement / last_session.current_iteration
                diminishing_returns = avg_per_iteration < 0.1

        return {
            'total_iterations': total_iterations,
            'average_improvements_per_iteration': avg_improvements,
            'average_score_gain': avg_score_gain,
            'diminishing_returns_detected': diminishing_returns
        }

    @staticmethod
    def _collect_learning_stats(learning_store) -> Dict:
        """Collect learning system statistics.

        Args:
            learning_store: LearningStore instance

        Returns:
            Dict with learning stats
        """
        # Note: This would be async in real implementation
        # For now, return mock structure for TDD
        return {
            'improvement_patterns_count': 0,
            'successful_papers_count': 0,
            'user_interactions_count': 0,
            'rag_usage_count': 0,
            'average_relevance_score': 0.0,
            'exemplars_referenced_count': 0
        }

    @staticmethod
    def _generate_recommendations(analytics_data: Dict) -> List[str]:
        """Generate intelligent recommendations from analytics.

        Args:
            analytics_data: Dict with all analytics data

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Recommendation 1: Best improvement type
        if 'improvement_type_effectiveness' in analytics_data:
            types = analytics_data['improvement_type_effectiveness']
            if types:
                best_type = max(types.items(), key=lambda x: x[1]['average_impact'])
                recommendations.append(
                    f"Focus on {best_type[0]} improvements (highest impact: {best_type[1]['average_impact']:.1f})"
                )

                # Worst performing type
                worst_type = min(types.items(), key=lambda x: x[1]['success_rate'])
                if worst_type[1]['success_rate'] < 50:
                    recommendations.append(
                        f"Avoid {worst_type[0]} improvements (low success rate: {worst_type[1]['success_rate']:.0f}%)"
                    )

        # Recommendation 2: Diminishing returns
        if 'iteration_efficiency' in analytics_data:
            if analytics_data['iteration_efficiency'].get('diminishing_returns_detected'):
                recommendations.append("Stop iterating - diminishing returns detected")

        # Recommendation 3: Section focus
        if 'section_improvements' in analytics_data:
            sections = analytics_data['section_improvements']
            if sections:
                # Find least improved section
                least_improved = min(sections.items(), key=lambda x: x[1]['count'])
                if least_improved[1]['count'] < 5:
                    recommendations.append(
                        f"{least_improved[0]} section needs attention (only {least_improved[1]['count']} improvements)"
                    )

        return recommendations
