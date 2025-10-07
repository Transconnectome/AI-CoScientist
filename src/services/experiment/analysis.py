"""Data analysis service."""

import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.models.project import Experiment, ExperimentStatus
from src.services.llm import LLMService


class DataAnalyzer:
    """Service for statistical data analysis and visualization."""

    def __init__(self, llm_service: LLMService, db: AsyncSession):
        """Initialize data analyzer.

        Args:
            llm_service: LLM service for interpretation
            db: Database session
        """
        self.llm = llm_service
        self.db = db

    async def analyze_experiment_data(
        self,
        experiment_id: UUID,
        data: Dict[str, Any],
        analysis_types: List[str] = None,
        visualization_types: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze experimental data.

        Args:
            experiment_id: Experiment UUID
            data: Experimental data (flexible structure)
            analysis_types: Types of analyses to perform
            visualization_types: Types of visualizations to generate

        Returns:
            Analysis results with statistics and visualizations
        """
        if analysis_types is None:
            analysis_types = ["descriptive", "inferential", "effect_size"]
        if visualization_types is None:
            visualization_types = ["distribution", "comparison"]

        # Load experiment
        result = await self.db.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = result.scalar_one_or_none()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Parse data into DataFrame
        df = self._parse_data(data)

        # Perform analyses
        analysis_results = {}

        if "descriptive" in analysis_types:
            analysis_results["descriptive"] = self._descriptive_statistics(df)

        if "inferential" in analysis_types:
            analysis_results["inferential"] = self._inferential_tests(
                df, experiment.significance_level
            )

        if "effect_size" in analysis_types:
            analysis_results["effect_size"] = self._calculate_effect_sizes(df)

        # Generate visualizations
        visualizations = []
        if "distribution" in visualization_types:
            viz = self._plot_distributions(df)
            visualizations.append(viz)

        if "comparison" in visualization_types:
            viz = self._plot_comparisons(df)
            visualizations.append(viz)

        if "correlation" in visualization_types:
            viz = self._plot_correlations(df)
            visualizations.append(viz)

        # Generate interpretation using LLM
        interpretation = await self._generate_interpretation(
            analysis_results, experiment
        )

        # Store results
        experiment.statistical_results = json.dumps(analysis_results)
        experiment.visualization_urls = json.dumps(
            [v["url"] for v in visualizations if v]
        )
        experiment.interpretation = interpretation
        experiment.status = ExperimentStatus.COMPLETED.value

        await self.db.commit()
        await self.db.refresh(experiment)

        return {
            "experiment_id": experiment_id,
            "descriptive_statistics": analysis_results.get("descriptive", {}),
            "statistical_tests": analysis_results.get("inferential", []),
            "effect_sizes": analysis_results.get("effect_size", {}),
            "visualizations": visualizations,
            "interpretation": interpretation,
            "recommendations": self._generate_recommendations(analysis_results)
        }

    def _parse_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse data into DataFrame.

        Args:
            data: Input data (dict or list format)

        Returns:
            Pandas DataFrame
        """
        if isinstance(data, dict):
            # Convert dict to DataFrame
            if "records" in data:
                return pd.DataFrame(data["records"])
            else:
                return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError("Data must be dict or list")

    def _descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics.

        Args:
            df: Data DataFrame

        Returns:
            Descriptive statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats_dict = {}
        for col in numeric_cols:
            stats_dict[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "count": int(df[col].count())
            }

        return stats_dict

    def _inferential_tests(
        self,
        df: pd.DataFrame,
        alpha: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Perform inferential statistical tests.

        Args:
            df: Data DataFrame
            alpha: Significance level

        Returns:
            List of test results
        """
        results = []

        # Identify groups if 'group' column exists
        if "group" in df.columns:
            groups = df["group"].unique()
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if len(groups) == 2:
                    # Two-sample t-test
                    group1 = df[df["group"] == groups[0]][col].dropna()
                    group2 = df[df["group"] == groups[1]][col].dropna()

                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1, group2)

                        results.append({
                            "test_name": "Independent t-test",
                            "variable": col,
                            "groups": [str(groups[0]), str(groups[1])],
                            "statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": p_value < alpha,
                            "interpretation": (
                                f"{'Significant' if p_value < alpha else 'Not significant'} "
                                f"difference between groups (p={p_value:.4f})"
                            )
                        })

                elif len(groups) > 2:
                    # One-way ANOVA
                    group_data = [
                        df[df["group"] == g][col].dropna()
                        for g in groups
                    ]

                    if all(len(g) > 0 for g in group_data):
                        f_stat, p_value = stats.f_oneway(*group_data)

                        results.append({
                            "test_name": "One-way ANOVA",
                            "variable": col,
                            "groups": [str(g) for g in groups],
                            "statistic": float(f_stat),
                            "p_value": float(p_value),
                            "significant": p_value < alpha,
                            "interpretation": (
                                f"{'Significant' if p_value < alpha else 'Not significant'} "
                                f"difference among groups (p={p_value:.4f})"
                            )
                        })

        return results

    def _calculate_effect_sizes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate effect sizes.

        Args:
            df: Data DataFrame

        Returns:
            Effect size calculations
        """
        effect_sizes = {}

        if "group" in df.columns:
            groups = df["group"].unique()
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if len(groups) == 2:
                    group1 = df[df["group"] == groups[0]][col].dropna()
                    group2 = df[df["group"] == groups[1]][col].dropna()

                    if len(group1) > 0 and len(group2) > 0:
                        # Cohen's d
                        pooled_std = np.sqrt(
                            ((len(group1) - 1) * group1.std() ** 2 +
                             (len(group2) - 1) * group2.std() ** 2) /
                            (len(group1) + len(group2) - 2)
                        )

                        if pooled_std > 0:
                            cohens_d = (group1.mean() - group2.mean()) / pooled_std

                            effect_sizes[col] = {
                                "cohens_d": float(cohens_d),
                                "magnitude": self._interpret_cohens_d(cohens_d)
                            }

        return effect_sizes

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _plot_distributions(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Plot data distributions.

        Args:
            df: Data DataFrame

        Returns:
            Visualization info or None
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return None

        fig, axes = plt.subplots(
            nrows=(len(numeric_cols) + 1) // 2,
            ncols=2,
            figsize=(12, 4 * ((len(numeric_cols) + 1) // 2))
        )

        if len(numeric_cols) == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            if "group" in df.columns:
                for group in df["group"].unique():
                    data = df[df["group"] == group][col].dropna()
                    axes[idx].hist(data, alpha=0.6, label=str(group), bins=20)
                axes[idx].legend()
            else:
                axes[idx].hist(df[col].dropna(), bins=20)

            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
            axes[idx].set_title(f"Distribution of {col}")

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()

        return {
            "visualization_type": "distribution",
            "url": f"data:image/png;base64,{img_str}",
            "description": "Distribution plots for all numeric variables",
            "format": "png"
        }

    def _plot_comparisons(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Plot group comparisons.

        Args:
            df: Data DataFrame

        Returns:
            Visualization info or None
        """
        if "group" not in df.columns:
            return None

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return None

        fig, axes = plt.subplots(
            nrows=(len(numeric_cols) + 1) // 2,
            ncols=2,
            figsize=(12, 4 * ((len(numeric_cols) + 1) // 2))
        )

        if len(numeric_cols) == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            sns.boxplot(data=df, x="group", y=col, ax=axes[idx])
            axes[idx].set_title(f"Comparison of {col} by Group")

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()

        return {
            "visualization_type": "comparison",
            "url": f"data:image/png;base64,{img_str}",
            "description": "Box plots comparing groups",
            "format": "png"
        }

    def _plot_correlations(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Plot correlation matrix.

        Args:
            df: Data DataFrame

        Returns:
            Visualization info or None
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return None

        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Matrix")
        plt.tight_layout()

        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()

        return {
            "visualization_type": "correlation",
            "url": f"data:image/png;base64,{img_str}",
            "description": "Correlation matrix heatmap",
            "format": "png"
        }

    async def _generate_interpretation(
        self,
        analysis_results: Dict[str, Any],
        experiment: Experiment
    ) -> str:
        """Generate interpretation using LLM.

        Args:
            analysis_results: Statistical analysis results
            experiment: Experiment model

        Returns:
            Interpretation text
        """
        prompt = f"""Interpret the following experimental results:

Experiment: {experiment.title}
Protocol: {experiment.protocol[:500]}...

Statistical Results:
{json.dumps(analysis_results, indent=2)}

Provide a comprehensive interpretation that addresses:
1. Summary of findings
2. Statistical significance and practical importance
3. Effect sizes and their meaning
4. Confidence in conclusions
5. Limitations and caveats
6. Implications for the hypothesis

Be rigorous, balanced, and scientifically accurate.
"""

        interpretation = await self.llm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1500
        )

        return interpretation

    def _generate_recommendations(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis.

        Args:
            analysis_results: Analysis results

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for significance
        if "inferential" in analysis_results:
            significant_tests = [
                t for t in analysis_results["inferential"]
                if t.get("significant", False)
            ]

            if len(significant_tests) > 0:
                recommendations.append(
                    "Significant differences were found. Consider replication studies."
                )
            else:
                recommendations.append(
                    "No significant differences detected. Consider increasing sample size or "
                    "refining methodology."
                )

        # Check effect sizes
        if "effect_size" in analysis_results:
            large_effects = [
                k for k, v in analysis_results["effect_size"].items()
                if v.get("magnitude") == "large"
            ]

            if large_effects:
                recommendations.append(
                    f"Large effect sizes found for: {', '.join(large_effects)}. "
                    "These findings warrant further investigation."
                )

        return recommendations
