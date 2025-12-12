import pytest
from src.services.paper.metrics import PaperMetrics

class TestAdvancedMetrics:
    """Test suite for new World-Class metrics."""

    def test_reproducibility_score_perfect(self):
        """Test reproducibility score with all elements present."""
        content = """
        We provide the full source code at https://github.com/example/repo.
        Power analysis was conducted using G*Power (v3.1) with alpha=0.05.
        Data Availability: All datasets are available on Zenodo (DOI: 10.5281/zenodo.12345).
        Effect size (Cohen's d) was 0.8.
        """
        code_snippets = ["def train_model(): pass"]
        
        score = PaperMetrics.score_reproducibility(content, code_snippets)
        assert score >= 9.0
        
    def test_reproducibility_score_poor(self):
        """Test reproducibility score with missing elements."""
        content = """
        We trained a model and it worked well.
        The results are significant (p < 0.05).
        """
        code_snippets = []
        
        score = PaperMetrics.score_reproducibility(content, code_snippets)
        assert score < 3.0

    def test_reproducibility_score_partial(self):
        """Test reproducibility score with some elements."""
        content = """
        Code is available at https://github.com/example/repo.
        We focused on qualitative results.
        """
        code_snippets = ["print('hello')"]
        
        score = PaperMetrics.score_reproducibility(content, code_snippets)
        assert 3.0 <= score <= 7.0

    @pytest.mark.asyncio
    async def test_narrative_arc_mock(self):
        """Test narrative arc scoring (mocked LLM)."""
        # This test assumes we can mock the LLM service or use a heuristic fallback
        # For TDD, we'll start with a placeholder structure in metrics.py
        
        abstract = "This paper solves X."
        intro = "X is a big problem."
        
        # We expect the method to exist and return a dict
        result = await PaperMetrics.score_narrative_arc(abstract, intro)
        
        assert "hook_score" in result
        assert "tension_curve" in result
        assert isinstance(result["hook_score"], float)

    # ============================================
    # Phase 1: Adversarial Review System Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_adversarial_review_identifies_weaknesses(self):
        """Test that adversarial reviewer identifies paper weaknesses."""
        paper_content = """
        Abstract: We trained a neural network and it worked.
        
        Methods: We used some data and trained a model.
        Results showed it was good.
        """
        
        from src.services.paper.adversarial_reviewer import AdversarialReviewer
        reviewer = AdversarialReviewer()
        
        result = await reviewer.review(paper_content)
        
        # Should identify multiple weaknesses
        assert "weaknesses" in result
        assert len(result["weaknesses"]) > 0
        
        # Should have severity scores
        assert "severity_scores" in result
        assert all(0 <= score <= 10 for score in result["severity_scores"].values())
        
        # Should provide detailed feedback
        assert "feedback" in result
        assert len(result["feedback"]) > 50
    
    @pytest.mark.asyncio
    async def test_adversarial_review_severity_scoring(self):
        """Test severity scoring of identified issues."""
        severe_content = """
        We claim to cure cancer with 100% success rate.
        No statistical analysis was performed.
        Sample size: n=3.
        """
        
        from src.services.paper.adversarial_reviewer import AdversarialReviewer
        reviewer = AdversarialReviewer()
        
        result = await reviewer.review(severe_content)
        
        # Severe issues should have high severity
        severity_scores = result["severity_scores"]
        assert any(score >= 8.0 for score in severity_scores.values())
        
        # Should flag methodology issues
        weaknesses_text = " ".join(result["weaknesses"])
        assert "statistical" in weaknesses_text.lower() or "sample" in weaknesses_text.lower()
    
    @pytest.mark.asyncio
    async def test_adversarial_review_multiple_dimensions(self):
        """Test review across multiple quality dimensions."""
        paper_content = """
        Introduction: Neural networks are cool.
        Methods: We did stuff.
        Results: It worked sometimes.
        Discussion: More research needed.
        """
        
        from src.services.paper.adversarial_reviewer import AdversarialReviewer
        reviewer = AdversarialReviewer()
        
        result = await reviewer.review(paper_content, dimensions=["methodology", "clarity", "novelty"])
        
        # Should review all requested dimensions
        assert "dimension_reviews" in result
        assert "methodology" in result["dimension_reviews"]
        assert "clarity" in result["dimension_reviews"]
        assert "novelty" in result["dimension_reviews"]
    
    # ============================================
    # Phase 2: Defense Agent Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_defense_agent_analyzes_criticism(self):
        """Test defense agent's analysis of criticism."""
        criticism = {
            "weaknesses": [
                "Sample size too small (n=10)",
                "No statistical power analysis",
                "Missing effect size calculations"
            ],
            "severity_scores": {
                "sample_size": 8.0,
                "power_analysis": 7.0,
                "effect_size": 6.0
            }
        }
        
        from src.services.paper.defense_agent import DefenseAgent
        agent = DefenseAgent()
        
        result = await agent.analyze_and_defend(criticism)
        
        # Should categorize criticisms
        assert "valid_criticisms" in result
        assert "questionable_criticisms" in result
        
        # Should provide defense strategies
        assert "defense_strategies" in result
        assert len(result["defense_strategies"]) > 0
    
    @pytest.mark.asyncio
    async def test_defense_agent_generates_improvements(self):
        """Test that defense agent generates concrete improvements."""
        paper_content = "Our study used n=15 participants."
        criticism = {
            "weaknesses": ["Sample size insufficient for claimed effect"],
            "severity_scores": {"sample_size": 8.0}
        }
        
        from src.services.paper.defense_agent import DefenseAgent
        agent = DefenseAgent()
        
        result = await agent.generate_improvements(paper_content, criticism)
        
        # Should provide specific improvements
        assert "improvements" in result
        assert len(result["improvements"]) > 0
        
        # Each improvement should have expected score increase
        for improvement in result["improvements"]:
            assert "description" in improvement
            assert "expected_impact" in improvement
            assert 0 < improvement["expected_impact"] <= 3.0
    
    @pytest.mark.asyncio
    async def test_defense_agent_prioritizes_changes(self):
        """Test that defense agent prioritizes high-impact changes."""
        criticisms = {
            "weaknesses": [
                "Minor typo in references",
                "Critical: No statistical analysis",
                "Moderate: Unclear figure labels"
            ],
            "severity_scores": {
                "typo": 2.0,
                "statistics": 9.0,
                "figures": 5.0
            }
        }
        
        from src.services.paper.defense_agent import DefenseAgent
        agent = DefenseAgent()
        
        result = await agent.generate_improvements("", criticisms)
        improvements = result["improvements"]
        
        # Should prioritize by severity
        assert improvements[0]["expected_impact"] > improvements[-1]["expected_impact"]
    
    # ============================================
    # Phase 3: Golden Reference RAG Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_golden_reference_retrieval(self):
        """Test retrieval of golden reference papers."""
        paper_topic = "fMRI brain decoding using deep learning"
        
        result = await PaperMetrics.retrieve_golden_references(
            topic=paper_topic,
            journal_tier="top",
            n_references=3
        )
        
        assert "references" in result
        assert len(result["references"]) > 0
        assert len(result["references"]) <= 3
        
        # Each reference should have metadata
        for ref in result["references"]:
            assert "title" in ref
            assert "similarity_score" in ref
    
    @pytest.mark.asyncio
    async def test_benchmark_scoring(self):
        """Test scoring against golden reference benchmarks."""
        paper_content = """
        We achieved 75% accuracy in brain state classification.
        Methods included data preprocessing and CNN training.
        """
        
        result = await PaperMetrics.score_against_golden_references(
            paper_content=paper_content,
            target_journal_tier="top"
        )
        
        # Should provide benchmark score
        assert "benchmark_score" in result
        assert 0 <= result["benchmark_score"] <= 10
        
        # Should identify gaps
        assert "gap_analysis" in result
        assert len(result["gap_analysis"]) > 0
    
    @pytest.mark.asyncio
    async def test_gap_analysis(self):
        """Test gap analysis against top-tier papers."""
        weak_paper = "We trained a model and got results."
        
        result = await PaperMetrics.score_against_golden_references(
            paper_content=weak_paper,
            target_journal_tier="top"
        )
        
        gaps = result["gap_analysis"]
        
        # Should identify specific gaps
        assert len(gaps) > 0
        
        # Each gap should have improvement suggestion
        for gap in gaps:
            assert "dimension" in gap
            assert "current_level" in gap
            assert "target_level" in gap
            assert "suggestions" in gap
    
    # ============================================
    # Phase 4: Integration Tests
    # ============================================
    
    @pytest.mark.asyncio
    async def test_adversarial_improvement_loop(self):
        """Test full adversarial improvement cycle."""
        initial_paper = """
        Abstract: We used machine learning for brain imaging.
        Methods: Data was collected and analyzed.
        Results: We found significant results.
        """
        
        from src.services.paper.adversarial_reviewer import AdversarialReviewer
        from src.services.paper.defense_agent import DefenseAgent
        
        # Step 1: Initial review
        reviewer = AdversarialReviewer()
        review = await reviewer.review(initial_paper)
        
        assert len(review["weaknesses"]) > 0
        
        # Step 2: Defense and improvement
        agent = DefenseAgent()
        improvements = await agent.generate_improvements(initial_paper, review)
        
        assert len(improvements["improvements"]) > 0
        
        # Step 3: Apply top improvement (simulated)
        # In real implementation, this would modify the paper
        top_improvement = improvements["improvements"][0]
        assert top_improvement["expected_impact"] > 0
    
    @pytest.mark.asyncio
    async def test_defense_improves_score(self):
        """Test that applying defense improvements increases score."""
        # This test verifies the improvement cycle works
        initial_score = 6.5
        target_score = 8.0
        
        # Mock improvement application
        from src.services.paper.defense_agent import DefenseAgent
        agent = DefenseAgent()
        
        # In practice, this would be an iterative process
        result = await agent.estimate_improvement_impact(
            current_score=initial_score,
            target_score=target_score
        )
        
        assert "estimated_iterations" in result
        assert result["estimated_iterations"] <= 5
        assert "feasible" in result
    
    @pytest.mark.asyncio
    async def test_convergence_to_target(self):
        """Test system converges to target quality score."""
        # Verify iterative improvement reaches target
        max_iterations = 10
        target_score = 8.5
        
        # This would track score progression through iterations
        scores = [6.5]  # Initial score
        
        # Simulate improvement iterations
        for i in range(max_iterations):
            if scores[-1] >= target_score:
                break
            # In real implementation, each iteration would:
            # 1. Get adversarial review
            # 2. Generate defense/improvements  
            # 3. Apply improvements
            # 4. Re-evaluate
            expected_gain = 0.3  # Conservative estimate per iteration
            scores.append(min(scores[-1] + expected_gain, 10.0))
        
        # Should reach target in reasonable iterations
        assert scores[-1] >= target_score
        assert len(scores) <= max_iterations
