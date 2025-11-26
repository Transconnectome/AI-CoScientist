"""
Demo script for the new advanced metrics improvements.

This demonstrates the complete adversarial improvement loop:
1. Adversarial Review (Reviewer #2)
2. Defense Agent response
3. Golden Reference comparison
4. Iterative improvement
"""

import asyncio
from src.services.paper.adversarial_reviewer import AdversarialReviewer
from src.services.paper.defense_agent import DefenseAgent
from src.services.paper.metrics import PaperMetrics


async def demo_adversarial_improvement_loop():
    """Demonstrate the complete adversarial improvement workflow."""
    
    print("=" * 80)
    print("🔬 Advanced Metrics Improvement Demo")
    print("=" * 80)
    print()
    
    # Sample paper with obvious weaknesses
    initial_paper = """
    Abstract: We used machine learning for brain imaging analysis.
    
    Introduction: Neural networks are cool and useful for analyzing fMRI data.
    We think they can help understand the brain.
    
    Methods: We collected data from some participants. We trained a neural 
    network and it worked. The results were good.
    
    Results: The model achieved good accuracy. We found interesting patterns.
    
    Discussion: This is important for neuroscience. More research is needed.
    """
    
    print("📄 Initial Paper (v1.0)")
    print("-" * 80)
    print(initial_paper[:200] + "...")
    print()
    
    # Phase 1: Adversarial Review
    print("\n🔴 Phase 1: Adversarial Review (Reviewer #2)")
    print("-" * 80)
    
    reviewer = AdversarialReviewer()
    review = await reviewer.review(initial_paper)
    
    print(f"✗ Weaknesses identified: {len(review['weaknesses'])}")
    for i, weakness in enumerate(review['weaknesses'][:5], 1):
        severity_key = list(review['severity_scores'].keys())[min(i-1, len(review['severity_scores'])-1)]
        severity = review['severity_scores'].get(severity_key, 5.0)
        print(f"  {i}. [{severity:.1f}/10] {weakness}")
    
    print(f"\n📝 Reviewer Feedback:")
    print(review['feedback'][:300] + "...")
    
    # Phase 2: Defense Agent
    print("\n\n🛡️ Phase 2: Defense Agent Response")
    print("-" * 80)
    
    agent = DefenseAgent()
    defense = await agent.analyze_and_defend(review)
    
    print(f"✓ Valid criticisms: {len(defense['valid_criticisms'])}")
    print(f"? Questionable criticisms: {len(defense['questionable_criticisms'])}")
    print(f"💡 Defense strategies proposed: {len(defense['defense_strategies'])}")
    
    # Generate improvements
    improvements = await agent.generate_improvements(initial_paper, review)
    
    print(f"\n🔧 Top 3 Improvements:")
    for i, improvement in enumerate(improvements['improvements'][:3], 1):
        print(f"  {i}. [{improvement['priority']}] {improvement['description']}")
        print(f"     Expected impact: +{improvement['expected_impact']:.2f} points")
    
    # Phase 3: Golden Reference Comparison
    print("\n\n⭐ Phase 3: Golden Reference Comparison")
    print("-" * 80)
    
    benchmark = await PaperMetrics.score_against_golden_references(
        initial_paper,
        target_journal_tier="top"
    )
    
    print(f"📊 Benchmark Score: {benchmark['benchmark_score']:.2f}/10")
    print(f"🎯 Tier Assessment: {benchmark['tier_assessment']}")
    
    print(f"\n📉 Quality Gaps Identified: {len(benchmark['gap_analysis'])}")
    for gap in benchmark['gap_analysis'][:3]:
        print(f"\n  Dimension: {gap['dimension'].upper()}")
        print(f"  Current: {gap['current_level']:.2f} | Target: {gap['target_level']:.2f} | Gap: {gap['gap']:.2f}")
        print(f"  Suggestions:")
        for suggestion in gap['suggestions'][:2]:
            print(f"    • {suggestion}")
    
    # Phase 4: Estimate Improvement Path
    print("\n\n📈 Phase 4: Improvement Trajectory Estimation")
    print("-" * 80)
    
    current_score = benchmark['benchmark_score']
    target_score = 8.5
    
    estimate = await agent.estimate_improvement_impact(current_score, target_score)
    
    print(f"Current Score: {current_score:.2f}")
    print(f"Target Score: {target_score:.2f}")
    print(f"Score Gap: {estimate['score_gap']:.2f}")
    print(f"Estimated Iterations: {estimate['estimated_iterations']}")
    print(f"Feasible: {'✓ Yes' if estimate['feasible'] else '✗ No'}")
    print(f"Note: {estimate['note']}")
    
    # Simulate improvement iterations
    print("\n\n🔄 Simulated Improvement Iterations")
    print("-" * 80)
    
    scores = [current_score]
    for iteration in range(1, estimate['estimated_iterations'] + 1):
        # Apply top improvement (simulated gain)
        current_score = min(current_score + 0.4, 10.0)
        scores.append(current_score)
        
        print(f"Iteration {iteration}: {scores[-2]:.2f} → {scores[-1]:.2f} (+{scores[-1] - scores[-2]:.2f})")
        
        if current_score >= target_score:
            print(f"\n✅ Target score reached in {iteration} iterations!")
            break
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    print(f"\nInitial Score: {scores[0]:.2f}/10")
    print(f"Final Score: {scores[-1]:.2f}/10")
    print(f"Improvement: +{scores[-1] - scores[0]:.2f} points")
    print(f"Iterations: {len(scores) - 1}")
    print(f"\nJournal Tier Progression:")
    print(f"  {PaperMetrics._assess_tier(scores[0])}")
    print(f"  → {PaperMetrics._assess_tier(scores[-1])}")
    print()


async def demo_golden_references():
    """Demonstrate golden reference retrieval."""
    
    print("\n" + "=" * 80)
    print("📚 Golden Reference Retrieval Demo")
    print("=" * 80)
    print()
    
    topic = "fMRI brain decoding using deep learning"
    print(f"Topic: {topic}")
    print()
    
    result = await PaperMetrics.retrieve_golden_references(
        topic=topic,
        journal_tier="top",
        n_references=3
    )
    
    print(f"Found {len(result['references'])} top-tier references:")
    print()
    
    for i, ref in enumerate(result['references'], 1):
        print(f"{i}. {ref['title']}")
        print(f"   Journal: {ref['journal']} ({ref['year']})")
        print(f"   Impact Factor: {ref['impact_factor']}")
        print(f"   Similarity: {ref['similarity_score']:.2%}")
        print()


if __name__ == "__main__":
    print("\n🚀 Starting Advanced Metrics Improvements Demo\n")
    
    # Run main demo
    asyncio.run(demo_adversarial_improvement_loop())
    
    # Run golden references demo
    asyncio.run(demo_golden_references())
    
    print("\n✨ All demos completed successfully!\n")
