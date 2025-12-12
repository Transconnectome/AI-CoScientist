import sys
import os
import asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def verify():
    print("Verifying Nature-Level Upgrade...")
    
    # 1. Verify NatureMetrics
    try:
        from src.services.paper.metrics import NatureMetrics
        print("✅ NatureMetrics imported successfully")
        assert hasattr(NatureMetrics, 'score_conceptual_novelty')
        assert hasattr(NatureMetrics, 'score_cross_disciplinary_impact')
        assert hasattr(NatureMetrics, 'score_methodological_rigor')
        print("✅ NatureMetrics methods verified")
    except ImportError as e:
        print(f"❌ Failed to import NatureMetrics: {e}")
    except AssertionError as e:
        print(f"❌ NatureMetrics missing methods: {e}")

    # 2. Verify Agents
    try:
        from src.services.review.editor_agent import EditorAgent
        print("✅ EditorAgent imported successfully")
        from src.services.review.domain_expert_agent import DomainExpertAgent
        print("✅ DomainExpertAgent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Agents: {e}")

    # 3. Verify ImprovementService Loop
    try:
        from src.services.paper.improvement_service import ImprovementService
        print("✅ ImprovementService imported successfully")
        assert hasattr(ImprovementService, 'run_publication_committee_loop')
        print("✅ run_publication_committee_loop method verified")
    except ImportError as e:
        print(f"❌ Failed to import ImprovementService: {e}")
    except AssertionError as e:
        print(f"❌ ImprovementService missing loop method: {e}")

    print("Verification Complete.")

if __name__ == "__main__":
    asyncio.run(verify())
