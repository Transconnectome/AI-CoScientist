import asyncio
import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proposal.samsung_grant_generator import SamsungGrantGenerator, SamsungGrantSpec, ProposalStatus

async def generate_revised_grant():
    print('🚀 Starting New Strategy Samsung Grant Generation...')
    
    # 1. Define the Revised Strategy Spec
    spec = SamsungGrantSpec(
        research_topic='NeuroX-Fusion 10B: Hospital-Ready Foundation Model for Developmental Disorders',
        budget_amount='5,000,000,000',  # 50억
        duration_years=5,
        innovation_level='paradigm_shift_to_efficiency',
        risk_level='high_impact_proven_feasibility',
        target_audience='samsung_future_tech_committee',
        language='korean'
    )
    
    # 2. Initialize Generator
    generator = SamsungGrantGenerator()
    await generator.initialize()
    
    # 3. Inject the Strategic Narrative (150B Resource + 10B Model)
    # We manually override the prompt context or specific sections if possible, 
    # or relying on the 'research_topic' to drive the agent.
    # For this script, we rely on the agent to pick up the topic nuances.
    
    print(f'📝 Generating Proposal for: {spec.research_topic}')
    proposal = await generator.generate_full_proposal(spec)
    
    # 4. Save Output
    output_path = Path('data/발달장애/Samsung_Future_Tech_Proposal_AI_Generated.md')
    print(f'💾 Saving to {output_path}')
    await generator._save_proposal_as_markdown(proposal, output_path)
    
    print('✅ Generation Complete!')

if __name__ == '__main__':
    asyncio.run(generate_revised_grant())
