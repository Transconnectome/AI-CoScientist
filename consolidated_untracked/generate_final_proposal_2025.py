import asyncio
import sys
import re
from pathlib import Path
from typing import Dict, Any

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proposal.samsung_grant_generator import (
    SamsungGrantGenerator, 
    SamsungGrantSpec, 
    ProposalSection,
    ValidationResult,
    SectionType,
    SectionSpec,
    GeneratedSection
)

GRANT_SOURCE_PATH = Path('data/발달장애/_grant.md')
OUTPUT_PATH = Path('data/발달장애/FINAL_SUBMISSION_PROPOSAL.md')

def extract_preserved_sections(file_path: Path) -> Dict[str, str]:
    """Extracts 'Necessity' and 'Research Contents' from the source markdown with robust multiline handling."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    preserved = {}
    
    # Extract "1. 연구의 필요성" - Capture everything until "## 연구내용"
    # DOTALL is essential here. We also want to be careful about not cutting off at the end.
    match_necessity = re.search(r'## 연구의 필요성\s*\n(.*?)(?=\n## 연구내용)', content, re.DOTALL)
    if match_necessity:
        text = match_necessity.group(1).strip()
        # Remove the instructional text if present
        text = re.sub(r'\[\s*연구의 필요성은.*?삭제\]', '', text, flags=re.DOTALL).strip()
        preserved['necessity'] = text
    else:
        print("WARNING: Could not find '## 연구의 필요성' section.")
        preserved['necessity'] = "[CONTENT PRESERVATION FAILED: Necessity]"

    # Extract "2. 연구내용" - Capture everything until "## 연구인력"
    # Note: original file has "## 연구내용" and next is "## 연구인력" (from file view lines 1-76)
    match_contents = re.search(r'## 연구내용\s*\n(.*?)(?=\n## 연구인력)', content, re.DOTALL)
    if match_contents:
        text = match_contents.group(1).strip()
         # Remove the instructional text if present
        text = re.sub(r'\[\s*연구내용에는.*?삭제\]', '', text, flags=re.DOTALL).strip()
        
        # 1.3 Major Development Content Expansion
        # The user complained 1.3 is not explained. We will expand it here if it's too short.
        # But 'Research Contents' is Section 2 in the source.
        # Section 1.3 is "Business Overview -> Major Contents".
        # Let's save the raw content first.
        preserved['research_contents'] = text
    else:
        print("WARNING: Could not find '## 연구내용' section.")
        preserved['research_contents'] = "[CONTENT PRESERVATION FAILED: Research Contents]"
        
    return preserved

class PreservationSamsungGrantGenerator(SamsungGrantGenerator):
    """
    Subclass that overrides specific section generation to strictly preserve
    original text for 'Necessity' and 'Goals', while using AI for others.
    """
    
    def __init__(self, preserved_content: Dict[str, str]):
        super().__init__()
        self.preserved_content = preserved_content

    async def _generate_samsung_section(self, section_spec: SectionSpec, grant_spec: SamsungGrantSpec) -> GeneratedSection:
        """Override generation logic to inject preserved content."""
        
        # Section 1: Research Objectives -> Inject "Necessity"
        if section_spec.type == SectionType.RESEARCH_OBJECTIVES:
            print(f"🔒 Preserving Content for {section_spec.type.value}...")
            # We combine the preserved 'necessity' with some wrapper text to fit the format if needed,
            # but the instruction said "preserve textually".
            content = self.preserved_content.get('necessity', "")
            return GeneratedSection(
                type=section_spec.type,
                content=content,
                word_count=len(content.split()),
                citations_count=0,
                confidence=1.0,
                reasoning="Preserved from original grant document",
                quality_metrics={"preservation_score": 1.0},
                persona_used=section_spec.persona,
                generation_time_ms=0
            )

        # Section 2 (Methodology in generator map) -> Inject "Research Contents"
        # Note: The generator maps METHODOLOGY to "section_2_1_2_2"
        elif section_spec.type == SectionType.METHODOLOGY:
            print(f"🔒 Preserving Content for {section_spec.type.value}...")
            content = self.preserved_content.get('research_contents', "")
            return GeneratedSection(
                type=section_spec.type,
                content=content,
                word_count=len(content.split()),
                citations_count=0,
                confidence=1.0,
                reasoning="Preserved from original grant document",
                quality_metrics={"preservation_score": 1.0},
                persona_used=section_spec.persona,
                generation_time_ms=0
            )

        # For other sections (Innovation, Timeline, Budget), use the AI Agent (DD-RAPTOR)
        else:
            print(f"🤖 AI Generating Content for {section_spec.type.value} using DD-RAPTOR...")
            # Modify the DD query to ensure compliance with new rules (NeuroX-Fusion 10B, Zebrafish downplay)
            original_query = self._create_dd_query_for_section(section_spec.type)
            enhanced_query = f"{original_query} NeuroX-Fusion 10B Spatiotemporal Manifold Zebrafish screening"
            
            return await self.proposal_agent.generate_with_dd_knowledge(
                section_type=section_spec.type.value,
                dd_query=enhanced_query
            )

async def generate_final_proposal():
    print('🚀 Starting FINAL Samsung Grant Generation (2.5B Special Edition)...')
    
    # 1. Extract Preserved Content
    print(f'📖 Reading source from {GRANT_SOURCE_PATH}...')
    preserved_content = extract_preserved_sections(GRANT_SOURCE_PATH)
    
    # 2. Define the Strict Spec (2.5B, NeuroX-Fusion 10B)
    spec = SamsungGrantSpec(
        research_topic='NeuroX-Fusion 10B: Hospital-Ready Foundation Model for Developmental Disorders',
        budget_amount='2,500,000,000',  # 25억 Strict
        duration_years=5,
        innovation_level='paradigm_shift_to_efficiency',
        risk_level='high_impact_proven_feasibility',
        target_audience='samsung_future_tech_committee',
        language='korean'
    )
    
    # 3. Initialize Custom Generator
    generator = PreservationSamsungGrantGenerator(preserved_content)
    await generator.initialize()
    
    # 4. Generate
    print(f'📝 Generating Proposal for: {spec.research_topic}')
    proposal = await generator.generate_full_proposal(spec)
    
    # 5. Save Output
    print(f'💾 Saving to {OUTPUT_PATH}')
    # We call the internal save method or manually save. 
    # Since generate_full_proposal saves to a timestamped file in output/, we also want to force save to our target path.
    await generator._save_proposal_as_markdown(proposal, OUTPUT_PATH)
    
    print('✅ Final Proposal Generation Complete!')

if __name__ == '__main__':
    asyncio.run(generate_final_proposal())
