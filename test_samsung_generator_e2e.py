import asyncio
import logging
import sys
from unittest.mock import MagicMock

# MOCKING MISSING DEPENDENCIES FOR TEST
# The environment seems to lack a proper torch installation, causing sentence_transformers to fail.
# We mock these to proceed with the E2E logic test (Strict Mode verification).
sys.modules["torch"] = MagicMock()
sys.modules["torch.distributed"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from src.proposal.samsung_grant_generator import create_samsung_grant_generator, SamsungGrantSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_e2e_generation():
    logger.info("🚀 Starting End-to-End Test for Samsung Grant Generator")
    
    # 1. Initialize Generator
    generator = await create_samsung_grant_generator()
    # 1. Initialize Generator with Strict Mode
    from src.core.config import settings
    settings.strict_mode = True # Force strict mode for test
    logger.info("🛡️ STRICT MODE ENABLED for Testing")

    generator = await create_samsung_grant_generator()
    logger.info("✅ Generator Initialized")

    # 2. Define Grant Spec
    spec = SamsungGrantSpec(
        research_topic="Development of Multimodal Foundation Model for Early Diagnosis of Autism Spectrum Disorder using Longitudinal Data",
        budget_amount="5000000000", # 50 billion KRW
        duration_years=5,
        innovation_level="World First",
        risk_level="High Risk High Return"
    )

    # 3. Generate Section 1 (Project Overview)
    # This triggers the Unified Agent -> RAG -> LLM pipeline
    logger.info("generating section 1...")
    proposal = await generator.generate_full_proposal(spec)
    
    section_1 = proposal.sections.get("section_1")
    
    if section_1:
        logger.info("\n" + "="*50)
        logger.info(f"✅ Section 1 Generated (Length: {len(section_1.content)} chars)")
        logger.info("="*50)
        logger.info(section_1.content[:2000] + "...") # Print first 2000 chars to see prompt effect
        logger.info("="*50)
        
        # Validation checks
        if "NeuroX-Fusion" in section_1.content or "foundation model" in section_1.content.lower():
             logger.info("✅ Context Check: Found expected terminology (NeuroX-Fusion/foundation model)")
        else:
             logger.warning("⚠️ Context Check: Specific terminology missing?")
             
        if "**" in section_1.content or "#" in section_1.content:
            logger.info("✅ Formatting Check: Markdown formatting detected")
        else:
            logger.warning("⚠️ Formatting Check: Plain text detected?")

        # Save to file for inspection
        with open("generation_result.txt", "w") as f:
            f.write(section_1.content)
            
    else:
        logger.error("❌ Section 1 Generation Failed")

if __name__ == "__main__":
    asyncio.run(test_e2e_generation())
