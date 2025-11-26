import asyncio
import json
import os

# Set mock env vars BEFORE importing src modules
os.environ["SECRET_KEY"] = "test_secret_key_at_least_32_chars_long_value"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost/db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["RABBITMQ_URL"] = "amqp://guest:guest@localhost:5672/"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
os.environ["GEMINI_API_KEY"] = "test-gemini"
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek"

from src.services.paper.style_extractor import StyleMetrics
from src.services.paper.style_transfer import StyleTransferService
from src.services.paper.generator import SectionGenerator, GenerationRequest
from src.services.review.loop import AdversarialReviewLoop
from src.services.review.agents import ReviewerAgent, DefenseAgent
from src.services.paper.formatter import PaperFormatter

# Mock LLM
class MockLLM:
    async def generate(self, prompt, **kwargs):
        if "Reviewer" in prompt:
            return json.dumps({"score": 8.5, "comments": "Good.", "passed": True}), "mock"
        if "Defense" in prompt:
            return json.dumps({"revised_text": "Revised...", "explanation": "Fixed."}), "mock"
        return "Generated content for section...", "mock"

async def main():
    print("Starting Phase 4 End-to-End Verification...")
    
    # 1. Style Setup
    print("\n1. Style Transfer Setup...")
    style_service = StyleTransferService()
    style_guide = style_service.construct_style_guide()
    print("✓ Style guide constructed")

    # 2. Drafting
    print("\n2. Automated Drafting...")
    llm = MockLLM()
    generator = SectionGenerator(llm, rag_client=None)
    
    sections = {}
    for section_type in ["introduction", "methods", "results", "discussion"]:
        req = GenerationRequest(
            section_type=section_type,
            topic="Neural Networks",
            key_points=["Point 1", "Point 2"],
            style_guide=style_guide
        )
        content = await generator.generate_section(req)
        sections[section_type] = content
        print(f"✓ Generated {section_type}")

    # 3. Review Loop (on Introduction only for demo)
    print("\n3. Adversarial Review Loop (Introduction)...")
    reviewer = ReviewerAgent(llm)
    defense = DefenseAgent(llm)
    loop = AdversarialReviewLoop(reviewer, defense)
    
    final_intro, history = await loop.run_loop(sections["introduction"])
    sections["introduction"] = final_intro
    print(f"✓ Review loop completed ({len(history)} iterations)")

    # 4. Formatting
    print("\n4. Final Formatting...")
    formatter = PaperFormatter()
    paper_data = {
        "title": "A Novel Approach to Neural Networks",
        "authors": "AI-CoScientist",
        **sections
    }
    
    md_output = formatter.to_markdown(paper_data)
    latex_output = formatter.to_latex(paper_data)
    
    print("✓ Markdown generated")
    print("✓ LaTeX generated")
    
    # Save outputs
    with open("output/final_paper.md", "w") as f:
        f.write(md_output)
    
    print("\n✅ Phase 4 Verification Successful!")
    print("Output saved to output/final_paper.md")

if __name__ == "__main__":
    asyncio.run(main())
