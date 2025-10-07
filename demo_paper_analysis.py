#!/usr/bin/env python3
"""Demo script to analyze paper using AI-CoScientist services."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.models.project import Project, Paper
from src.services.paper import PaperParser, PaperAnalyzer, PaperImprover
from src.services.llm.service import LLMService
from src.core.config import settings


async def main():
    """Run paper analysis workflow."""

    # Read extracted paper text
    print("📄 Reading extracted paper...")
    with open("paper_extracted.txt", "r", encoding="utf-8") as f:
        paper_text = f.read()

    print(f"✅ Loaded paper ({len(paper_text)} characters)\n")

    # Initialize database connection
    print("🔧 Initializing database connection...")
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Initialize LLM service
        print("🤖 Initializing LLM service...")
        llm_service = LLMService(
            primary_provider=settings.llm_primary_provider,
            fallback_provider=settings.llm_fallback_provider
        )

        # Create services
        parser = PaperParser(llm_service, db)
        analyzer = PaperAnalyzer(llm_service, db)
        improver = PaperImprover(llm_service, db)

        # Create a test project
        print("\n📋 Creating test project...")
        project = Project(
            title="Neuro-Genetic Susceptibility Study",
            research_question="How does neighborhood socioeconomic adversity interact with genetic and neural factors to influence children's psychotic-like experiences?",
            status="active"
        )
        db.add(project)
        await db.commit()
        await db.refresh(project)
        print(f"✅ Created project: {project.title} (ID: {project.id})")

        # Create paper
        print("\n📝 Creating paper from extracted text...")
        paper = Paper(
            project_id=project.id,
            title="Quasi-Experimental Analysis Reveals Neuro-Genetic Susceptibility to Neighborhood Socioeconomic Adversity",
            content=paper_text,
            status="draft",
            version=1
        )
        db.add(paper)
        await db.commit()
        await db.refresh(paper)
        print(f"✅ Created paper (ID: {paper.id})")

        # Step 1: Parse paper into sections
        print("\n" + "="*80)
        print("STEP 1: PARSING PAPER INTO SECTIONS")
        print("="*80)

        try:
            sections = await parser.extract_sections(paper_text)
            print(f"\n✅ Parsed {len(sections)} sections:")
            for section in sections:
                content_preview = section['content'][:100].replace('\n', ' ')
                print(f"  • {section['name']}: {content_preview}... ({len(section['content'])} chars)")

            # Save sections to database
            from src.models.project import PaperSection
            for section in sections:
                paper_section = PaperSection(
                    paper_id=paper.id,
                    name=section['name'],
                    content=section['content'],
                    order=section['order'],
                    version=1
                )
                db.add(paper_section)
            await db.commit()
            print(f"\n✅ Saved {len(sections)} sections to database")

        except Exception as e:
            print(f"❌ Error parsing paper: {e}")
            return

        # Step 2: Analyze paper quality
        print("\n" + "="*80)
        print("STEP 2: ANALYZING PAPER QUALITY")
        print("="*80)

        try:
            analysis = await analyzer.analyze_quality(paper.id)

            print(f"\n📊 Quality Score: {analysis['quality_score']:.1f}/10")
            print(f"💡 Clarity Score: {analysis.get('clarity_score', 'N/A')}")
            print(f"🔗 Coherence Score: {analysis.get('coherence_score', 'N/A')}")

            print("\n💪 Strengths:")
            for strength in analysis.get('strengths', []):
                print(f"  ✓ {strength}")

            print("\n⚠️  Weaknesses:")
            for weakness in analysis.get('weaknesses', []):
                print(f"  ✗ {weakness}")

            print("\n💡 Suggestions:")
            for suggestion in analysis.get('suggestions', [])[:5]:
                print(f"  → {suggestion}")

        except Exception as e:
            print(f"❌ Error analyzing paper: {e}")
            import traceback
            traceback.print_exc()

        # Step 3: Check section coherence
        print("\n" + "="*80)
        print("STEP 3: CHECKING SECTION COHERENCE")
        print("="*80)

        try:
            coherence = await analyzer.check_section_coherence(paper.id)

            print(f"\n🔗 Overall Coherence: {coherence.get('coherence_score', 'N/A')}/10")
            print(f"\n📋 Section Analysis:")
            for section_analysis in coherence.get('section_analyses', [])[:5]:
                print(f"  • {section_analysis}")

        except Exception as e:
            print(f"❌ Error checking coherence: {e}")

        # Step 4: Identify gaps
        print("\n" + "="*80)
        print("STEP 4: IDENTIFYING CONTENT GAPS")
        print("="*80)

        try:
            gaps = await analyzer.identify_gaps(paper.id)

            if gaps:
                print(f"\n🔍 Found {len(gaps)} potential gaps:")
                for gap in gaps[:5]:
                    print(f"\n  📌 {gap.get('area', 'Unknown area')}")
                    print(f"     Severity: {gap.get('severity', 'N/A')}")
                    print(f"     Description: {gap.get('description', 'N/A')}")
            else:
                print("\n✅ No significant gaps identified")

        except Exception as e:
            print(f"❌ Error identifying gaps: {e}")

        # Step 5: Generate improvements for introduction section
        print("\n" + "="*80)
        print("STEP 5: GENERATING IMPROVEMENT FOR INTRODUCTION")
        print("="*80)

        try:
            improvement = await improver.improve_section(
                paper.id,
                section_name="introduction",
                feedback="Make it more concise and emphasize the novel contribution"
            )

            print(f"\n✨ Improvement Score: {improvement.get('improvement_score', 'N/A')}/10")
            print(f"\n📝 Changes Summary:")
            print(f"  {improvement.get('changes_summary', 'N/A')}")
            print(f"\n📄 Improved Content Preview (first 300 chars):")
            improved_content = improvement.get('improved_content', '')
            print(f"  {improved_content[:300]}...")

        except Exception as e:
            print(f"❌ Error generating improvements: {e}")

        # Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"""
✅ Successfully analyzed paper:
   - Project ID: {project.id}
   - Paper ID: {paper.id}
   - Sections parsed: {len(sections)}
   - Quality score: {analysis.get('quality_score', 'N/A')}/10

📊 Next steps:
   1. Review analysis results above
   2. Apply improvements to specific sections
   3. Re-analyze after improvements
   4. Export final version
""")


if __name__ == "__main__":
    asyncio.run(main())
