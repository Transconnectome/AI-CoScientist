#!/usr/bin/env python3
"""Test Word export functionality with actual paper."""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from src.models.project import Project, Paper, PaperSection
from src.services.paper.exporter import PaperExporter
from src.core.config import settings


async def main():
    """Test Word export with sample paper."""

    load_dotenv()

    print("="*80)
    print("WORD EXPORT TEST")
    print("="*80)

    # Read extracted paper
    print("\n📄 Reading extracted paper...")
    with open("paper_extracted.txt", "r", encoding="utf-8") as f:
        paper_text = f.read()

    print(f"✅ Loaded paper ({len(paper_text)} characters)\n")

    # Initialize database
    print("🔧 Initializing database...")
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Create test project
        print("📋 Creating test project...")
        project = Project(
            name="Neuro-Genetic Susceptibility Study",
            description="How does neighborhood socioeconomic adversity interact with genetic and neural factors?",
            domain="neuroscience",
            status="active"
        )
        db.add(project)
        await db.commit()
        await db.refresh(project)
        print(f"✅ Project created: {project.id}")

        # Create paper
        print("\n📝 Creating paper...")
        paper = Paper(
            project_id=project.id,
            title="Quasi-Experimental Analysis Reveals Neuro-Genetic Susceptibility to Neighborhood Socioeconomic Adversity",
            abstract="This study explores the relationship between neighborhood socioeconomic adversity and children's psychotic-like experiences (PLEs), investigating how genetic and neural factors moderate this association.",
            content=paper_text,
            status="draft",
            version=1
        )
        db.add(paper)
        await db.commit()
        await db.refresh(paper)
        print(f"✅ Paper created: {paper.id}")

        # Create sample sections manually
        print("\n📑 Creating paper sections...")
        sections_data = [
            {
                "name": "introduction",
                "content": """Introduction

A child's environment significantly influences their lifelong health, economic, and social outcomes. Adverse conditions, such as poverty, malnutrition, abuse, and unsafe neighborhoods, increase the risk of mental and physical health issues, impair cognitive abilities, and promote risky behaviors.

This study investigates how neighborhood socioeconomic adversity interacts with genetic and neural factors to influence children's psychotic-like experiences (PLEs). We use instrumental variable (IV) forest methodology with data from the ABCD Study to examine these complex relationships.

Our research contributes to understanding the mechanisms through which environmental adversity affects child development and mental health outcomes.""",
                "order": 0
            },
            {
                "name": "methods",
                "content": """Methods

Participants:
The ABCD Study recruited 11,878 participants aged 9-10 years from 21 research sites across the United States. Our final sample included 2,135 participants after applying inclusion criteria.

Measures:
- Neighborhood socioeconomic adversity indices
- Genomic data (polygenic risk scores)
- Structural MRI scans
- Psychotic-like experiences assessment

Statistical Analysis:
We employed instrumental variable (IV) forest methodology to address potential confounding from unobserved factors. This quasi-experimental approach allows causal inference while accounting for complex gene-environment interactions.""",
                "order": 1
            },
            {
                "name": "results",
                "content": """Results

Our analysis revealed significant interactions between neighborhood adversity and both genetic and neural factors in predicting PLEs.

Key Findings:
1. Higher neighborhood adversity associated with increased PLEs (β = 0.34, p < 0.001)
2. Genetic vulnerability moderated this relationship (interaction p = 0.012)
3. Brain structure metrics (particularly in prefrontal regions) showed moderating effects
4. IV forest approach confirmed causal relationships while controlling for confounding

The results demonstrate that children with certain genetic and neural characteristics are more susceptible to the negative effects of neighborhood adversity on mental health outcomes.""",
                "order": 2
            },
            {
                "name": "discussion",
                "content": """Discussion

This study provides evidence for neuro-genetic susceptibility to neighborhood socioeconomic adversity in children's psychotic-like experiences. Our findings have several important implications:

1. Personalized Intervention: Children with higher genetic or neural vulnerability may benefit from targeted interventions when living in adverse neighborhoods.

2. Policy Implications: Understanding these susceptibility factors can inform public health policies aimed at protecting vulnerable children.

3. Mechanistic Insights: The results suggest that gene-environment interactions operate through neural pathways to influence mental health outcomes.

Limitations include the cross-sectional design and potential selection bias. Future research should examine longitudinal trajectories and test intervention strategies.""",
                "order": 3
            }
        ]

        for section_data in sections_data:
            section = PaperSection(
                paper_id=paper.id,
                **section_data,
                version=1
            )
            db.add(section)

        await db.commit()
        print(f"✅ Created {len(sections_data)} sections")

        # Test 1: Export basic Word document
        print("\n" + "="*80)
        print("TEST 1: BASIC WORD EXPORT")
        print("="*80)

        exporter = PaperExporter(db)

        print("\n📄 Exporting to Word format...")
        output_path = await exporter.export_to_word(
            paper.id,
            output_path="paper_original.docx",
            include_metadata=True
        )

        print(f"✅ Word document created: {output_path}")
        print(f"📊 File size: {Path(output_path).stat().st_size:,} bytes")

        # Test 2: Export with improvements
        print("\n" + "="*80)
        print("TEST 2: EXPORT WITH AI IMPROVEMENTS")
        print("="*80)

        # Simulate improved content
        improvements = {
            "introduction": """Introduction (AI-Improved)

Children's environments profoundly shape their lifelong trajectories across health, economic, and social domains. Exposure to adverse conditions—including poverty, malnutrition, abuse, and unsafe neighborhoods—significantly elevates risks for mental and physical health disorders, cognitive impairments, and maladaptive behaviors.

This investigation examines the interplay between neighborhood socioeconomic adversity and genetic/neural factors in shaping children's psychotic-like experiences (PLEs). Leveraging instrumental variable (IV) forest methodology with ABCD Study data, we elucidate these intricate gene-environment-brain interactions.

Our findings advance mechanistic understanding of how environmental adversity influences developmental and mental health outcomes in genetically and neurally susceptible children.""",

            "discussion": """Discussion (AI-Improved)

Our findings provide compelling evidence for neuro-genetic susceptibility moderating the impact of neighborhood socioeconomic adversity on children's psychotic-like experiences. Three key implications emerge:

First, the identification of genetic and neural vulnerability markers enables precision-targeted interventions. Children exhibiting these susceptibility factors may derive maximal benefit from enhanced support when residing in adverse neighborhoods.

Second, these findings inform evidence-based public health policy. Understanding susceptibility heterogeneity allows resource allocation toward protecting the most vulnerable children.

Third, our results illuminate mechanistic pathways: gene-environment interactions manifest through neural architecture to shape mental health trajectories.

Study limitations—notably cross-sectional design and potential selection bias—suggest caution in causal interpretation. Longitudinal research examining developmental trajectories and intervention efficacy represents a critical next step."""
        }

        print("\n📄 Exporting improved version to Word...")
        improved_path = await exporter.export_improved_paper(
            paper.id,
            improvements=improvements,
            output_path="paper_improved.docx"
        )

        print(f"✅ Improved Word document created: {improved_path}")
        print(f"📊 File size: {Path(improved_path).stat().st_size:,} bytes")

        # Summary
        print("\n" + "="*80)
        print("EXPORT TEST COMPLETE")
        print("="*80)

        print(f"""
✅ Successfully created Word documents:

1️⃣  Original Paper:
   📄 File: {output_path}
   📊 Size: {Path(output_path).stat().st_size:,} bytes
   📝 Contents:
      - Title page with metadata
      - Abstract
      - 4 sections (Introduction, Methods, Results, Discussion)
      - Professional formatting (Times New Roman, 12pt, 1.5 spacing)

2️⃣  Improved Paper:
   📄 File: {improved_path}
   📊 Size: {Path(improved_path).stat().st_size:,} bytes
   📝 Contents:
      - All original content
      - AI-improved sections marked with notes
      - 2 sections improved (Introduction, Discussion)
      - Blue note indicating AI improvements

🎯 Word Export Features:
   ✓ Professional academic formatting
   ✓ Automatic page breaks
   ✓ Proper heading hierarchy
   ✓ Metadata (version, status, date)
   ✓ AI improvement annotations
   ✓ Ready for submission or further editing

💡 Next Steps:
   1. Open the .docx files in Microsoft Word
   2. Review the formatting and content
   3. Make any final edits
   4. Use for submission or collaboration
""")

        print("\n📂 Files created in current directory:")
        print(f"   • {output_path}")
        print(f"   • {improved_path}")


if __name__ == "__main__":
    asyncio.run(main())
