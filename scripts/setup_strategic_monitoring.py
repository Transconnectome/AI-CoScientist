"""Strategic Literature Monitoring Setup for AI-CoScientist.

This script configures the monitoring system to capture boundary-crossing research
at the intersection of:
- Brain imaging
- Data science
- Psychology (biological psychology)
- Machine learning
- Foundation models
- AI for science

Target conferences: NeurIPS, ICLR, ICML, MICCAI

Strategy:
- 8 targeted sources (4 ArXiv + 4 PubMed)
- 4 precision alerts for filtering
- Expected: 400-600 high-quality papers per month
"""

import asyncio
import httpx
from datetime import datetime
from typing import Dict, List


BASE_URL = "http://localhost:8000/api/v1"


class MonitoringSetup:
    """Setup strategic monitoring sources and alerts for AI-CoScientist."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = None
        self.created_sources = []
        self.created_alerts = []

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    def get_arxiv_sources(self) -> List[Dict]:
        """Define strategic ArXiv sources.

        Each source targets specific conference types and research areas.
        """
        return [
            {
                "name": "Core ML (NeurIPS/ICLR/ICML preprints)",
                "source_type": "arxiv",
                "category": "cs.LG,cs.AI,stat.ML",
                "sync_frequency": "daily",
                "status": "active",
                "rationale": "Captures majority of NeurIPS/ICLR/ICML submissions as preprints. "
                            "90%+ of accepted papers appear on ArXiv first. "
                            "Will filter with alerts for neuroscience/brain imaging keywords."
            },
            {
                "name": "Computational Neuroscience",
                "source_type": "arxiv",
                "category": "q-bio.NC,cs.NE,q-bio.QM",
                "sync_frequency": "daily",
                "status": "active",
                "rationale": "Direct intersection of neuroscience + computational methods. "
                            "Includes NeurIPS Computational Neuroscience track. "
                            "Small volume (~10-20 papers/day), high relevance."
            },
            {
                "name": "Medical Imaging + AI (MICCAI style)",
                "source_type": "arxiv",
                "category": "cs.CV,eess.IV,physics.med-ph",
                "sync_frequency": "weekly",
                "status": "active",
                "rationale": "Targets MICCAI-style research (brain imaging + deep learning). "
                            "Computer Vision + Medical Physics intersection. "
                            "Will filter for 'brain', 'neuroimaging', 'fMRI' keywords."
            },
            {
                "name": "AI for Science (Meta-research)",
                "source_type": "arxiv",
                "category": "cs.AI,cs.CL,cs.HC",
                "sync_frequency": "weekly",
                "status": "active",
                "rationale": "Captures AI for scientific discovery papers. "
                            "Includes LLMs for research, automated experiments, hypothesis generation. "
                            "Direct competitors and inspiration for AI-CoScientist."
            }
        ]

    def get_pubmed_queries(self) -> List[Dict]:
        """Define strategic PubMed queries.

        Uses MeSH terms for precision, targets published research in
        top journals (Nature Neuroscience, PNAS, NeuroImage, etc.)
        """
        return [
            {
                "name": "Neuroimaging + ML (Core)",
                "source_type": "pubmed",
                "query": "(Brain Mapping[MeSH] OR Neuroimaging[MeSH] OR Magnetic Resonance Imaging[MeSH]) "
                        "AND (Machine Learning[MeSH] OR Deep Learning[Title/Abstract] OR Neural Networks, Computer[MeSH])",
                "sync_frequency": "weekly",
                "status": "active",
                "rationale": "Core intersection: brain imaging + modern ML. "
                            "Captures MICCAI, NeuroImage, Nature Neuroscience papers. "
                            "Clinical + methodological innovations."
            },
            {
                "name": "Computational Psychiatry",
                "source_type": "pubmed",
                "query": "(Mental Disorders[MeSH] OR Psychiatry[MeSH] OR Psychology[MeSH]) "
                        "AND (Machine Learning[MeSH] OR Computational Biology[MeSH] OR Data Science[Title/Abstract]) "
                        "AND (Neuroimaging[MeSH] OR Brain[MeSH])",
                "sync_frequency": "weekly",
                "status": "active",
                "rationale": "Psychology + biological psychology + data science intersection. "
                            "Mental disorder prediction, behavioral modeling, digital phenotyping. "
                            "Biological Psychiatry, JAMA Psychiatry papers."
            },
            {
                "name": "AI for Biomedical Research",
                "source_type": "pubmed",
                "query": "(Artificial Intelligence[MeSH] OR Deep Learning[Title/Abstract]) "
                        "AND (Biomedical Research[MeSH] OR Drug Discovery[MeSH] OR Precision Medicine[MeSH])",
                "sync_frequency": "weekly",
                "status": "active",
                "rationale": "AI applications in real scientific discovery. "
                            "Drug discovery, precision medicine, experimental design. "
                            "Cell, Nature Medicine, Science papers."
            },
            {
                "name": "Cognitive Neuroscience + Foundation Models",
                "source_type": "pubmed",
                "query": "(Cognition[MeSH] OR Cognitive Science[Title/Abstract]) "
                        "AND (Neural Networks, Computer[MeSH] OR large language model[Title/Abstract] "
                        "OR foundation model[Title/Abstract] OR transformer[Title/Abstract])",
                "sync_frequency": "weekly",
                "status": "active",
                "rationale": "Cutting-edge: cognitive science + LLMs/transformers. "
                            "Brain-inspired AI, cognitive modeling with foundation models. "
                            "Nature, Science, PNAS papers."
            }
        ]

    def get_alerts(self) -> List[Dict]:
        """Define precision alerts for filtering.

        Each alert captures a specific boundary-crossing research question.
        Acts as second-stage filter on collected papers.
        """
        return [
            {
                "topic": "Brain Decoding + Foundation Models",
                "keywords": [
                    "brain decoding", "neural decoding", "fMRI",
                    "transformer", "foundation model", "large language model", "CLIP",
                    "visual cortex", "neural representation", "encoding model",
                    "shared embedding", "cross-modal"
                ],
                "frequency": "daily",
                "rationale": "Hot topic at NeurIPS/ICLR. "
                            "Intersection: neuroscience + foundation models. "
                            "Examples: using CLIP for brain decoding, transformers for fMRI analysis."
            },
            {
                "topic": "AI for Scientific Discovery",
                "keywords": [
                    "automated experiment", "hypothesis generation", "scientific discovery",
                    "research automation", "experimental design", "active learning",
                    "bayesian optimization", "self-driving lab", "robot scientist",
                    "literature mining", "knowledge graph"
                ],
                "frequency": "daily",
                "rationale": "Direct competitors/inspiration for AI-CoScientist. "
                            "ICML/NeurIPS 'AI for Science' track. "
                            "Automated hypothesis, experiment design, research workflows."
            },
            {
                "topic": "Computational Psychiatry",
                "keywords": [
                    "computational psychiatry", "mental disorder prediction",
                    "depression", "anxiety", "schizophrenia", "ADHD",
                    "resting-state fMRI", "functional connectivity",
                    "predictive model", "biomarker", "precision psychiatry"
                ],
                "frequency": "weekly",
                "rationale": "Clinical impact. "
                            "Psychology + biological psychology + ML + brain imaging. "
                            "Biological Psychiatry, high citation potential."
            },
            {
                "topic": "Multimodal Neuroimaging + AI",
                "keywords": [
                    "multimodal", "cross-modal", "fusion", "integration",
                    "fMRI", "EEG", "MEG", "PET", "DTI",
                    "vision-language", "contrastive learning", "self-supervised",
                    "MICCAI", "medical image analysis"
                ],
                "frequency": "daily",
                "rationale": "MICCAI + NeurIPS intersection. "
                            "Modern ML techniques (contrastive learning, SSL) applied to multimodal brain data. "
                            "Growing field with foundation model influence."
            }
        ]

    async def create_source(self, source_config: Dict) -> Dict:
        """Create a monitoring source via API."""
        name = source_config.pop("name")
        rationale = source_config.pop("rationale")

        response = await self.client.post(
            f"{self.base_url}/monitoring/sources",
            json=source_config
        )
        response.raise_for_status()

        result = response.json()
        result["_name"] = name
        result["_rationale"] = rationale
        return result

    async def create_alert(self, alert_config: Dict) -> Dict:
        """Create a monitoring alert via API."""
        rationale = alert_config.pop("rationale")

        response = await self.client.post(
            f"{self.base_url}/monitoring/alerts",
            json=alert_config
        )
        response.raise_for_status()

        result = response.json()
        result["_rationale"] = rationale
        return result

    async def setup_all(self) -> Dict:
        """Execute full setup: create all sources and alerts."""
        print("=" * 80)
        print("🚀 AI-CoScientist Strategic Monitoring Setup")
        print("=" * 80)
        print()
        print("📊 Strategy:")
        print("  • 8 targeted sources (4 ArXiv + 4 PubMed)")
        print("  • 4 precision alerts for filtering")
        print("  • Target: NeurIPS, ICLR, ICML, MICCAI papers")
        print("  • Expected: 400-600 papers/month")
        print()

        # Create ArXiv sources
        print("📚 Creating ArXiv Sources...")
        print("-" * 80)
        arxiv_sources = self.get_arxiv_sources()
        for i, source_config in enumerate(arxiv_sources, 1):
            try:
                result = await self.create_source(source_config.copy())
                self.created_sources.append(result)
                print(f"✅ [{i}/4] {result['_name']}")
                print(f"    ID: {result['id']}")
                print(f"    Categories: {result['category']}")
                print(f"    Frequency: {result['sync_frequency']}")
                print(f"    💡 {result['_rationale']}")
                print()
            except Exception as e:
                print(f"❌ [{i}/4] Failed: {e}")
                print()

        # Create PubMed sources
        print("🔬 Creating PubMed Sources...")
        print("-" * 80)
        pubmed_queries = self.get_pubmed_queries()
        for i, source_config in enumerate(pubmed_queries, 1):
            try:
                result = await self.create_source(source_config.copy())
                self.created_sources.append(result)
                print(f"✅ [{i}/4] {result['_name']}")
                print(f"    ID: {result['id']}")
                print(f"    Query: {result['query'][:80]}...")
                print(f"    Frequency: {result['sync_frequency']}")
                print(f"    💡 {result['_rationale']}")
                print()
            except Exception as e:
                print(f"❌ [{i}/4] Failed: {e}")
                print()

        # Create alerts
        print("🔔 Creating Precision Alerts...")
        print("-" * 80)
        alerts = self.get_alerts()
        for i, alert_config in enumerate(alerts, 1):
            try:
                result = await self.create_alert(alert_config.copy())
                self.created_alerts.append(result)
                print(f"✅ [{i}/4] {result['topic']}")
                print(f"    ID: {result['id']}")
                print(f"    Keywords: {len(result['keywords'])} terms")
                print(f"    Frequency: {result['frequency']}")
                print(f"    💡 {result['_rationale']}")
                print()
            except Exception as e:
                print(f"❌ [{i}/4] Failed: {e}")
                print()

        # Summary
        print("=" * 80)
        print("📊 Setup Summary")
        print("=" * 80)
        print(f"✅ Sources created: {len(self.created_sources)}/8")
        print(f"✅ Alerts created: {len(self.created_alerts)}/4")
        print()

        if len(self.created_sources) == 8 and len(self.created_alerts) == 4:
            print("🎉 Setup completed successfully!")
            print()
            print("📋 Next Steps:")
            print("  1. Verify sources: curl http://localhost:8000/api/v1/monitoring/sources")
            print("  2. Trigger first sync: curl -X POST http://localhost:8000/api/v1/monitoring/sync/all")
            print("  3. Monitor Celery logs for sync progress")
            print("  4. Check collected papers in database")
            print()
            print("⏰ Automatic Schedule:")
            print("  • Daily syncs: Core ML, Computational Neuroscience")
            print("  • Weekly syncs: Medical Imaging, AI for Science, All PubMed")
            print("  • Alerts: Check hourly for keyword matches")
        else:
            print("⚠️  Some sources/alerts failed to create. Check errors above.")

        print()
        print("=" * 80)

        return {
            "sources_created": len(self.created_sources),
            "alerts_created": len(self.created_alerts),
            "sources": self.created_sources,
            "alerts": self.created_alerts,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def verify_setup(self) -> Dict:
        """Verify all sources and alerts are active."""
        print()
        print("🔍 Verifying Setup...")
        print("-" * 80)

        # Get all sources
        response = await self.client.get(f"{self.base_url}/monitoring/sources")
        sources = response.json()

        # Get all alerts
        response = await self.client.get(f"{self.base_url}/monitoring/alerts")
        alerts = response.json()

        active_sources = [s for s in sources if s['status'] == 'active']
        active_alerts = [a for a in alerts if a['active']]

        print(f"✅ Active sources: {len(active_sources)}")
        print(f"✅ Active alerts: {len(active_alerts)}")

        # Show ArXiv vs PubMed breakdown
        arxiv_count = len([s for s in active_sources if s['source_type'] == 'arxiv'])
        pubmed_count = len([s for s in active_sources if s['source_type'] == 'pubmed'])

        print(f"   • ArXiv sources: {arxiv_count}")
        print(f"   • PubMed sources: {pubmed_count}")
        print()

        return {
            "active_sources": len(active_sources),
            "active_alerts": len(active_alerts),
            "arxiv_sources": arxiv_count,
            "pubmed_sources": pubmed_count
        }

    async def trigger_first_sync(self) -> Dict:
        """Trigger initial sync for all sources."""
        print()
        print("🔄 Triggering Initial Sync...")
        print("-" * 80)

        response = await self.client.post(f"{self.base_url}/monitoring/sync/all")
        result = response.json()

        print(f"✅ Sync task dispatched: {result['task_id']}")
        print(f"   Status: {result['status']}")
        print()
        print("📊 Monitor progress:")
        print("  • Watch Celery worker logs for real-time progress")
        print("  • Each source will sync independently")
        print("  • Estimated time: 5-10 minutes for first sync")
        print()

        return result


async def main():
    """Main execution function."""
    async with MonitoringSetup() as setup:
        # Execute setup
        result = await setup.setup_all()

        # Verify
        verification = await setup.verify_setup()

        # Optionally trigger first sync
        print("🤔 Trigger first sync now? (y/n): ", end="")
        try:
            import sys
            if sys.stdin.isatty():
                choice = input().strip().lower()
                if choice == 'y':
                    await setup.trigger_first_sync()
                else:
                    print()
                    print("⏭️  Skipping first sync. You can trigger it later with:")
                    print("   curl -X POST http://localhost:8000/api/v1/monitoring/sync/all")
            else:
                print("(non-interactive mode, skipping)")
        except:
            print("(skipping)")

        return result


if __name__ == "__main__":
    print()
    print("⚠️  Prerequisites:")
    print("  1. API server running: uvicorn src.main:app --reload")
    print("  2. Database migrated: alembic upgrade head")
    print("  3. Celery worker: celery -A src.core.celery_app worker")
    print("  4. Celery beat: celery -A src.core.celery_app beat")
    print()
    print("Press Enter to continue...")
    try:
        input()
    except:
        pass

    result = asyncio.run(main())
