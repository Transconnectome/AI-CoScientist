# 📋 Unified RAG Proposal Optimization - Quick Reference Guide

## 🚀 Next-Generation Proposal Enhancement System

**Major Upgrade (2025)**: Migrated from single DD-RAPTOR strategy to **Unified RAG Orchestrator** with intelligent 6-strategy routing and cross-domain knowledge synthesis.

## 🎯 Quick Commands

### Basic Optimization
```bash
# Complete Unified RAG optimization (95+ score target)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode full --enable-cross-domain

# Quick improvement (85+ score target)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode quick --strategies "HYBRID,GRAPH_RAG"

# Cross-domain synthesis (ESM3 + Neuroscience + Quantum ML)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode cross_domain
```

### Interactive Mode
```bash
# Interactive wizard with 6-strategy selection
poetry run python scripts/proposal_optimizer_unified.py wizard --unified-rag

# Strategy-specific optimization
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --strategies "GRAPH_RAG,MULTIMODAL_RAG" \
    --domains "neuroscience,protein_research"
```

### Quality Assessment
```bash
# Unified multi-strategy quality assessment
poetry run python scripts/map_proposal_to_unified_evidence.py \
    --proposal "proposal.md" --output "assessment.json" \
    --unified-rag --quality-assessment

# Cross-domain evidence validation
poetry run python scripts/validate_claims_unified_rag.py \
    --input "proposal.md" --enable-cross-domain --strategies "ALL"
```

## 🔧 6-Strategy RAG System

### Available Strategies

| Strategy | Best For | Example Use Case |
|----------|----------|------------------|
| **HYBRID** | General optimization, balanced performance | Standard proposal improvement |
| **GRAPH_RAG** | Complex reasoning, relationship analysis | Research methodology sections |
| **ENHANCED_DD_RAPTOR** | Domain-specific research | Neurodevelopmental studies |
| **GOLDEN_REFERENCE** | High-quality validation | Literature review sections |
| **MULTIMODAL_RAG** | Cross-modal analysis | Proposals with figures/tables |
| **PSYCHOLOGY_RAG** | Psychology domain expertise | Behavioral research proposals |

### Strategy Selection Guide

```bash
# For neuroscience research proposals
--strategies "ENHANCED_DD_RAPTOR,GRAPH_RAG,HYBRID"

# For protein research (ESM3) proposals
--strategies "GRAPH_RAG,MULTIMODAL_RAG,GOLDEN_REFERENCE"

# For quantum ML research
--strategies "HYBRID,GRAPH_RAG,GOLDEN_REFERENCE"

# For psychological studies
--strategies "PSYCHOLOGY_RAG,HYBRID,ENHANCED_DD_RAPTOR"

# For interdisciplinary research
--strategies "ALL" --enable-cross-domain
```

## 🌐 Cross-Domain Modes

### Available Domains
- `neuroscience` - Brain research, neural networks, developmental disorders
- `protein_research` - ESM3, protein structure, evolution, drug discovery
- `quantum_ml` - Quantum computing, machine learning optimization
- `general` - General AI/ML research

### Cross-Domain Examples

```bash
# Neuroscience + Protein research synthesis
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "brain_protein_proposal.md" --mode cross_domain \
    --domains "neuroscience,protein_research" \
    --strategies "GRAPH_RAG,MULTIMODAL_RAG"

# ESM3 + Neuroscience + Quantum ML integration
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "ai_breakthrough_proposal.md" --mode full \
    --enable-cross-domain \
    --domains "neuroscience,protein_research,quantum_ml"

# Grant proposal optimization (all domains)
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "samsung_grant.md" --mode full \
    --enable-cross-domain --strategies "ALL"
```

## 📊 Optimization Modes

### Mode Comparison

| Mode | Steps | Time | Strategies | Best For |
|------|-------|------|------------|----------|
| **full** | 1,2,3,4,5 | 30-70min | ALL | Complete optimization |
| **quick** | 1,3,5 | 15-30min | HYBRID, GRAPH_RAG | Fast improvement |
| **research** | 1,2,3,4 | 25-50min | GRAPH_RAG, ENHANCED_DD_RAPTOR | Research-focused |
| **validation** | 1,2,5 | 20-40min | GOLDEN_REFERENCE, HYBRID | Quality assurance |
| **cross_domain** | 1,3,4,5 | 30-60min | GRAPH_RAG, MULTIMODAL_RAG | Multi-domain synthesis |

### Mode Selection Guide

```bash
# For Samsung Future Technology grants
--mode full --enable-cross-domain

# For quick conference paper improvement
--mode quick --strategies "HYBRID,GOLDEN_REFERENCE"

# For comprehensive research proposals
--mode research --domains "neuroscience,protein_research"

# For validation and quality assurance
--mode validation --strategies "GOLDEN_REFERENCE,HYBRID"

# For breakthrough interdisciplinary research
--mode cross_domain --domains "neuroscience,protein_research,quantum_ml"
```

## 🔍 Enhanced 5-Stage Pipeline

### Stage Details

1. **Unified Evidence Mapping**
   - Script: `map_proposal_to_unified_evidence.py`
   - Purpose: Cross-domain scientific claim analysis
   - Strategies: ALL (automatic selection)
   - Output: Multi-strategy evidence scores

2. **Multi-Strategy Validation**
   - Script: `validate_claims_unified_rag.py`
   - Purpose: 6-strategy claim verification
   - Strategies: GOLDEN_REFERENCE, HYBRID, SIMPLE_RAG
   - Output: Validated claims with confidence scores

3. **Advanced RAG Literature Review**
   - Script: `advanced_unified_query.py`
   - Purpose: Multi-modal systematic search
   - Strategies: GRAPH_RAG, MULTIMODAL_RAG, PSYCHOLOGY_RAG
   - Output: Enhanced literature citations

4. **Multi-Agent Unified Enhancement**
   - Script: `multi_agent_unified_pipeline.py`
   - Purpose: 6 AI specialists + RAG integration
   - Strategies: Agent-specific optimization
   - Output: Collaboratively improved content

5. **Intelligent Unified Citation**
   - Script: `unified_citation_generator.py`
   - Purpose: Cross-domain auto-reference generation
   - Strategies: GOLDEN_REFERENCE, GRAPH_RAG
   - Output: Comprehensive bibliography

## 📈 Expected Performance

### Quality Targets

| Proposal Type | Target Score | Key Strategies | Expected Improvement |
|---------------|--------------|----------------|---------------------|
| Samsung Grant | 95+ | ALL | +15-25% |
| Research Paper | 90+ | GRAPH_RAG, HYBRID | +10-20% |
| Conference Abstract | 85+ | QUICK mode | +8-15% |
| Grant Proposal | 92+ | CROSS_DOMAIN | +12-22% |

### Success Metrics

- **Multi-Domain Coverage**: ESM3 protein + Neuroscience + Quantum ML synthesis
- **6-Strategy Validation**: >85% claims supported across multiple strategies
- **Cross-Modal Intelligence**: Text + Image + Table + Citation analysis
- **Response Time**: <2s for 95% of cross-domain queries
- **Strategy Diversity**: Optimal routing across 6 intelligent strategies

## 🚫 Common Issues & Solutions

### Issue: Low Strategy Confidence
```bash
# Problem: Single strategy giving low confidence scores
# Solution: Enable cross-domain synthesis
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --enable-cross-domain --strategies "ALL"
```

### Issue: Missing Domain Coverage
```bash
# Problem: Proposal needs specific domain knowledge
# Solution: Specify target domains explicitly
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --domains "neuroscience,protein_research" \
    --strategies "GRAPH_RAG,MULTIMODAL_RAG"
```

### Issue: Long Processing Time
```bash
# Problem: Full optimization taking too long
# Solution: Use quick mode with key strategies
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode quick \
    --strategies "HYBRID,GOLDEN_REFERENCE"
```

### Issue: Inconsistent Quality
```bash
# Problem: Quality varies across sections
# Solution: Run validation-focused optimization
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" --mode validation \
    --strategies "GOLDEN_REFERENCE,ENHANCED_DD_RAPTOR"
```

## 🎓 Advanced Usage

### Custom Strategy Configuration
```bash
# Create custom strategy mix for specific research
poetry run python scripts/proposal_optimizer_unified.py optimize \
    --input "proposal.md" \
    --strategies "GRAPH_RAG,MULTIMODAL_RAG" \
    --domains "protein_research" \
    --mode research
```

### Batch Processing
```bash
# Process multiple proposals with unified configuration
poetry run python scripts/batch_optimizer_unified.py \
    --config unified_batch_config.yaml \
    --enable-cross-domain
```

### Performance Analysis
```bash
# Get unified RAG statistics
poetry run python scripts/proposal_optimizer_unified.py stats

# Analyze strategy performance
python -c "
from scripts.proposal_optimizer_unified import UnifiedProposalOptimizer
optimizer = UnifiedProposalOptimizer()
stats = optimizer.get_optimization_stats()
print('Strategy Performance:', stats['strategy_performance'])
print('Cross-Domain Success Rate:', stats['cross_domain_success_rate'])
"
```

## 🔗 Integration with Other Tools

### Samsung Grant Generator
```bash
# Generate Samsung grant with unified RAG
python -c "
import asyncio
from src.proposal.samsung_grant_generator_unified import create_unified_samsung_generator, SamsungGrantSpec, RiskLevel

async def generate():
    generator = create_unified_samsung_generator()
    await generator.initialize()

    spec = SamsungGrantSpec(
        title='AI-Powered Neurodevelopmental Research',
        research_area='AI Healthcare',
        primary_pi='Dr. Researcher',
        institution='Korean AI Institute',
        total_budget=500000000,
        duration_years=3,
        risk_level=RiskLevel.HIGH,
        innovation_keywords=['AI', 'neuroscience', 'ESM3'],
        knowledge_domains=['neuroscience', 'protein_research'],
        cross_domain_synthesis=True
    )

    proposal = await generator.generate_proposal(spec)
    print(f'Generated: {proposal.proposal_id}')
    print(f'Quality: {proposal.quality_metrics.get(\"overall_score\", 0):.3f}')

asyncio.run(generate())
"
```

### Direct Unified RAG Access
```bash
# Access unified RAG orchestrator directly
python -c "
import asyncio
from src.services.rag.unified_rag_orchestrator import create_unified_orchestrator, QueryContext, QueryComplexity, QueryDomain

async def test_rag():
    orchestrator = create_unified_orchestrator()
    await orchestrator.warmup()

    context = QueryContext(
        query='ESM3 protein folding applications neuroscience brain research',
        complexity=QueryComplexity.COMPLEX,
        domain=QueryDomain.NEUROSCIENCE,
        intent='synthesis',
        confidence=0.9
    )

    response = await orchestrator.search(context)
    print(f'Strategy: {response.strategy_used}')
    print(f'Confidence: {response.confidence:.3f}')
    print(f'Sources: {len(response.sources) if response.sources else 0}')

asyncio.run(test_rag())
"
```

## 📚 Additional Resources

- **Architecture Guide**: `UNIFIED_RAG_ARCHITECTURE.md`
- **API Documentation**: `docs/API_REFERENCE_UNIFIED.md`
- **Strategy Comparison**: `docs/RAG_STRATEGY_COMPARISON.md`
- **Cross-Domain Examples**: `examples/cross_domain_proposals/`
- **Performance Benchmarks**: `benchmarks/unified_rag_performance.md`

## 🆘 Support

For issues with Unified RAG Proposal Optimization:

1. **Check Strategy Health**: `scripts/unified_rag_health_check.py`
2. **Validate Configuration**: `scripts/validate_unified_config.py`
3. **Performance Debugging**: `scripts/debug_unified_performance.py`
4. **Strategy Analysis**: `scripts/analyze_strategy_performance.py`

---

**🎯 Result**: Samsung Future Technology Grant 1st Grade eligibility with cross-domain innovation bonus through intelligent 6-strategy RAG orchestration.**