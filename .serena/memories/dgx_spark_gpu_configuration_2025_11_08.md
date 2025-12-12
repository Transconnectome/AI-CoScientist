# dgx-spark GPU Configuration - Verified 2025-11-08

## Accurate Hardware Specifications

**Platform**: NVIDIA GB10 (Grace Blackwell Superchip)

### Key Specifications
- **GPU**: NVIDIA GB10 Blackwell (5th gen Tensor Cores)
- **CPU**: 20-core ARM (10× Cortex-X925 + 10× Cortex-A725)
- **Memory**: **128GB unified LPDDR5X-9400** (shared CPU+GPU)
- **Memory Bandwidth**: 275 GB/s
- **AI Performance**: 1 PFLOP sparse FP4
- **Power**: 140W TDP
- **System RAM**: 119Gi (~125GB total, 113Gi available)

### Critical Correction

**WRONG** (previous documentation):
- 8x NVIDIA RTX 3090 (192GB total VRAM)
- Discrete GPUs with separate VRAM
- Ampere architecture

**CORRECT** (verified via nvidia-smi):
- 1x NVIDIA GB10
- 128GB unified memory (CPU+GPU shared)
- Blackwell architecture (latest generation)

## Unified Memory Advantages

1. **No PCIe bottleneck**: CPU and GPU share same 128GB pool
2. **Both models fit simultaneously**: DeepSeek-R1 (18.5GB) + Nemotron (14GB) = 32.5GB of 128GB
3. **Faster model switching**: No VRAM transfers needed
4. **95GB headroom**: Ample space for operations and future models
5. **Purpose-built for AI**: DGX Spark product line optimized for inference

## Implications for Connectome1 Upgrade

### Memory Budget
```
Total: 128GB unified memory
├─ DeepSeek-R1 32B:    18.5 GB
├─ Nemotron Nano 9B:   14.0 GB
├─ OS + System:        ~15 GB
├─ Connectome1:        ~5 GB
├─ ChromaDB:           ~3 GB
└─ Available:          ~72 GB (56% free)
```

### Performance Expectations
- Both models can stay loaded permanently
- No swapping overhead
- Unified memory eliminates PCIe transfers
- Blackwell architecture optimized for transformers
- FP4 precision support for efficiency

## Verification Commands
```bash
ssh dgx-spark "nvidia-smi"
ssh dgx-spark "free -h"
ssh dgx-spark "lspci | grep -i vga"
```

## Updated Documents
- ✅ CONNECTOME1_LLM_UPGRADE_RESEARCH.md (corrected)
- ✅ GPU_CONFIGURATION_CORRECTION.md (created)
- ⚠️ AI-CoScientist deployment docs (need update)

## Key Takeaway
The GB10 unified memory architecture is **superior** to discrete RTX 3090s for LLM inference. This platform is ideally suited for the proposed dynamic routing upgrade.
