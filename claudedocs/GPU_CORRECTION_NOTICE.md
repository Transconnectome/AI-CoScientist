# GPU Configuration Correction Notice

**Date**: 2025-11-08
**Issue**: Documentation incorrectly states dgx-spark has 8x RTX 3090 GPUs
**Severity**: High - affects deployment planning and resource assumptions

---

## Correction Required

### Incorrect Information (Multiple Files)

The following documents contain **incorrect GPU specifications** for dgx-spark:

1. **DEPLOYMENT_GUIDE_DYNAMIC_ROUTING.md**
   - States: "8x NVIDIA RTX 3090 GPUs (24GB VRAM each)"
   - Line 12, 217, 314, 459

2. **WEEK1_DEPLOYMENT.md**
   - References "8x RTX 3090" in troubleshooting section
   - Line 218

3. **DEPLOYMENT_LOG_20251104.md**
   - Historical log with incorrect GPU info
   - Multiple references

4. **scripts/monitor_routing_performance.py**
   - Comments about GPU utilization calculations
   - Based on 8 GPUs assumption

### Actual Configuration (Verified)

**dgx-spark has**:
- **1x NVIDIA GB10** (Grace Blackwell superchip)
- **128GB unified LPDDR5X memory** (shared CPU+GPU)
- **NOT 8x RTX 3090**

```bash
$ ssh dgx-spark "nvidia-smi"
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
+-----------------------------------------------------------------------------------------+
```

---

## Impact on Deployment

### What Changed

**Memory Capacity**:
- ❌ NOT 192GB discrete VRAM (8 × 24GB)
- ✅ **128GB unified memory** (CPU+GPU shared)

**Architecture**:
- ❌ NOT discrete Ampere GPUs with separate VRAM
- ✅ **Unified memory architecture** with Blackwell GPU

**Model Loading**:
- ✅ **Still works**: Both DeepSeek-R1 (18.5GB) + Nemotron (14GB) fit in 128GB
- ✅ **Actually better**: Unified memory eliminates PCIe overhead
- ✅ **More headroom**: 95GB remaining after loading both models

### Why This Is Good News

The GB10 platform is **superior** for LLM inference:

1. **Unified Memory**: No PCIe bottleneck between CPU and GPU
2. **Both models fit**: 32.5GB of 128GB = plenty of headroom
3. **Modern architecture**: Blackwell (2025) > Ampere (2020)
4. **Optimized for AI**: DGX Spark product line purpose-built for inference
5. **Power efficient**: 140W vs 350W per RTX 3090

---

## Files Requiring Update

### High Priority
- [ ] `DEPLOYMENT_GUIDE_DYNAMIC_ROUTING.md`
  - Update all "8x RTX 3090" references to "NVIDIA GB10"
  - Update VRAM calculations (192GB → 128GB unified)
  - Add note about unified memory advantages

- [ ] `WEEK1_DEPLOYMENT.md`
  - Update GPU references
  - Adjust troubleshooting for single GPU
  - Update verification commands

### Medium Priority
- [ ] `scripts/monitor_routing_performance.py`
  - Update GPU utilization calculation
  - Change from "8 GPUs" to "unified memory" monitoring
  - Update nvidia-smi queries

### Low Priority (Historical)
- [ ] `DEPLOYMENT_LOG_20251104.md`
  - Add correction notice at top
  - Keep for historical record
  - Note: "GPU specs corrected 2025-11-08"

---

## Recommended Updates

### DEPLOYMENT_GUIDE_DYNAMIC_ROUTING.md

**Replace**:
```markdown
- ✅ 8x NVIDIA RTX 3090 GPUs (24GB VRAM each)
```

**With**:
```markdown
- ✅ NVIDIA GB10 (Grace Blackwell, 128GB unified memory)
```

### GPU Monitoring

**Old approach** (8 discrete GPUs):
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

**New approach** (unified memory):
```bash
# GPU + system memory as unified pool
free -h  # System memory (includes GPU shared pool)
nvidia-smi  # GPU-specific info
```

---

## Deployment Plan Validation

Despite the GPU correction, the **deployment plan remains valid**:

✅ **Week 1**: Model installation works (128GB > 32.5GB needed)
✅ **Week 2**: Routing proxy unchanged (independent of GPU type)
✅ **Week 3**: Testing procedures unchanged
✅ **Week 4**: Monitoring adapted for unified memory

**No changes required** to core deployment strategy - only documentation updates.

---

## Verification

To verify dgx-spark GPU configuration:

```bash
# Basic info
ssh dgx-spark "nvidia-smi"

# Memory details
ssh dgx-spark "free -h && nvidia-smi --query-gpu=name,memory.total --format=csv"
```

**Expected output**:
```
NVIDIA GB10, [N/A]
Total System Memory: 119Gi
```

---

## References

- Connectome1 repository: `GPU_CONFIGURATION_CORRECTION.md` (detailed analysis)
- NVIDIA GB10 specs: [Technical details](https://wccftech.com/nvidia-gb10-superchip-soc-3nm-20-arm-v9-2-cpu-cores-nvfp4-blackwell-gpu-lpddr5x-9400-memory-140w-tdp/)

---

**Action Required**: Update documentation files listed above to reflect accurate NVIDIA GB10 specifications.

**Priority**: Medium (deployment works, but documentation should be accurate)

**Assignee**: Documentation update recommended for next maintenance cycle
