# Phase A Plan — Mechanistic Mapping & Uncertainty Program (Dec 2025 – Feb 2026)

## Objectives
1. **Mechanistic Latent Space:** Embed validated neurodevelopmental pathways (synaptic pruning, myelination, neuro-immune coupling) directly into the AI backbone.
2. **Uncertainty-Aware Digital Twin:** Deliver probabilistic developmental trajectory models with calibrated confidence intervals.

## Workstreams

### WS1. Mechanistic Feature Atlas
- **Tasks**
  - Curate multi-scale biomarker panels (genomics, proteomics, imaging, neurophysiology).
  - Engineer pathway-aware latent factors using Graph Neural Networks aligned to KEGG/Reactome pathways.
  - Implement joint training objective tying latent factors to both observed outcomes and mechanistic markers.
- **Owners:** Neurobiology PI + AI modeling lead.

### WS2. Poly-Omic Integration & Validation
- **Tasks**
  - Integrate 2024–2025 polygenic risk score toolkits (PRS-CSx, MegaPRS) customized for Korean ancestry.
  - Fuse glial transcriptomics and gut microbiome enterotypes via multi-view VAEs.
  - Run cross-validation to ensure biomarkers remain predictive across cohorts.
- **Owners:** Genomics lead + Data integration team.

### WS3. Probabilistic Trajectory Engine
- **Tasks**
  - Build Bayesian Neural ODE/SDE models for growth trajectories.
  - Apply conformal prediction + temperature scaling for calibration.
  - Benchmark against historical cohorts to verify calibration error <3%.
- **Owners:** Applied math team + Clinical statisticians.

### WS4. Verification & Reporting
- **Tasks**
  - Design mechanistic validation study (CSF + advanced DTI subset).
  - Draft clinician-facing explainer (heatmaps, causal graphs).
  - Prepare Milestone 1 deliverable: Mechanistic mapping whitepaper.
- **Owners:** Clinical translation lead + Documentation team.

## Timeline
| Month | Milestone |
|-------|-----------|
| Dec 2025 | Complete biomarker atlas + pathway graph |
| Jan 2026 | Deliver calibrated probabilistic engine prototype |
| Feb 2026 | Submit whitepaper + validation protocol to reviewers |

## Dependencies & Risks
- **Data Access:** Ensure IRB approval for CSF + transcriptomic datasets (submit amendments by Dec 10).  
- **Compute:** Allocate 0.5M GPU-hours on Aurora for PEFT experiments; confirm slot with INCITE liaison.  
- **Mitigation:** If data delays occur, use public ASD multi-omic datasets (e.g., SFARI) for pre-validation.



