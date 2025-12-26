# Compute & Data Strategy (PEFT + Federated + Aurora) — v0.1

## Goals
1. Achieve scalable adaptation of NeuroX-Fusion 130B without full retraining.
2. Expand effective dataset beyond local 3k cohort via federated + synthetic augmentation.
3. Optimize Aurora supercomputer usage for periodic heavy lifts while keeping day-to-day work on local clusters.

## Components

### 1. Parameter-Efficient Fine-Tuning (PEFT)
- **Technique:** LoRA/QLoRA + AdapterFusion on NeuroX-Fusion pretrained weights.  
- **Scope:** Freeze 99.9% of parameters; train modality-specific adapters (imaging, genomics, behavior).  
- **Infrastructure:**  
  - Local DGX nodes for experimentation (precision BF16).  
  - Aurora bursts (2-week windows/quarter) for large-scale adapter sweeps and hyperparameter searches.  
- **Outcome:** Reduce compute requirement by >95% compared to full-model finetuning; enable rapid iteration.

### 2. Federated & Privacy-Preserving Data Expansion
- **Partners:** SingHealth (SG), RIKEN CBS (JP), SNUH (KR), EU BRAIN sub-cohorts.  
- **Framework:** Secure enclave with homomorphic encryption gradients (OpenFL 2025).  
- **Governance:** Data never leaves host sites; model updates aggregated using differential privacy (ε ≤ 3).  
- **Target Scale:** +2,000 international subjects in Year 1; +5,000 by Year 3.

### 3. Synthetic & Simulation Augmentation
- **Diffusion-Based Connectome Generator:** Conditional diffusion models to create plausible DTI/EEG combinations, validated against held-out statistics.  
- **Agent-Based Environmental Simulation:** Generate socio-environmental trajectories for stress/adversity scenarios to stress-test digital twin.  
- **Quality Control:** Use Fréchet Multimodal Distance (FMD) to ensure synthetic data stays within distributional tolerances (<5% deviation).

### 4. Aurora Utilization Plan
- **Allocation:** 15M node-hours/year secured via INCITE NeuroX-Fusion partnership (LoI in progress).  
- **Use Cases:**  
  - Quarterly PEFT mega-sweeps (adapter search + LoRA rank tuning).  
  - Large-scale uncertainty calibration (ensembles of trajectory models).  
  - Federated aggregator simulations under varying latency/failure conditions.  
- **Monitoring:** Real-time telemetry dashboard; cost per experiment tracked for budgeting.

### 5. Operational Stack
- **Orchestrator:** Kubeflow 2.0 for workflow management across local + Aurora resources.  
- **Data Lake:** Delta Lake on on-prem object storage with automated lineage metadata.  
- **Compliance:** Continuous audit logs, model cards, and data provenance tracking per MFDS/WHO AI guidelines.

## Deliverables
1. **Compute/Data Master Plan** (Dec 2025): Architecture diagrams + budget alignment.  
2. **Federated Pilot Report** (Feb 2026): Demonstrate cross-site training with differential privacy.  
3. **Aurora Runbook** (Mar 2026): SOP for scheduling, monitoring, and cost attribution.



