# Brain-AI Convergence Grant — Expert Evaluation & Revision Plan (Nov 30, 2025)

## 1. Executive Summary (Grade: B-/C+)

Dr. Elena Rostova-Kim (MIT/Harvard Computational Neuro-AI) commends the ambition to fuse a 130B multimodal brain foundation model with RL-driven precision therapeutics, yet finds the proposal overstretches current scientific evidence, data availability, and clinical governance. Key neurodevelopmental mechanisms (e.g., synaptic pruning trajectories, glial-immune coupling, polygenic architecture) are only loosely tied to the AI agenda, and the reliance on a 3,000-subject cohort to power a 130B model is not credible even with Aurora-scale compute. Without a staged translational pathway, the risk of algorithmic overreach and ethical blowback is high. Verdict: **Revise & Resubmit after major reconstruction**.

## 2. Detailed Critique

| Dimension | Strengths | Critical Weaknesses |
|-----------|-----------|---------------------|
| **Scientific Validity** | • Recognizes heterogeneity across imaging, genomics, behavior, environment.  <br>• Mentions cutting-edge resources (EU HBP, BRAIN Initiative). | • Lacks mechanistic links between AI outputs and validated biomarkers (e.g., mTOR/PI3K pathways, microglial activation).  <br>• No plan for harmonizing multi-site acquisition noise (critical for DTI, fMRI in toddlers).  <br>• Overlooks 2024 findings on neuro-immune interactions and gut-brain axis in ASD. |
| **Technical Feasibility** | • Uses contemporary architectures (4D Swin, channel-equivariant nets, hierarchical RL). <br>• Mentions federated learning and PPO/PPO-style controllers. | • 3k longitudinal subjects cannot support pretraining or even PEFT of a 130B model without massive augmentation.  <br>• Aurora allocation unsubstantiated (no memorandum or quota).  <br>• RL stack lacks safety layers (constraint RL, verifiable reward shaping).  <br>• “Autonomous scientific reasoning” conflates GPT-5/Claude with domain-grounded discovery pipelines. |
| **Clinical Translatability** | • Targets full pipeline: prediction → diagnosis → therapy. <br>• Envisions digital twin for simulation. | • “Real-time neonatal prediction (AUC>0.95)” ignores base rate fallacy and false-positive harm.  <br>• Digital twin claim disregards inter-individual variability in structural maturation.  <br>• No shadow-mode or silent deployment stage; jumps straight to interventional RL. |
| **Innovation vs. Hype** | • Ambitious integration of RL, causal inference, generative models. | • Multiple buzzwords (multi-agent RL, inverse RL, CRISPR-RL) with no data pathways or safety proofs.  <br>• No differentiation between research prototypes and deployable clinical systems. |
| **Ethics & Safety** | • Mentions blockchain privacy, international data sharing. | • Missing plan for differential privacy, pediatric assent, or family counseling for “risk scores.”  <br>• No fairness audit across socio-economic strata within Korea.  <br>• RL treatment suggestions could violate medical device regulations without human oversight. |

## 3. Killer Questions for the PI

1. **Data-to-Model Alignment:** How will 3,000 multi-modal subjects (even with public augmentation) suffice to fine-tune a 130B NeuroX-Fusion derivative without catastrophic overfitting or hallucinated biomarkers? Provide concrete sample complexity calculations.
2. **Mechanistic Grounding:** Which validated molecular or circuit-level biomarkers (e.g., SHANK3 variants, cerebellar vermis microstructure) does the AI model anchor to when generating predictions, and how will these be experimentally verified?
3. **RL Safety Gatekeeping:** What formal safety constraints (Constrained MDPs, shielded RL, human override) govern the reinforcement-learning treatment engine before it interacts with pediatric patients?
4. **Digital Twin Fidelity:** How will you quantify and bound the uncertainty of the so-called “digital twin,” especially when extrapolating beyond observed developmental trajectories?
5. **Ethical/Risk Management:** What is the protocol for communicating early-risk predictions that may never materialize? How do you mitigate psychological harm and legal liability from false alarms?

**Verdict:** _Revise & Resubmit after major revision with mechanistic grounding, staged validation, and realistic compute/data strategy._

---

## 4. Revision Plan (Grounded in 2025 AI, Medicine & Neuroscience)

### Phase A — Scientific & Biomarker Grounding (0–3 months)
1. **Mechanistic Mapping Layer:** Introduce a *neurobiologically informed latent space* that explicitly encodes known developmental pathways (synaptic pruning indices, myelination curves, neuro-immune markers). Tie AI outputs to measurable biomarkers (e.g., CSF cytokines, diffusion kurtosis metrics).
2. **Polygenic & Multi-omic Integration:** Incorporate 2024–2025 polygenic risk scores (PRS-CSx updates) and glial transcriptomics to justify predictive claims; partner with Korean Genome Project for expanded allele frequencies.
3. **Uncertainty Quantification:** Reframe digital twin as a *probabilistic developmental trajectory engine* with Bayesian neural ODEs and calibrated conformal prediction intervals.

### Phase B — Data & Model Realignment (2–6 months, overlapping)
1. **PEFT Strategy:** Pivot from training a bespoke 130B model to parameter-efficient adaptation (LoRA, QLoRA, AdapterFusion) of **NeuroX-Fusion** or Gemini 3 BioMed variants. Document licensing and compute access.
2. **Data Expansion Plan:** 
   - Launch federated data partnerships (e.g., SingHealth, RIKEN) with secure enclaves.  
   - Deploy synthetic twin generation via diffusion-based morphometry augmentation validated against held-out cohorts.
3. **RL Safety Stack:** Adopt *Safe RL* frameworks (e.g., Constrained Policy Optimization + shield models) and **Reinforcement Learning from Clinical Feedback (RLCF)** prior to patient-facing deployment.

### Phase C — Clinical Translation Pathway (4–12 months)
1. **Shadow-Mode Trials:** Run silent prospective studies where AI predicts outcomes but human clinicians maintain authority; compare to standard-of-care to estimate uplift.
2. **Tiered Intervention Protocol:** Move from decision-support to semi-autonomous interventions only after IRB-approved adaptive trials demonstrating non-inferiority and safety.
3. **Ethical Guardrails:** Establish family counseling protocols, fairness dashboards, and pediatric assent workflows; add clinical ethicist + patient advocate to governance board.

### Phase D — Infrastructure & Budget Optimization
1. **Compute vs. Data Rebalance:** Allocate ≥40% of budget to high-fidelity longitudinal acquisition (wearables, home sensors, neuroimmune assays); rely on Aurora primarily for occasional large-scale finetuning rather than continuous pretraining.
2. **Verification & Compliance:** Incorporate regulation experts for MFDS (Korean FDA) and EU MDR alignment; plan for real-world evidence (RWE) capture.
3. **Documentation & Transparency:** Produce clinician-facing explainability reports (counterfactual heatmaps, causal graphs) and researcher-facing model cards detailing data provenance and limitations.

### Phase E — Deliverables & Metrics
- **Milestone 1 (M3):** Mechanistic mapping whitepaper + biomarker validation plan.  
- **Milestone 2 (M6):** PEFT-adapted NeuroX-Fusion prototype with uncertainty calibration on 1,000 held-out subjects.  
- **Milestone 3 (M9):** Safe RL treatment recommender running in shadow-mode with human override dashboard.  
- **Milestone 4 (M12):** Ethics & compliance dossier, including patient communication protocol and fairness audit.

Implementing this plan aligns the proposal with the state of AI medicine circa 2025, converting aspirational rhetoric into a credible translational roadmap.



