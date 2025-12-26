# Responses to Killer Questions (draft) — Nov 30 2025

## 1. Data-to-Model Alignment
- **Strategy:** Abandon full 130B pretraining. Instead adopt **PEFT/LoRA adaptation** of NeuroX-Fusion (130B) with Korean cohort as high-fidelity conditioning set.
- **Sample Complexity:** Empirical studies (NeuroX-Fusion 2025 release notes) show that LoRA at 0.1% parameters requires 2–5M token-equivalents (~10k multimodal episodes). We have 3,000 subjects × average 800 structured measurements ≈ 2.4M tokens. Augment with:  
  - Synthetic diffusion-based connectome augmentation (target +1.5M tokens)  
  - Federated partners (SingHealth, RIKEN, SNUH) supplying additional 2k cases.  
- **Regularization:** Use *hierarchical Bayesian fine-tuning* + *per-modality dropout* + *causal consistency losses* to prevent hallucinated biomarkers. Provide learning-curve simulations demonstrating expected generalization bounds.

## 2. Mechanistic Grounding
- **Biomarker Library:** Anchor latent dimensions to validated markers: SHANK3/SCN2A variants, cerebellar vermis FA, microglial TSPO PET signals, CSF cytokine ratios (IL-6/IL-10), gut microbiome enterotype indices.  
- **Linkage:** Implement multi-task heads that predict both clinical outcomes and biomarker levels; require consistency with known biological pathways (e.g., mTOR, PI3K-AKT).  
- **Verification:** Prospective sub-study with CSF + advanced DTI to confirm AI-generated biomarkers; replicate findings in zebrafish CRISPR knock-in models to validate causality.

## 3. RL Safety Gatekeeping
- **Safe RL Stack:** Constrained Markov Decision Process with medical guardrails; apply *Shielded PPO* where unsafe actions are clipped via clinician-defined rules.  
- **Human-in-the-Loop:** Deploy **Reinforcement Learning from Clinical Feedback (RLCF)** to ensure every suggested intervention is reviewed by pediatric neurologists before release.  
- **Validation Stages:** Shadow-mode → clinician-verified recommendation stage → limited interventional trial (single-arm) only after MFDS approval. Real-time override dashboard records every AI action.

## 4. Digital Twin Fidelity
- **Definition Shift:** Recast as *Probabilistic Developmental Trajectory Engine* built on Bayesian Neural ODEs + neural SDEs capturing stochastic growth.  
- **Uncertainty Bounds:** Provide conformal prediction intervals (95%) and calibration plots per phenotype; digital twin outputs always include variance and counterfactual comparisons.  
- **External Validation:** Validate on held-out cohorts (Korean + international) with diverse socio-economic and genetic backgrounds; measure Earth Mover’s Distance between simulated and observed trajectories.

## 5. Ethical/Risk Management
- **Communication Protocol:** Multi-stage disclosure — risk category (low/moderate/high) explained via trained counselors; families receive decision aids and follow-up support.  
- **False Positive Handling:** Mandatory re-evaluation before any label is recorded in medical charts; integrate psychological support to mitigate anxiety.  
- **Bias Monitoring:** Quarterly fairness audit across SES, regional, and genetic subgroups; publish transparency reports.  
- **Governance:** Add clinical ethicist + patient advocate to steering committee; align with MFDS and WHO AI in Health 2025 guidelines.



