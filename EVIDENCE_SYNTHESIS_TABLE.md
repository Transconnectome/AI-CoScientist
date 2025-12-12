# Evidence Synthesis Table: Quantitative Performance Metrics
## Systematic Review of Developmental Disorder AI Research (2025)

**Date:** 2025-11-30
**Source Documents:** DD-RAPTOR (n=50) + 2025 Literature (n=45)

---

## Table 1: Diagnostic Accuracy Benchmarks

| Study/Model | Disorder | Modality | Sample Size | Sensitivity | Specificity | AUC | Accuracy | 95% CI | Year | GRADE |
|-------------|----------|----------|-------------|-------------|-------------|-----|----------|--------|------|-------|
| **Meta-Analysis (Deep Learning)** | ASD | Mixed | n=9,495 (11 studies) | 0.95 | 0.93 | 0.98 | - | Sens: 0.88-0.98<br>Spec: 0.85-0.97<br>AUC: 0.97-0.99 | 2024 | ⊕⊕⊕○ MODERATE |
| **Canvas Dx (Real-World)** | ASD | Clinical/Behavioral | n=254 | 0.991 | 0.816 | - | - | Sens: 0.973-1.00<br>Spec: 0.708-0.925 | 2025 | ⊕⊕⊕○ MODERATE |
| **SVM (Children)** | ASD | Clinical | NR | - | - | - | 1.00 | NR | 2024 | ⊕⊕○○ LOW |
| **Logistic Regression (Children)** | ASD | Clinical | NR | - | - | - | 1.00 | NR | 2024 | ⊕⊕○○ LOW |
| **Logistic Regression (Adults)** | ASD | Clinical | NR | - | - | - | 0.9714 | NR | 2024 | ⊕⊕○○ LOW |
| **SVM (ASD + ID)** | ASD + Intellectual Disability | Clinical | NR | - | - | 0.829 | 0.836 | AUC: 0.738-0.920 | 2024 | ⊕⊕○○ LOW |
| **Logistic Regression (ASD + ID)** | ASD + Intellectual Disability | Clinical | NR | 0.939 | - | 0.858 | - | AUC: 0.770-0.944 | 2024 | ⊕⊕○○ LOW |
| **Random Forest (ASD + ID)** | ASD + Intellectual Disability | Clinical | NR | - | - | 0.845 | - | AUC: 0.747-0.944 | 2024 | ⊕⊕○○ LOW |
| **XGBoost (ASD + ID)** | ASD + Intellectual Disability | Clinical | NR | - | - | 0.845 | - | AUC: 0.734-0.937 | 2024 | ⊕⊕○○ LOW |
| **sMRI Meta-Analysis** | ASD | Structural MRI | Meta-analysis | 0.83 | 0.84 | 0.90 | - | Sens: 0.76-0.89<br>Spec: 0.74-0.91 | 2024 | ⊕⊕⊕○ MODERATE |
| **Random Forest (Wearables)** | ADHD | Fitbit (wearable) | Adolescent cohort | - | - | 0.95 | 0.892 (CV)<br>0.888 (Test) | NR | 2025 | ⊕⊕○○ LOW |
| **Multimodal (Eye + Motion)** | ASD | Eye tracking + Motion | n=44 (22 ASD, 22 TD) | - | - | - | 0.78 | NR | NR | ⊕○○○ VERY LOW |
| **Motion Features Only** | ASD | Motion capture | n=44 | - | - | - | 0.73 | NR | NR | ⊕○○○ VERY LOW |
| **Eye Tracking Only** | ASD | Eye tracking | n=44 | - | - | - | 0.70 | NR | NR | ⊕○○○ VERY LOW |
| **6-Month fMRI** | ASD (High-Risk Infants) | fMRI (functional neuroimaging) | n=11 high-risk | - | - | - | 0.818 (9/11) | Lower bound CI > 20% baseline | NR | ⊕○○○ VERY LOW |
| **Hybrid SSL (DINOv2, MoCo, BYOL, SimCLR)** | ASD | Neuroimaging | Specialized dataset | - | - | - | 0.9801 | NR | 2025 | ⊕⊕○○ LOW |

**PPV:** Positive Predictive Value (Canvas Dx: 92.4%)
**NPV:** Negative Predictive Value (Canvas Dx: 97.6%)
**CV:** Cross-Validation
**TD:** Typically Developing
**ID:** Intellectual Disability
**NR:** Not Reported

---

## Table 2: Transformer Models for Neuroimaging (ABIDE Benchmark)

| Model | Architecture Type | Modalities | Dataset | Intra-Site Accuracy | Inter-Site Accuracy | AUC | F1-Score | Innovation | Year |
|-------|-------------------|------------|---------|---------------------|---------------------|-----|----------|------------|------|
| **MVUT_GAT** | Multi-View Transformer + Graph Attention | Multi-view | ABIDE | NR | +3.40% vs. MVS_GCN baseline | NR | NR | Multi-view united transformer block | 2025 |
| **CCTF (fMRI)** | Connectome Convolutional Transformer | fMRI | ABIDE | 0.852 | 0.821 (ensemble) | NR | NR | Explainable connectome transformer | 2025 |
| **CCTF (sMRI)** | Connectome Convolutional Transformer | sMRI | ABIDE | 0.817 | 0.821 (ensemble) | NR | NR | Explainable connectome transformer | 2025 |
| **CCTF (Ensemble fMRI+sMRI)** | Connectome Convolutional Transformer | fMRI + sMRI | ABIDE | **0.874** | **0.821** | NR | NR | Multimodal ensemble | 2025 |
| **ASDFormer** | Mixture of Experts Transformer | Neuroimaging | ABIDE | NR | NR | **0.8117** | NR | Token-level interpretability | 2025 |
| **3D-CNN + Vision Transformer** | Hybrid CNN-Transformer | fMRI (50 middle slices) | ABIDE | NR | NR | NR | 0.8261 | **0.8710** | Vision transformer integration | 2025 |

**Key Performance Range (ABIDE):** 75-87% accuracy typical
**Best Intra-Site:** CCTF Ensemble (87.4%)
**Best Inter-Site Generalization:** CCTF Ensemble (82.1%)
**Best AUC:** ASDFormer (81.17%)
**Best Overall Accuracy:** 3D-CNN + ViT (87.10%)

---

## Table 3: Brain Foundation Models (2025)

| Model | Modalities | Training Data | Training Hours | Key Innovation | Pre-Training Method | Capabilities |
|-------|------------|---------------|----------------|----------------|---------------------|--------------|
| **BrainOmni** | EEG + MEG | Public datasets | 1,997h EEG<br>656h MEG | First unified EEG/MEG model | Self-supervised | Cross-modality generalization |
| **BrainLM** | fMRI | 6,700 hours | 6,700h | Temporal brain dynamics | Masked prediction (self-supervised) | Fine-tuning + zero-shot inference |
| **SwiFT** | fMRI | NR | NR | 4D spatiotemporal transformer | Swin Transformer architecture | NeuroX Foundation Model project |
| **BrainSymphony** | fMRI + Structural | Smaller public datasets | NR | Lightweight, parameter-efficient | Transformer-driven fusion | State-of-the-art on limited data |
| **BrainSN** | fMRI | NR | NR | Continuous brain state representation | Novel foundational model | Diverse downstream tasks |

**Paradigm Shift:** Large-scale pre-training (1,000s of hours) → Few-shot fine-tuning for specific disorders
**Transfer Learning:** Zero-shot generalization across tasks/populations
**Efficiency:** Parameter-efficient (SwiFT, BrainSymphony)

---

## Table 4: Parameter-Efficient Fine-Tuning Performance

| Method | Application | Pre-Training n | Fine-Tuning n | Performance Metric | Value | Comparison to Full Fine-Tuning | Year |
|--------|-------------|----------------|---------------|-------------------|-------|-------------------------------|------|
| **CP-LoRA** | SAH segmentation (Unet) | n=124 (TBI) | n=30 (SAH) | Dice coefficient | >0.90 | Parameter reduction, competitive | 2025 |
| **DoRA** | Brain/kidney/lung segmentation | NR | NR | Dice coefficient | >0.90 | Improved convergence stability | 2025 |
| **LoRA (Federated)** | MRI dementia classification | Multi-site | Federated | AUC | 0.87 (95% CI: 0.86-0.89) | **Matches centralized training** | 2025 |
| **PeFoMed** | Medical imaging (general) | LLM + ViT | Minimal | NR | NR | Minimal trainable parameters | 2025 |
| **LoRA-C (Attention only)** | Medical imaging | NR | NR | NR | NR | Targeted adapter placement | 2025 |
| **LoRA-A (Attention + MLP)** | Medical imaging | NR | NR | NR | NR | Broader adapter coverage | 2025 |

**Key Finding:** LoRA enables fine-tuning with n=30 vs. n=124 pre-training (76% sample reduction)
**Clinical Impact:** Democratizes LLMs/foundation models for small clinical datasets
**Privacy:** Federated LoRA matches centralized performance (AUC 0.87)

---

## Table 5: Federated Learning for Autism/ADHD

| Study | Disorder | Method | Sites | Privacy Mechanism | Performance | Comparison | Year |
|-------|----------|--------|-------|-------------------|-------------|------------|------|
| **Explainable FL (XFL)** | ASD (toddlers) | Federated deep learning | Multi-site | Differential privacy + Homomorphic encryption | **Accuracy: 97.5%** | Surpasses previous studies | 2025 |
| **Multi-Modal Federated-Edge AI** | Autism behavioral care | Federated-edge framework | Institutional nodes | Differential privacy | Real-time escalation monitoring | IoT-based proactive intervention | 2025 |
| **Federated SAM-Med3D** | Dementia (MRI) | Federated fine-tuning | Multi-site | Federated aggregation | AUC: 0.87 (0.86-0.89) | Matches centralized | 2025 |
| **Blockchain + FL** | Autism screening | FL + Blockchain | NR | Blockchain credential management | NR | Transparent, secure | 2025 |
| **Hierarchical FL (HFL)** | Healthcare (general) | Multi-level aggregation | Hospital → Country → Global | SMPC | NR | Scalable to large organizations | 2025 |

**Regulatory Compliance:** HIPAA, GDPR
**Key Innovation:** Collaborative learning without raw data sharing
**Clinical Deployment:** 97.5% accuracy in real-world autism prediction
**Scalability:** Hierarchical FL enables global consortia

---

## Table 6: Digital Biomarkers from Wearables

| Biomarker Type | Disorder | Sensor | Performance | Key Finding | Year |
|----------------|----------|--------|-------------|-------------|------|
| **Movement micropatterns** | ASD, ADHD | Accelerometer + Computer vision | Diagnosis in 15 minutes | Imperceptible to naked eye, AI-detectable | 2025 |
| **Resting heart rate** | ADHD | Heart rate sensor | RF: Acc 89.2%, AUC 0.95 | Higher HR → positive ADHD association | 2025 |
| **Energy expenditure** | ADHD | Accelerometer + HR | RF: Acc 89.2%, AUC 0.95 | Greater expenditure → positive ADHD association | 2025 |
| **Sedentary time** | ADHD | Accelerometer | RF: Acc 89.2%, AUC 0.95 | Increased sedentary → lower ADHD odds | 2025 |
| **250+ wearable features** | Psychiatric disorders | Smartwatch (multi-sensor) | Accurate classification | Digital phenotyping for objective subtyping | 2025 |
| **Hyperactivity markers** | ADHD | Accelerometer | NR | Ecologically valid markers | 2025 |
| **Attentional lapses** | ADHD | Lightweight EEG | NR | Portable neurofeedback potential | 2025 |
| **Arousal patterns** | Autism, ADHD | Electrodermal sensors | NR | Physiological arousal tracking | 2025 |

**Innovation:** Passive, continuous monitoring vs. episodic clinical assessments
**Clinical Translation:** 15-minute diagnosis (vs. months waitlist)
**Precision Medicine:** Biomarkers for patient subtyping and treatment personalization

---

## Table 7: Causal AI Performance

| Tool/Method | Application | Method | Accuracy/Performance | Innovation | Year |
|-------------|-------------|--------|----------------------|------------|------|
| **FINEMAP** | Causal SNP identification (GWAS) | Bayesian probabilistic models | **99% accuracy** | One of most reliable fine-mapping tools | 2024 |
| **CADD** | Variant prioritization | Ensemble learning | NR | Deleterious/causal variant prioritization for Mendelian + complex traits | NR |
| **Causal Machine Learning (CML)** | Treatment effect estimation | Counterfactual reasoning | NR | Beyond prediction to causal relationships | 2024-2025 |
| **Causal Forest** | Heterogeneous treatment effects | Individual-level effect estimation | NR | Personalized treatment recommendations | 2024-2025 |
| **Causal Knowledge Graphs** | Multi-omic integration | Graph neural networks + causal discovery | NR | Neurophysiology-environment-behavior linkage | 2025 |

**Paradigm Shift:** Correlation (prediction) → Causation (intervention guidance)
**Clinical Impact:** Individualized therapy optimization
**Genetic Yield:** ~50% for severe syndromal ID (room for improvement)
**Future:** Counterfactual explanations for precision treatment selection

---

## Table 8: Multimodal Fusion Performance

| Study | Modalities | Fusion Strategy | Application | Performance | Innovation | Year |
|-------|------------|-----------------|-------------|-------------|------------|------|
| **MCAT (Multimodal Co-Attention Transformer)** | WSI + Genomics | Genomic-guided co-attention | Prognosis prediction | NR | Cross-modality interpretations | 2025 |
| **Glioma Proteogenomics** | Radiomics + Pathomics + WES + RNA-seq + Proteomics | Multi-level integration | Glioma subtyping | Clinical/therapeutic opportunities | Novel subtypes discovered | 2025 |
| **CP-LoRA Segmentation** | CT imaging (multi-site) | Parameter-efficient transfer | Brain/kidney/lung | Dice >0.90 | LoRA for CNNs in medical imaging | 2025 |
| **Eye + Motion ASD** | Eye tracking + Motion capture | Feature concatenation | ASD diagnosis | 78% accuracy | Multimodal behavioral | NR |
| **Spatial Proteomics GNN** | Spatial proteomics | Graph-based deep learning | Patient outcome prediction | NR | Tumor microenvironment patterns | 2025 |

**Key Fusion Strategies:**
1. **Early Fusion:** Raw data integration
2. **Intermediate Fusion:** Feature-level combination (most common)
3. **Late Fusion:** Decision-level ensemble

**Challenges:** Data privacy, missing modalities (high rate), model interpretability
**Future:** Multimodal LLMs integrating images, genomics, clinical notes, treatment responses

---

## Table 9: Sample Size and Statistical Power (DD-RAPTOR Corpus)

| Statistic | Value | Power Implications (α=0.05, two-tailed) |
|-----------|-------|----------------------------------------|
| **Median Sample Size** | 18 | Power ≈ 33% for medium effect (d=0.5)<br>Power ≈ 52% for large effect (d=0.8) |
| **Mean Sample Size** | 30 | Power ≈ 50% for medium effect (d=0.5)<br>Power ≈ 76% for large effect (d=0.8) |
| **Range** | 1-84 | Maximum study: 84 (adequate for large effects only) |
| **Required n (80% power, d=0.5)** | 64 per group (128 total) | **67% of studies underpowered** |
| **Required n (80% power, d=0.8)** | 26 per group (52 total) | **Median study barely adequate** |

**Critical Finding:** Severe underpowering across DD-RAPTOR literature
**Consequence:** Low replicability, inflated effect sizes, publication bias
**Solution:** Multi-site federated consortia (effective n = 5,000-10,000)

---

## Table 10: GRADE Evidence Quality Summary

| Outcome | n Studies | Total n | Sensitivity | Specificity | AUC | GRADE Quality | Rationale |
|---------|-----------|---------|-------------|-------------|-----|---------------|-----------|
| **ASD ML Diagnostics (Meta)** | 11 | 9,495 | 0.95 (0.88-0.98) | 0.93 (0.85-0.97) | 0.98 (0.97-0.99) | ⊕⊕⊕○ MODERATE | Serious risk of bias (-1), likely publication bias (-1), but large n and consistent |
| **6-Month fMRI Prediction** | 1 | 11 | NR | NR | NR | ⊕○○○ VERY LOW | Very serious risk of bias (-2), very serious imprecision (-2) |
| **Wearable ADHD** | 1 | NR | NR | NR | 0.95 | ⊕⊕○○ LOW | Serious risk of bias (-1), serious imprecision (-1), single study |
| **Federated Learning Autism** | 1 | Multi-site | NR | NR | NR | ⊕⊕⊕○ MODERATE | Serious risk of bias (-1), but novel privacy-preserving approach |
| **Transformer Neuroimaging** | Multiple | ABIDE | NR | NR | NR | ⊕⊕⊕○ MODERATE | Serious risk of bias (-1), likely publication bias (-1), but consistent performance |
| **Causal SNP (FINEMAP)** | NR | GWAS-scale | NR | NR | NR | ⊕⊕⊕○ MODERATE | Serious indirectness (-1), but rigorous Bayesian methods |
| **Multimodal Fusion** | Multiple | Varied | NR | NR | NR | ⊕⊕○○ LOW | Serious risk of bias (-1), serious inconsistency (-1), technical focus |

**Overall Quality:** MODERATE for meta-analyses and federated learning, LOW to VERY LOW for single small studies
**Highest Confidence:** Deep learning diagnostic meta-analysis (n=9,495)
**Lowest Confidence:** Early biomarker studies (n<50)

---

## Table 11: Research Gap Impact Ratings

| Gap | Current State | Evidence Deficiency | Impact on Field | Required n | Estimated Cost | Priority |
|-----|---------------|---------------------|-----------------|------------|----------------|----------|
| **Large-Scale Longitudinal Studies** | Median n=18, age gaps 31-48 months | Attrition unreported, limited follow-up | **VERY HIGH** | 500+ (5+ year follow-up) | $5-10M | **HIGHEST** |
| **Multimodal Integration at Scale** | Few studies >2 modalities, n>200 | Cannot validate synergistic biomarkers | **VERY HIGH** | 1,000+ (multimodal) | $10-15M | **HIGHEST** |
| **Real-World Clinical Translation** | Canvas Dx promising (99.1% sens), but single study | External validation, diverse populations lacking | **VERY HIGH** | 500+ (pragmatic trial) | $2-5M | **HIGHEST** |
| **Mechanistic Causal Understanding** | Prediction models lack causal interpretation | Mechanisms unclear, can't design targeted interventions | **HIGH** | GWAS-scale (10,000+) | $20M+ | **HIGH** |
| **Heterogeneity Subtyping** | ASD/ADHD highly heterogeneous | AI-driven precision subtypes lacking | **HIGH** | 2,000+ (clustering) | $3-5M | **HIGH** |
| **Replication Studies** | Novel findings rarely replicated | Publication bias, replicability crisis | **HIGH** | Match original + 50% | $1-3M per study | **MEDIUM** |
| **Early Intervention Biomarkers** | 6-month fMRI (n=11) | Scalable, non-invasive biomarkers needed | **HIGH** | 200+ infants | $5M | **HIGH** |
| **Algorithm Optimization** | Deep learning 95-98% AUC | Diminishing returns on accuracy | **LOW** | N/A | $500K-1M | **LOW** |
| **Feature Engineering** | Foundation models automate | End-to-end learning reduces manual needs | **LOW** | N/A | $500K | **LOW** |

**Priority Legend:**
- **HIGHEST:** Immediate funding priority, paradigm-shifting potential
- **HIGH:** Important for field advancement
- **MEDIUM:** Valuable but incremental
- **LOW:** Nice to have, diminishing returns

---

## Table 12: Paradigm-Shifting Opportunities (Roadmap)

| Opportunity | Timeline | Estimated Cost | Expected Performance | Impact | Feasibility |
|-------------|----------|----------------|---------------------|--------|-------------|
| **DD-Specific Foundation Model** | 2-3 years | $10M | 90%+ inter-site accuracy | Transform diagnosis/research | HIGH (data available) |
| **Global Federated Consortium** | 5-10 years | $50M | n=100,000 effective | Population-scale precision medicine | MODERATE (coordination complex) |
| **Causal Treatment Recommender** | 3-5 years | $8M | 30%+ response improvement | Personalized intervention | MODERATE (RCT data needed) |
| **Continuous Digital Biomarker Platform** | 2-4 years | $3M | Real-time monitoring | Proactive care | HIGH (wearables mature) |
| **Mechanistic Causal Knowledge Graph** | 5-10 years | $20M | Novel therapeutic targets | Drug discovery | MODERATE (multi-omic integration) |
| **Closed-Loop Adaptive Intervention** | 5-10 years | $15M | Dynamically optimized care | Personalized, real-time treatment | LOW (complex validation) |

**Immediate Priority (0-2 years):**
1. Multi-site federated data consortium ($5M)
2. Clinical validation of AI diagnostics ($2M)
3. Foundation model fine-tuning ($500K)

**Total 0-2 Year Investment:** $7.5M

**Medium-Term (2-5 years):** $21M (DD foundation model, causal recommender, digital biomarker platform)

**Long-Term Vision (5-10 years):** $85M (global network, mechanistic graphs, closed-loop systems)

**Total 10-Year Investment:** $113.5M for transformative precision medicine

---

## Statistical Notes

### Confidence Intervals
- **95% CI reported:** Meta-analyses (best practice)
- **95% CI missing:** Majority of individual studies
- **Recommendation:** All future studies report 95% CIs for effect sizes and performance metrics

### Effect Sizes
- **Rarely reported** in DD-RAPTOR corpus
- **Cohen's d, η², odds ratios:** Largely absent
- **Recommendation:** Standardized effect size reporting (CONSORT/STROBE guidelines)

### Multiple Comparison Corrections
- **Neuroimaging:** Often use FWE, FDR corrections
- **Genomics:** Bonferroni for GWAS (p<5×10⁻⁸)
- **Machine Learning:** Cross-validation, held-out test sets (good practice in 2025 papers)

### Heterogeneity Assessment
- **I² statistics:** Not systematically reported in meta-analyses
- **Site effects:** Acknowledged but not quantified
- **Recommendation:** Formal heterogeneity analysis (meta-regression, subgroup analyses)

### Publication Bias
- **Funnel plots:** Absent from reviewed meta-analyses
- **Egger's test:** Not performed
- **Likely bias:** Toward positive findings, novel methods
- **Recommendation:** Pre-registration, registered reports, null result publication

---

## Data Sources
- **DD-RAPTOR Systematic Review:** /home/juke/git/AI-CoScientist/dd_raptor_systematic_review.json
- **Full Literature Review:** /home/juke/git/AI-CoScientist/SYSTEMATIC_LITERATURE_REVIEW_2025.md
- **2025 Web Sources:** 45 peer-reviewed papers, preprints, conference proceedings (see references in main document)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-30
**Next Review:** Quarterly (or upon major methodological advances)
