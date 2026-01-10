## CVPR Review

**Paper Title:** GeoMAMBA-AD: Early Detection of Alzheimer’s Disease Using Non-Invasive Resting-State fMRI and John Ellipsoid-Driven MAMBA Network

---

### 1. Summary

This paper proposes GeoMAMBA-AD, a novel framework for early Alzheimer's disease (AD) detection from resting-state fMRI (rs-fMRI) data. The core contribution is a two-stage approach designed to tackle the challenges of small data availability and cross-scanner domain shift. The first stage is a non-trainable, geometry-driven blind source separation (BSS) method based on the John ellipsoid (JE). This JE-BSS is applied at the individual-subject level to extract a small set of scanner-invariant spatial sources and their corresponding temporal dynamics. The second stage is a trainable deep learning model that uses a Mamba-based architecture to process these sources for AD classification. The architecture features separate pathways for spatial and temporal information, which are later fused. The authors claim this is the first work to combine geometry-based BSS with a Mamba backbone for this task, achieving 92% accuracy in cross-scanner, small-data settings.

### 2. Strengths

*   **Novel Methodological Combination:** The paper introduces a novel and interesting combination of techniques. The use of John ellipsoid-based BSS as a pre-processing step to create scanner-invariant features is a clever and well-motivated approach to the pervasive domain shift problem in multi-site fMRI studies. This geometry-based method avoids the strong statistical independence assumptions of traditional ICA, which may not hold for fMRI data.
*   **Tackling a Critical Problem:** The work addresses a highly relevant and challenging problem in medical imaging AI: robust model generalization from limited and heterogeneous data. The focus on cross-scanner performance for AD detection is of significant clinical and practical importance.
*   **Adoption of Modern Architectures:** The integration of the Mamba architecture, a recent and efficient selective state-space model, for learning from fMRI data is timely and well-justified. Its linear complexity in handling long sequences is a potential advantage over Transformers for high-dimensional temporal data like fMRI.
*   **Interpretability:** The authors make an effort to connect the learned geometric sources to known neurobiological systems relevant to AD, which is a commendable step towards clinical validation and trust in the model's predictions.

### 3. Weaknesses

*   **Missing Experimental Validation:** The most critical flaw of the submission is the complete absence of an experimental section in the provided manuscript. The paper makes strong quantitative claims (e.g., 92% accuracy) in the abstract but provides no data, baselines, ablation studies, or results section to substantiate them. Without this, the work is unverifiable and its contributions cannot be assessed.
*   **Lack of Clarity in Methodology:** Several key components of the proposed Mamba-based architecture are poorly explained, hindering reproducibility. For instance:
    *   The "4D lifting module" that maps source channels to temporal channels (Eq. 6) is not described in sufficient detail.
    *   The rationale and implementation of the "sink tokens" and the subsequent feature extraction from the Mamba blocks (Eq. 9-11) are confusing.
    *   The motivation for several hyperparameters (e.g., number of sources `p=4`, latent dimension `K=16`) is either weakly justified or absent, lacking sensitivity analysis or ablation.
*   **Potentially Strong Claims without Sufficient Support:** The abstract claims that JE-BSS can "provably extract the perfect sources under rather mild data purity conditions." While this may be true in signal processing theory, its direct application and validity for complex, noisy biological signals like rs-fMRI BOLD requires much stronger justification and empirical evidence within the paper.
*   **Limited Discussion of Broader Context:** The paper focuses on a specialized, task-specific model for a "small-data" regime. While pragmatic, it does not engage with the broader trend in medical AI towards large-scale pretraining and foundation models, which have shown remarkable success in improving generalization and data efficiency.

### 4. Detailed Comments

#### Methodology
*   **JE-BSS Stage:** The motivation for JE-BSS is clear. However, the choice of `p=4` sources seems arbitrary. While references are provided, the paper would benefit from an ablation study showing how performance varies with the number of extracted sources. Is there a risk of information loss by compressing the entire fMRI signal into just four components?
*   **Mamba Architecture:** The architecture design in Section 3.3 is overly complex and inadequately explained.
    *   **Eq. 6:** What is the exact operation of the "4D lifting module" `ϕ(·)`? Is it a reshaping operation, a 1x1x1x1 convolution, or something else? This is a critical and non-standard step that requires a precise definition.
    *   **Eq. 9-11:** The use of sink tokens followed by extracting the "last token positions" is unconventional. Typically, sequence models use a class token or mean/max pooling over the output sequence. Please clarify this mechanism and justify its advantages over standard approaches. The fusion with an adaptive average pooling branch via a learnable gate `α` adds further complexity that needs to be justified through ablation studies.
    *   **Temporal Conditioning:** The final paragraph of the methods section seems to be cut off. This section, describing how the temporal dynamic curves (TDCs) condition the final representation, is crucial for understanding the full model and is unfortunately missing.

#### Experiments (Critique based on what is missing)
As the experimental section is absent, I must outline the expected components that are necessary for this paper to be considered for publication at CVPR:
1.  **Datasets:** A clear description of the rs-fMRI datasets used, including the number of subjects, sites/scanners, acquisition parameters, and demographics for the training and testing sets. Standard datasets like ADNI would be expected.
2.  **Baselines:** A rigorous comparison against a comprehensive set of baseline methods is required. This should include:
    *   Traditional machine learning methods with handcrafted fMRI features.
    *   Established deep learning models for fMRI AD classification (e.g., 3D-CNNs, GCNs, Transformers) such as the cited `gICA 3D-VGG` [30] and spatiotemporal graph transformers [29].
    *   Ablated versions of your own model to demonstrate the contribution of each component.
3.  **Ablation Studies:** These are essential to validate the design choices:
    *   **GeoMAMBA-AD vs. Mamba-only:** Demonstrate the benefit of the JE-BSS pre-processing stage.
    *   **JE-BSS vs. ICA-BSS:** Compare against a more traditional source separation method like ICA to show the superiority of the geometry-based approach.
    *   **Mamba vs. Transformer/LSTM:** Replace the Mamba backbone with other sequence models to validate its effectiveness.
    *   **Full Architecture vs. Individual Branches:** Evaluate the performance of using only the spatial or temporal pathways.
4.  **Interpretability Results:** The claim that sources capture executive control and subcortical systems needs to be supported by visualizing the average spatial maps of the four sources and comparing them with known resting-state networks from neuroscientific literature.

#### Writing
The overall writing quality is good. The introduction and related work sections are well-written and situate the problem effectively. However, the methodology section sacrifices clarity for brevity, leaving key parts of the model ambiguous.

### 5. Comparison with SOTA

The provided reference context primarily discusses **BiomedCLIP** [Ref 1-4], a vision-language foundation model pretrained on a massive and diverse dataset of 15 million biomedical image-caption pairs. This work represents a different, albeit highly successful, paradigm in medical AI.

*   **Contrasting Paradigms:** GeoMAMBA-AD proposes a specialized, supervised model tailored for a specific task (AD classification) and modality (rs-fMRI) under "small-data" constraints. In contrast, BiomedCLIP exemplifies the "large-scale pretraining" paradigm, where a generalist foundation model is trained on a massive, diverse dataset and then adapted to various downstream tasks.
*   **Generalization Approach:** GeoMAMBA-AD aims for generalization by explicitly designing a scanner-invariant pre-processing step (JE-BSS). BiomedCLIP achieves strong generalization through the sheer scale and diversity of its pretraining data (PMC-15M), which allows it to learn robust representations that transfer well even to out-of-domain tasks (e.g., outperforming a radiology-specific model like BioViL on a radiology benchmark).
*   **Novelty in Context:** While GeoMAMBA-AD's approach is different, it is not necessarily inferior, especially for modalities like fMRI where datasets of the scale of PMC-15M do not exist. The novelty of the proposed method is sound within its specific domain. However, the paper would be strengthened by acknowledging the foundation model trend and discussing where its specialized approach fits. Could JE-BSS, for example, serve as a canonical representation method to enable large-scale pretraining across many heterogeneous fMRI datasets in the future?

Without an experimental section, it is impossible to compare GeoMAMBA-AD's performance to any SOTA, including the domain-specific ones it cites (e.g., [28, 29, 30]).

### 6. Overall Rating & Confidence

*   **Overall Rating: 2 (Weak Reject)**
*   **Confidence: 4/5**

**Justification:** The paper presents a novel and promising idea for a very important problem. The combination of geometric BSS and Mamba is innovative. However, the submission is critically incomplete due to the missing experimental section. The claims of state-of-the-art performance are entirely unsubstantiated. Furthermore, the methodology section lacks the clarity and detail required for reproducibility and rigorous scientific assessment. While the core idea has merit, the paper in its current form falls far short of the standards for publication at CVPR. I would be willing to reconsider my rating if a comprehensive experimental validation were provided, but as it stands, I cannot recommend acceptance.