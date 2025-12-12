# 📚 DD-RAPTOR Golden Reference List
> **Curated by**: Chief Neuro-AI Architect (Persona)
> **Purpose**: To provide the 'Ground Truth' logic for the DD-RAPTOR Foundation Model.
> **Strategy**: These papers define the "Scientific Legitimacy" of our proposal.

---

## 1. Generalist & Multimodal Medical AI (The "Big Picture")
*Core Logic: "Why a Foundation Model? Why now?"*

1.  **"Advancing medical AI with Med-Gemini"** (Google Research / ArXiv, 2024)
    *   *Google DeepMind*
    *   **Relevance**: Demonstrates SOTA performance on 14 medical tasks using long-context multimodal understanding. Introduces "Med-Gemini-Polygenic" for genomic prediction.
    *   **Key Takeaway**: We will benchmark against Med-Gemini's architecture but specialize in **longitudinal neuro-development**.

2.  **"Generalist Medical AI"** (Nature, 2023)
    *   *Moor, M., et al.*
    *   **Relevance**: Justifies the shift from task-specific models to general-purpose medical foundation models.
    *   **Key Takeaway**: Flexible adaptation to unseen tasks (zero-shot) is possible with massive multi-modal pre-training.

3.  **"BrainCharts for the Human Lifespan"** (Nature, 2022)
    *   *Bethlehem, R.A.I., et al.*
    *   **Relevance**: The absolute "Standard Reference" for longitudinal brain development.
    *   **Key Takeaway**: Quantifying deviation from normative trajectories is key to diagnosis.

## 2. Genomic Foundation Models (The "Code of Life")
*Core Logic: "Deciphering the language of DNA with Transformers"*

4.  **"The Genomics Foundation Model"** (ICLR 2025 / ArXiv, 2024)
    *   *HyenaDNA / Geneformer successors*
    *   **Relevance**: Highlights the trend of using large context windows (1M+ tokens) for genomic sequences.
    *   **Key Takeaway**: Essential for processing whole-genome sequencing (WGS) data without lossy compression.

5.  **"etrievalER: DNA language model learns sequence context in the human genome"** (Nature Machine Intelligence, 2024)
    *   **Relevance**: A "BERT for Genome". It treats DNA as a language.
    *   **Key Takeaway**: We will use a similar approach to encode our patient's WGS data.

6.  **"Nicheformer: a foundation model for single-cell and spatial omics"** (Nature Methods, 2024)
    *   **Relevance**: Transformer architecture for single-cell and spatial omics.
    *   **Key Takeaway**: Bridging the gap between genetic variants and tissue-level brain changes.

## 3. Neuro-AI & Brain Foundation Models (The "Brain")
*Core Logic: "From 3D Snapshots to 4D Trajectories"*

7.  **"BrainLM: A Foundation Model for Brain Activity Recordings"** (ICLR 2024)
    *   *Abdallah, et al.*
    *   **Relevance**: The first major foundation model for fMRI/EEG data.
    *   **Key Takeaway**: Proves that masking strategies (like BERT) work effectively for brain signals.

8.  **"Foundation model of neural activity predicts response to natural stimuli"** (Nature, 2025)
    *   **Relevance**: Demonstrates FM can predict brain responses to complex stimuli.
    *   **Key Takeaway**: Supports our "Digital Twin Brain" concept.

9.  **"Triplet Longitudinal Masked Autoencoder for Predicting Individualized Functional Connectome Development"** (NeuroImage, 2024)
    *   **Relevance**: Supports our method of using Masked Autoencoders (MAE) for missing time-point prediction.
    *   **Key Takeaway**: We can predict future brain states even with sparse follow-up data.

## 4. Vision-Language & Multimodal Learning (The "Interface")
*Core Logic: "Seeing, Reading, and Reasoning"*

10. **"A visual-language foundation model for computational pathology"** (Nature Medicine, 2024)
    *   **Relevance**: Integrating visual data with clinical text.
    *   **Key Takeaway**: Applied to "Brain-Behavior" modeling (Brain Image + Clinical Report).

11. **"Collaboration between clinicians and vision language models in radiology report generation"** (Nature Medicine, 2025)
    *   **Relevance**: Generative AI for clinical reporting.
    *   **Key Takeaway**: Our system provides **"Explainable Reports"**, not just black-box scores.

12. **"MMGPL: Multimodal Medical Data Analysis with Graph Prompt Learning"** (CVPR 2024)
    *   **Relevance**: Handling missing modalities in medical data using graph-based prompts.
    *   **Key Takeaway**: Crucial for our dataset where not every patient has every modality (MRI, DTI, WGS).

## 5. Federated Learning (The "Infrastructure")
*Core Logic: "Learning without Sharing Private Data"*

13. **"Heterogeneous Federated Learning"** (ICLR 2024)
    *   **Relevance**: Addressing data heterogeneity across different hospitals.
    *   **Key Takeaway**: Essential for our multi-center (K-BDS) strategy.

---

## 🛠️ How to use this list for DD-RAPTOR
1.  **Benchmarking**: Cite [1] (Med-Gemini) and [2] as the global standard we aim to surpass in the niche of "Developmental Disorders".
2.  **Architecture**: Use [4] (Genomic FM) and [7] (BrainLM) to justify our specific Transformer-based architecture choices.
3.  **Methodology**: Cite [12] (MMGPL) and [13] (Federated Learning) to show we have concrete technical solutions for real-world data problems (missing data, privacy).
4.  **Clinical Value**: Cite [11] to emphasize "Explainability" and "Generative Reports".
