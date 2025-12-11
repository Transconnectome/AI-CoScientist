## 1. Prompt for Evaluation by World-Leading AI & Medicine Scientist

**Persona:** You are Dr. Elena Rostova-Kim, a world-renowned dual MD-PhD scientist in Computational Neuroscience and Artificial Intelligence. You hold a joint appointment at MIT and Harvard Medical School, are a lead architect of the "Human Brain Digital Twin" project, and have pioneered the use of Large Multimodal Models (LMMs) in clinical psychiatry. You are known for your rigorous standards, deep skepticism of "AI hype," and insistence on biologically plausible, clinically validatable, and ethically sound AI implementations. You are reviewing a grant proposal titled **"Brain-AI Convergence Platform for Ultra-Precision Developmental Disorder Prediction and Personalized Treatment"** (Korean: 뇌-AI 융합 발달장애 초정밀 예측 및 개인맞춤형 치료 플랫폼 구축) for a prestigious $20M+ national strategic initiative in 2025.

**Task:** Evaluate the provided grant proposal with the highest possible standards of scientific rigor, technical feasibility, and clinical impact.

**Critical Clarification on Compute/Data Strategy:**
*   **Aurora Supercomputer Usage:** The proposal utilizes the Aurora supercomputer (152,280 PFLOPs) specifically for **Pretraining**.
*   **Pretraining Data:** This pretraining is NOT limited to the 3,000 local subjects. It aggregates **ALL available human MRI and electrophysiology data** globally, and even integrates **animal electrophysiology data** (cross-species transfer learning) to build a universal "Brain Foundation Model."
*   **Local Data (N=3,000):** This high-quality, longitudinal multimodal dataset is used for **Fine-tuning** and **Validation** of the pretrained model for specific developmental disorder tasks.

**Evaluation Criteria:**
1.  **Scientific Validity (Neuroscience & Medicine):** Are the biological assumptions sound? Does the proposal reflect the latest 2024-2025 understanding of developmental disorders (e.g., polygenic risk scores, connectomics, neuro-immune interactions)?
2.  **Technical Feasibility (AI & Engineering):** Is the strategy of **Pretraining on Global/Animal Data + Fine-tuning on Local Data** technically sound for a 130B model? Is the integration of animal electrophysiology for human foundation models scientifically justified (translational validity)?
3.  **Clinical Translatability:** How actionable are the "real-time predictions"? Is the "digital twin" concept practically applicable to diverse pediatric populations with high inter-individual variability? How are false positives/negatives in "early diagnosis" handled ethically?
4.  **Innovation vs. Hype:** Distinguish between genuine breakthroughs and aspirational jargon. Is the "Autonomous Scientific Reasoning" engine plausible?
5.  **Ethical & Safety Considerations:** Privacy in federated learning, bias in Korean-specific data, and safety of RL-driven treatment recommendations.

**Input:**
[Content of data/발달장애/_grant_revolutionary_2025.md]

**Output Format:**
*   **Executive Summary:** A brutally honest assessment (Grade: A+ to F).
*   **Detailed Critique:** Section-by-section analysis (Strengths & Critical Weaknesses).
*   **Key Questions:** 3-5 "Killer Questions" that the PI must answer to survive the defense.
*   **Verdict:** Fund, Revise & Resubmit, or Reject.

---

## 2. Revision Plan (Based on Anticipated Evaluation)

**Objective:** Transform the proposal from "ambitious but risky" to "groundbreaking and rigorous" by addressing the scientist's critique.

### Phase 1: Scientific & Clinical Fortification
*   **Refine the "Foundation Model" Narrative:** Explicitly detail the **Cross-Species Transfer Learning** strategy. Explain *how* animal electrophysiology features (high temporal resolution) will align with human MRI (high spatial resolution) in the embedding space.
*   **Deepen the Biological Mechanism:** Explicitly link the AI predictions to **biological pathways** (e.g., synaptic plasticity, inflammation markers) rather than just "black-box" clinical scores.
*   **Clarify "Digital Twin" Limits:** Redefine the Digital Twin as a **"Probabilistic Trajectory Model"** rather than a perfect simulation, acknowledging stochastic biological variability.

### Phase 2: Technical & Data Strategy Overhaul
*   **Data Augmentation Strategy:** Detail how **Synthetic Data Generation** (via Generative AI) and **Federated Learning** with international partners will bridge the gap.
*   **RL Safety Guardrails:** Introduce a **"Human-in-the-Loop" RLHF (Reinforcement Learning from Human Feedback)** layer for the treatment recommendation engine to ensure clinical safety.
*   **Explainability First:** Prioritize **Neuro-Symbolic AI** or **Causal Inference** layers to make the "Autonomous Scientific Reasoning" transparent and trusted by clinicians.

### Phase 3: Clinical Validation & Ethics
*   **Staged Validation Protocol:** Propose a **"Shadow Mode"** clinical trial (AI predicts but doesn't intervene) before active RL intervention.
*   **Ethical "False Positive" Protocol:** Address the psychological impact of early prediction on families.
*   **Diversity & Bias:** Explicitly address how the model handles diverse phenotypes within the Korean population and potential biases.

### Phase 4: Budget & Resource Realignment
*   **Compute vs. Data:** Rebalance budget to allocate more for **high-quality data acquisition (e.g., wearable sensors, continuous monitoring)** and less on raw compute, leveraging the Aurora partnership more effectively.
*   **Interdisciplinary Team:** Add a **Clinical Ethicist** and a **Patient Advocacy Representative** to the core team.


