Title*
Review: Statistically Improbable Results and Methodological Flaws in Small-Data Regime

Paper Summary*
The paper proposes **GeoMAMBA-AD**, a hybrid framework for early Alzheimer's Disease (AD) detection using resting-state fMRI (rs-fMRI). The method combines a non-trainable "John Ellipsoid-based Blind Source Separation" (**JE-BSS**) for preprocessing—borrowed from convex geometry—with a trainable **Mamba** (State Space Model) backbone. The authors claim this dual-pathway approach solves the "cross-scanner domain shift" problem, reporting **92% accuracy** and **100% recall** on a small, heterogeneous ADNI subset ($N=100$ total), significantly outperforming Graph Convolutional Networks (GCNs) and standard CNNs.

Paper Strengths*
*   **Novel Integration of Methods:** The integration of John Ellipsoid-based BSS with modern State Space Models (Mamba) is mathematically creative and unexplored in this domain.
*   **Architecture Design:** The dual-branch design (separating spatial source maps from temporal dynamic curves) theoretically aligns with the spatiotemporal nature of fMRI data.
*   **Visual Presentation:** Figure 1 (Architecture) is polished and clearly delineates the trainable vs. non-trainable components.

Major Weaknesses*
*   **Statistically Improbable Results (Fatal):** The paper reports **92.00% Accuracy** and **1.00 Recall** (Table 1) on a cross-scanner test set derived from a total pool of 100 subjects. In the context of neuroimaging, achieving perfect sensitivity ($Recall=1.0$) on a test set of $\approx 25$ subjects is practically impossible without data leakage or severe overfitting. This result is an outlier in the entire history of ADNI-based classification.
*   **Insufficient Sample Size:** Training a parameter-heavy model (3D CNN + Mamba) on roughly 75 subjects guarantees overfitting. The "small-data" claim is a methodological limitation, not a feature. Deep learning models require magnitudes more data to learn generalized features.
*   **Mamba Misapplication (AI Critique):** Mamba/SSMs are designed for very long sequences ($10k+$ tokens). rs-fMRI time series are short ($140-197$ points). Using Mamba here provides negligible computational advantage over attention and adds unnecessary complexity ("trend-chasing").
*   **Biological Plausibility (Neuro-AI Critique):** JE-BSS assumes data lies in a convex hull of pure sources. Brain networks overlap non-linearly. This geometric assumption is weak for fMRI BOLD signals corrupted by physiological noise.

Minor Weaknesses*
*   **Visual Clarity:** Figure 2 (Attribution Maps) shows tiny "yellow spots" that do not map to standard AD biomarkers (DMN/Salience networks), failing to support the "interpretable" claim.
*   **Undisclosed Math:** The "4D Lifting Module" mentioned in Section 3.3 is mathematically opaque. It is unclear how spatial channels are mapped to temporal channels.
*   **Baseline Discrepancy:** The **24% gap** between the proposed method ($92\%$) and baselines ($\approx 68\%$) suggests the baselines were improperly tuned or tested on different splits.

Preliminary Recommendation*
1: Reject

Justification For Recommendation And Suggestions For Rebuttal*
**Justification:**
The recommendation is **Strong Reject**. The reported results (**100% Recall on N=25**) are statistically incredible for this domain and strongly suggest data leakage (where the test set was seen during BSS feature selection) or memorization of site-specific artifacts. The application of Mamba to short functional sequences is technically unjustified, and the sample size is insufficient for the proposed architecture.

**Suggestions for Rebuttal:**
1.  **Leakage Audit:** Clarify strictly whether the JE-BSS projection matrix was learned on the *entire* dataset or only the training fold.
2.  **External Validation:** Validate the model on a completely external dataset (e.g., OASIS-3 or NACC) without retraining. If performance drops significantly (e.g., to $<70\%$), the current results are invalid.
3.  **Ablation:** Replace Mamba with a simple GRU/LSTM. If performance is identical, the use of Mamba is unnecessary.

Confidence Level*
5: Expert

Confidential Comments To AC
**Forensic Analysis:**
High Alert. The perfect Recall ($1.0$) on such a noisy medical dataset is a "Smoking Gun" for data leakage. It is highly likely the authors performed the BSS Unmixing on the full dataset *before* splitting, allowing the test set to leak into the projection matrix. This effectively invalidates the entire study. I strongly recommend rejection as the results are statistically impossible for this experimental design.