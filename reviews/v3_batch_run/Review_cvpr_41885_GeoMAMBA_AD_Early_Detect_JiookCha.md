Title*
Review: Statistically Improbable Results and Methodological Flaws in Small-Data Regime

Paper Summary*
The paper proposes **GeoMAMBA-AD**, a hybrid framework for early Alzheimer's Disease (AD) detection using resting-state fMRI (rs-fMRI). The method combines a non-trainable "John Ellipsoid-based Blind Source Separation" (**JE-BSS**) for preprocessing—borrowed from convex geometry—with a trainable **Mamba** (State Space Model) backbone. The authors claim this dual-pathway approach solves the "cross-scanner domain shift" problem, reporting **92% accuracy** and **100% recall** on a small, heterogeneous ADNI subset ($N=100$ total), significantly outperforming Graph Convolutional Networks (GCNs) and standard CNNs.

Paper Strengths*
*   **Novel Integration of Methods:** The integration of John Ellipsoid-based BSS with modern State Space Models (Mamba) is mathematically creative and unexplored in this domain.
*   **Architecture Design:** The dual-branch design (separating spatial source maps from temporal dynamic curves) theoretically aligns with the spatiotemporal nature of fMRI data.
*   **Visual Presentation:** Figure 1 (Architecture) is polished and clearly delineates the trainable vs. non-trainable components.

Major Weaknesses*
*   **Statistically Improbable Results :** The paper reports **92.00% Accuracy** and **1.00 Recall** (Table 1) on a cross-scanner test set derived from a total pool of 100 subjects. In the context of neuroimaging, achieving perfect sensitivity ($Recall=1.0$) on a test set of $\approx 25$ subjects is practically impossible without data leakage or severe overfitting. This result is an outlier in the entire history of ADNI-based classification.
*   **Model Complexity vs. Data Scale:** Training a parameter-rich hybrid model (3D CNN + Mamba) on roughly 75 subjects presents a non-trivial challenge. While the authors imply their architecture achieves superior data efficiency, differentiating between genuine efficient learning and overfitting in this regime is difficult. The paper would be strengthened by explicitly discussing the inductive biases that allow such a complex model to generalize from limited data, or by demonstrating robustness on an external dataset.
*   **Architectural Justification:** The authors motivate the use of Mamba (SSM) primarily for its long-sequence scaling properties ($O(L)$), yet apply it to relatively short rs-fMRI time series ($140-197$ points). While SSMs have theoretical advantages in modeling continuous dynamics, it is unclear if they offer a distinct benefit over standard Self-Attention ($O(L^2)$) or RNNs in this specific data regime. A direct comparison with a Transformer-based baseline is necessary to establish whether the Mamba backbone is contributing distinct value or simply adding unjustified complexity.
*   **Biological Plausibility:** JE-BSS relies on a geometric 'Pure Source' assumption—that clean neural sources exist at the vertices of the data's convex hull (simplex), similar to "100% soil" pixels in hyperspectral imaging. However, this is biologically flawed for fMRI. Brain networks (e.g., DMN, Salience) exhibit significant spatial overlap and non-linear interactions; there are rarely "pure" voxels that solely represent a single latent source. Furthermore, fMRI data is heavily corrupted by physiological noise (respiration, cardiac), which pushes data points outside any theoretical simplex. Thus, treating geometric vertices as robust neural biomarkers is a weak assumption.

Minor Weaknesses*
*   **Undisclosed Math:** The "4D Lifting Module" mentioned in Section 3.3 is mathematically opaque. It is unclear how spatial channels are mapped to temporal channels.
*   **Baseline Discrepancy:** The **24% gap** between the proposed method ($92\%$) and baselines ($\approx 68\%$) is notably wide. To ensure a fair comparison, it would be beneficial to provide additional details on how the hyperparameters for the baseline models were optimized, confirming that they were tuned with equal rigor to the proposed method.

Preliminary Recommendation*
1: Reject

Justification For Recommendation And Suggestions For Rebuttal*
**Justification:**
The recommendation is **Reject**. The reported results (**100% Recall on N=25**) are statistically incredible for this domain and strongly suggest data leakage (where the test set was seen during BSS feature selection) or memorization of site-specific artifacts. The application of Mamba to short functional sequences is technically unjustified, and the sample size is insufficient for the proposed architecture.

**Suggestions for Rebuttal:**
1.  **Leakage Audit:** Clarify strictly whether the JE-BSS projection matrix was learned on the *entire* dataset or only the training fold.
2.  **External Validation:** Validate the model on a completely external dataset (e.g., OASIS-3 or NACC) without retraining. If performance drops significantly (e.g., to $<70\%$), the current results are invalid.
3.  **Ablation:** Replace Mamba with a simple GRU/LSTM. If performance is identical, the use of Mamba is unnecessary.

Confidence Level*
5: Expert

Confidential Comments To AC
**Forensic Analysis:**
High Alert. The perfect Recall ($1.0$) on such a noisy medical dataset is a "Smoking Gun" for data leakage. It is highly likely the authors performed the BSS Unmixing on the full dataset *before* splitting, allowing the test set to leak into the projection matrix. This effectively invalidates the entire study. I strongly recommend rejection as the results are statistically impossible for this experimental design.