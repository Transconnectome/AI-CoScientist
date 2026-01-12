# ECMARS Dashboard
- **Meaningfulness**: **Incremental**
- **Revision Potential**: **0.4** (Low/Medium - The visual and nomenclatural sloppiness indicates a lack of rigor, and the "artifact normalization" strategy raises concerns about the validity of the performance gains that may require a fundamental re-evaluation of the data pipeline.)
- **Decision**: **REJECT**

---

### 1. Summary
The paper proposes "TS-SpectrumNet" (referred to inconsistently as "EEG-TSSnet"), a deep learning framework designed to classify natural images from EEG signals. The authors target two specific datasets: the randomized EEG-ImageNet (Ahmed et al.) and the deterministic EEG-ImageNet (Spampinato et al.). The architecture combines a multi-branch 1D Convolutional module (TSDL) for temporal dynamics and a Wavelet-based module (SpecRL) for spectral representation. The authors claim a state-of-the-art (SOTA) accuracy of 15.1% on the randomized task (doubling the previous 7.0%) and 50.3% on the deterministic task.

### 2. Strengths
*   **Performance Claims:** If valid, achieving 15.1% on the randomized EEG-ImageNet dataset is a statistically significant improvement over the stagnant baseline of ~7%.
*   **Ablation Scope:** The paper attempts to break down contributions via ablation studies (though the presentation of these results in Table 4 is suboptimal).
*   **Computational Efficiency:** The model is relatively lightweight (0.05M - 0.1M parameters) compared to heavy Transformer baselines like EEGChannelNet (5M parameters).

### 3. Weaknesses
*   **Nomenclature Inconsistency:** The paper inconsistently refers to the model as **"TS-SpectrumNet"** (title/abstract) and **"EEG-TSSnet"** (Table 1). While minor, this lack of polish creates unnecessary confusion.
*   **Questionable "Artifact" Handling:** The "Channel-wise normalization" (Section 3.1) is presented as a novel contribution to handle artifacts. However, simple Z-score normalization does not *remove* artifacts (like EOG/EMG); it merely scales them. If the model performance jumps from 7% to 15% based on this, there is a high probability the model is learning to classify the *scaled artifacts* (e.g., subject blinking or clenching in response to specific stimuli) rather than neural correlates.
*   **Misleading Efficiency Claims:** The authors claim superior efficiency. While smaller than EEGChannelNet, Table 1 shows the proposed model (0.05M) is **5x larger** than the direct competitor EEGNet (0.01M). The text glosses over this, framing the comparison primarily against the largest models.
*   **Space Utilization:** Figure 1 displays generic ImageNet photos, which offers limited scientific insight. This space could be more effectively used to illustrate the distinction between Randomized and Deterministic trial structures.

### 4. Figures and Tables
The clarity and presentation of figures and tables fall below the standards required for CVPR.
*   **Figure 1 (Stimuli):** **Remove.** We know what a "parachute" looks like. Replace this with a diagram illustrating the difference between Randomized vs. Deterministic trial structures, which is central to your problem statement.
*   **Figure 2 (Architecture):** Cluttered. The SpecRL module flow is tangled. The connection between the Wavelet Transform (WT) and the Inverse (IWT) needs to be linearized for readability.
*   **Figure 3 (EEG Signals):** **Unacceptable.** The axes labels are microscopic. Subplot (a) is compressed to a solid block of color; no oscillation is visible. You cannot claim to show "stable oscillatory waveforms" if the plot resolution renders them invisible.
*   **Table 1 (The "Branding" Error):** As noted, you call your model **EEG-TSSnet** here, but **TS-SpectrumNet** everywhere else. This creates confusion—are these different ablation variants?
*   **Table 4:** While providing raw accuracy numbers for every class increment is valuable for precision, the trend would be much clearer if accompanied by a line plot (X=Classes, Y=Accuracy). Furthermore, the caption claims it shows the effect of **"artifact removal,"** but the content appears to show **"label cardinality."** This inconsistency should be resolved to ensure clarity.

### 5. Relation to SOTA & Novelty
*   **Context:** The paper correctly identifies the difficulty gap between Randomized (SOTA ~7%) and Deterministic (SOTA ~48%) trials.
*   **Novelty:** The architecture is essentially an Inception-style CNN combined with a Wavelet transform. This is **Incremental**. Multi-scale 1D CNNs are standard in EEG processing (e.g., EEGNet, InceptionEEG). The addition of Wavelets is a known technique. The claim of "Paradigm-Shifting" performance relies entirely on the 15.1% accuracy figure, which, as noted in the Neuro-AI section below, is suspect.

### 6. Neuro-AI Considerations
The biological validity of the "Artifact" section (3.1) warrants scrutiny.
*   **The Artifact Fallacy:** Figure 3(b) shows high-amplitude spikes, likely ocular (EOG) or muscular (EMG) artifacts. The paper proposes Z-score normalization to "minimize the impact."
*   **Critique:** Normalization ($x - \mu / \sigma$) brings these spikes into the same numerical range as brain signals, but it preserves the *temporal morphology* of the artifact. If a specific class of images causes a "surprise" reaction (blink/saccade), and you normalize that blink, the CNN will simply learn the shape of the normalized blink. You are likely classifying muscle movements, not visual processing. A rigorous approach requires Independent Component Analysis (ICA) or regression-based artifact *removal*, not just normalization.
*   **Kernel Sizes:** You use kernels of size 63, 127, and 255. At standard EEG sampling rates (e.g., 1000Hz), a 255 kernel covers 250ms. This is biologically plausible for capturing ERP components (P300, N400), but it is computationally expensive for the first layer.

### 7. Detailed Feedback
1.  **Uniform Nomenclature:** Choose **TS-SpectrumNet** or **EEG-TSSnet** and strictly adhere to it. The current inconsistency appears sloppy and undermines the professional quality of the manuscript.
2.  **Rewrite Section 3.1:** You must prove that your normalization isn't preserving artifact-class correlations. Run an experiment where you *zero out* high-amplitude segments instead of normalizing them. If accuracy drops back to 7%, your model relies on artifacts.
3.  **Convert Table 4 to a Plot:** Show the scalability trend visually. Fix the caption error immediately.
4.  **Figure 3:** Re-plot with readable fonts (10pt+). Zoom in on the x-axis for Subplot (a) so we can see the waves.
5.  **Comparison Fairness:** In Table 1, explicitly acknowledge that EEGNet is 5x smaller than your model. Do not hide this.

### 8. Final Recommendation
The paper reports a significant performance jump, but the lack of rigor in presentation (naming inconsistencies, caption errors) and the scientifically suspect method of handling artifacts (normalization vs. removal) undermine confidence in the results. The visual presentation is currently not publication-ready.

**Decision: REJECT**