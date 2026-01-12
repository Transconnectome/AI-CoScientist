Title*
Review: Major Concerns regarding Artifact Handling and Incremental Novelty

Paper Summary*
This paper presents **"TS-SpectrumNet"** (also termed **"EEG-TSSnet"**), a multi-scale 1D CNN and Wavelet-based framework for Image-from-EEG classification. The authors target the problem of decoding visual stimuli from brain signals, testing on both randomized (Ahmed et al.) and deterministic (Spampinato et al.) datasets. The proposed architecture combines a multi-branch 1D Convolutional module (TSDL) for temporal dynamics and a Wavelet-based module (SpecRL) for spectral representation. The authors report a doubling of state-of-the-art (SOTA) accuracy (**7.0% to 15.1%**) on the randomized task and **50.3%** on the deterministic task.

Paper Strengths*
*   **Performance Claims:** If valid, achieving **15.1%** on the difficult randomized EEG-ImageNet dataset would represent a statistically significant improvement over the stagnant baseline of $\approx 7\%$.
*   **Ablation Scope:** The paper attempts to break down contributions via ablation studies to analyze the impact of different kernel sizes and the spec-temporal combination.
*   **Computational Efficiency:** The model is relatively lightweight ($0.05M - 0.1M$ parameters) compared to heavy Transformer baselines like EEGChannelNet ($5M$ parameters), which is of interest for collecting lightweight EEG decoding models.

Major Weaknesses*
*   **Scientifically Questionable "Artifact" Handling (Neuro-AI Flaw):** The paper proposes **"Channel-wise Z-score normalization"** in Section 3.1 to handle artifacts. This is methodologically flawed. Normalization ($x - \mu / \sigma$) scales artifacts (EOG/EMG spikes) but preserves their *temporal morphology*. If a specific class of images evokes a "surprise" reaction (blink/saccade), the model will learn to classify the normalized blink rather than the neural signal. Given the massive performance jump ($7\% \rightarrow 15\%$), it is highly probable the model is exploiting these artifact-class correlations rather than learning visual semantics.
*   **Incremental AI Innovation:** The architecture (TSDL + SpecRL) lacks fundamental novelty. Multi-scale 1D CNNs are the standard inductive bias for EEG (e.g., EEGNet), and Wavelet transforms are a traditional signal processing technique. The combination does not represent the "Paradigm-Shifting" AI breakthrough that would be expected to yield a $100\%$ relative performance improvement on its own.
*   **Misleading Efficiency Comparisons:** While smaller than huge Transformers, the model ($0.05M$ params) is **5x larger** than the direct CNN competitor, EEGNet ($0.01M$). The paper glosses over this to claim superior efficiency, which is deceptive.

Minor Weaknesses*
*   **Nomenclature Inconsistency:** The paper refers to the model as **"TS-SpectrumNet"** in the Abstract/Title but **"EEG-TSSnet"** in Table 1.
*   **Visualization Issues:**
    *   Figure 1 uses generic ImageNet photos instead of explaining the trial structure.
    *   Figure 3 (EEG Signals) has microscopic axis labels and the signal is compressed to a solid block, making it impossible to verify "stable oscillatory waveforms."
    *   Figure 2 (Architecture) is cluttered and follows a tangled flow.
*   **Table 4 Caption:** The caption claims to show "artifact removal" effects, but the columns appear to show label cardinality/scalability.

Preliminary Recommendation*
1: Reject

Justification For Recommendation And Suggestions For Rebuttal*
**Justification:**
The recommendation is **Reject** due to a fatal methodological flaw in data handling. The reliance on Z-score normalization for artifact management likely introduces a confound where the model classifies muscle/ocular movements rather than neural signals. This undermines the validity of the reported **15.1% accuracy**. Furthermore, the architectural contribution is incremental, and the presentation suffers from nomenclature inconsistencies and poor visualization.

**Suggestions for Rebuttal:**
To change this opinion, the authors must address the artifact issue rigorously:
1.  **Control Experiment:** Perform an analysis where high-amplitude segments (presumed artifacts) are *zeroed out* or removed via ICA, rather than normalized. If the accuracy drops back to baseline ($\approx 7\%$), the gains are artifact-driven.
2.  **Clarify Nomenclature:** Choose one name (**TS-SpectrumNet** or **EEG-TSSnet**) and stick to it.
3.  **Fix Visualization:** Re-plot Figure 3 with readable axes and proper zooming to show the signal morphology.

Confidence Level*
5: Expert

Confidential Comments To AC
**Forensic Analysis (ECMARS):**
The system logic flagged this paper as **Meaningfulness: Incremental** and **Revision Potential: Low (0.4)**. The primary concern is the "Artifact Efficiency" claim. Forensic review suggests the massive performance jump is likely due to the model exploiting normalized artifact leakage (EOG/EMG) rather than genuine neural decoding improvement. Without a re-run of the data pipeline using proper ICA/Artifact Rejection, the results are likely spurious.