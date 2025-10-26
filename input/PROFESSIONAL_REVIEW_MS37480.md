# Professional Peer Review
## Manuscript 37480_0: "Sensorimotor circuit connectivity as a candidate biomarker for responsiveness to selective serotonin reuptake inhibitors in obsessive–compulsive disorder"

**Journal:** Neuropsychopharmacology
**Review Date:** October 26, 2025
**Reviewer Role:** Expert in Neuroimaging Biomarkers & OCD Research
**Recommendation:** **MAJOR REVISIONS**

---

## Summary Assessment

This manuscript addresses a clinically important question—predicting SSRI response in OCD—using state-of-the-art neuroimaging methodology. The focus on the sensorimotor circuit is theoretically well-motivated, connecting to recent ENIGMA consortium findings and animal models of serotonergic function. The methodological approach is rigorous, employing standardized preprocessing (CONN Toolbox), appropriate statistical corrections (FWE cluster-level), and a well-characterized drug-naïve sample. The finding that responders show hyperconnectivity while non-responders show hypoconnectivity within the sensorimotor circuit is novel and converges with independent parallel work (Wang et al., 2025, J Affect Disord).

However, the paper has significant limitations that prevent acceptance in its current form. The sample size (N=54) is below current standards for biomarker validation studies (see NPP 2025 special issue recommendations, particularly Marek & Laumann, 2025). More critically, the absence of external validation coupled with high prediction accuracy (AUC 0.82-0.86, R² up to 0.48) raises concerns about overfitting and generalizability. Without validation, these findings represent exploratory discovery rather than definitive biomarker identification.

**Recommended Action:** Major revisions to recalibrate claims, add internal validation analyses, compare with clinical predictors, and substantially expand the limitations discussion. Alternatively, reject and invite resubmission after multi-site validation study.

---

## Major Strengths

### 1. Clinical Significance and Novelty
The paper addresses a genuine unmet clinical need. With approximately 50% of OCD patients showing inadequate response to SSRIs (as authors note), a predictive biomarker could substantially improve treatment outcomes by enabling early identification of non-responders. The focus on baseline functional connectivity as a predictor is clinically practical, requiring only a standard 8-minute resting-state fMRI scan that is widely available.

The novel contribution is identifying **hyperconnectivity** in responders within the sensorimotor circuit, challenging the prevailing view of hypoconnectivity in OCD. This bidirectional pattern (hyper in responders, hypo in non-responders) suggests heterogeneity in neural substrates and aligns with emerging circuit-based subtyping approaches (cf. Drysdale et al., 2017, Nature Medicine for depression biotypes).

### 2. Theoretical Integration
The sensorimotor circuit focus is exceptionally well-justified through multiple lines of evidence:
- **ENIGMA consortium findings** (Bruin et al., 2023): Widespread sensorimotor hypoconnectivity across 28 cohorts
- **Neuropsychological evidence:** Response inhibition deficits engaging sensorimotor regions
- **Serotonergic innervation:** Dense raphe nuclei projections to thalamus and sensorimotor cortex
- **Preclinical models:** 5-HT1B receptor modulation of sensorimotor gating
- **Neuroplasticity evidence:** SSRIs broaden network-level plasticity in sensorimotor circuits

This multilevel integration from molecules → circuits → behavior → treatment is exemplary and positions the work within a coherent mechanistic framework.

### 3. Methodological Rigor
The preprocessing and statistical approach are exemplary:

**Preprocessing Excellence:**
- CONN Toolbox 22.v with SPM12 (current standard pipelines)
- Comprehensive artifact detection (ART: FD >0.9mm, BOLD >5 SD)
- CompCor for physiological noise (5 components each WM/CSF)
- Appropriate bandpass filtering (0.017-0.1 Hz)
- Standardized atlases (Kong 2022, Harvard-Oxford)

**Statistical Rigor:**
- Appropriate seed-based approach for hypothesis-driven analysis
- Conservative thresholding (voxel p<0.001, cluster FWE p<0.05)
- Bonferroni correction for post-hoc comparisons
- FDR correction for multiple predictors in regression
- Transparent reporting of effect sizes (R², OR, AUC, 95% CI)

**Sample Characteristics:**
- 100% drug-naïve (eliminates medication confound)
- Strict exclusion criteria (no comorbid psychosis, severe depression, substance use)
- Matched controls on demographics
- Standardized treatment protocol (sertraline titration to max tolerated dose)
- Standard responder definition (≥35% Y-BOCS reduction)

These methodological strengths place the work among the most rigorous in the OCD neuroimaging literature.

### 4. Converging Evidence
An important strength not adequately emphasized by the authors: **independent replication** of the sensorimotor hypothesis. Wang et al. (2025, J Affect Disord) published parallel findings using a different metric (zfALFF vs functional connectivity) in a similar drug-naïve OCD cohort treated with sertraline. The convergence across methods (activation vs connectivity) strengthens confidence that the sensorimotor circuit genuinely predicts SSRI response, rather than being a methodological artifact.

This cross-method triangulation is powerful evidence and should be highlighted prominently in the discussion as it directly addresses replication concerns.

### 5. Cerebellar Findings
The unexpected finding of altered cerebellar-sensorimotor connectivity adds depth to the understanding of OCD circuits. The involvement of both sensorimotor (lobules VI, VIIb) and non-sensorimotor (Crus I) cerebellar regions suggests the cerebellum may integrate both motor and cognitive processing relevant to treatment response. This is consistent with emerging views of the cerebellum's role in higher-order cognition and psychiatric disorders (Van Overwalle, 2024, Nat Rev Neurosci).

---

## Major Concerns Requiring Response

### 1. **CRITICAL: Absence of External Validation**

This is the most significant limitation preventing acceptance. The paper proposes functional connectivity as a **"candidate biomarker"** (title) and concludes it **"holds promise as a neuroimaging biomarker to guide individualized pharmacological treatment"** (abstract). However, all analyses were conducted within a single sample without external validation.

**Why this is critical:**
- NPP 2025 standards (Marek & Laumann): External validation increasingly standard for biomarker claims
- Small sample (N=54) increases risk of capitalizing on chance
- High prediction accuracy (AUC 0.82-0.86) without validation suggests potential overfitting
- Extreme odds ratios (ORs from <0.001 to >70,000) indicate model instability
- Wide 95% confidence intervals noted by authors but still concerning

**Current field standard** (per NPP 2025 special issue): Biomarker validation requires demonstration in independent sample, preferably from different site/scanner. Within-sample cross-validation (k-fold, leave-one-out) is minimum expectation even in absence of external data.

**Required Actions:**
1. **Minimum (for major revisions):**
   - Perform internal cross-validation (leave-one-out or 10-fold)
   - Report cross-validated accuracy metrics
   - Recalibrate claims from "biomarker" to "candidate requiring validation"
   - Add substantial discussion of generalizability limitations

2. **Preferred (for acceptance without major concerns):**
   - Obtain independent validation cohort (even N=20-30 would strengthen claims substantially)
   - Demonstrate prediction accuracy holds in new data
   - Test across different scanners/sites

3. **Alternative framing:**
   - Reposition as **"exploratory discovery"** rather than biomarker validation
   - Emphasize hypothesis generation for future validation studies
   - Focus on mechanistic insights rather than clinical prediction

**Without addressing this concern, the paper cannot be accepted as a biomarker study**, regardless of other strengths.

---

### 2. **Sample Size Below Current Standards**

The sample of N=54 OCD patients (33 responders, 21 non-responders) is below emerging standards for psychiatric neuroimaging:

**NPP 2025 benchmarks** (Marek & Laumann, 2025):
- N ≥100 for reliable brain-behavior associations
- N ≥1000 for population-level generalizations
- Small samples (N<50) show elevated false positive rates

**Comparable recent studies:**
- Wang et al. (2025): Similar N, published in J Affect Disord (IF~6)
- Treatment prediction with 177 OCD/PTSD patients (NeuroImage: Clinical, 2025)
- Multi-site studies achieving N>100 are becoming standard

**Assessment:**
- N=54 is **adequate for exploratory** discovery (passes minimum threshold)
- N=54 is **insufficient for definitive** biomarker validation
- Imbalanced groups (33 vs 21) reduces statistical power for group comparisons
- Limited ability to detect effect size heterogeneity or moderators

**Impact on interpretation:**
- Findings should be considered **preliminary** pending replication
- Effect sizes may be overestimated (regression to mean in small samples)
- Generalizability to broader OCD population uncertain
- Particularly concerning given single-site, single-ethnicity (Han Chinese) sample

**Required Actions:**
1. Explicitly acknowledge sample size as major limitation
2. Recalibrate confidence in conclusions appropriately
3. Frame as foundation for larger validation study
4. Discuss planned or potential replication efforts

---

### 3. **Overfitting Risk: High Accuracy Without Validation**

The reported prediction metrics are exceptionally high:
- AUC values: 0.82-0.86 for top features
- R² values: 0.144-0.479 (upper range is high)
- Odds ratios: Range from <0.001 to >70,000 (extreme values)

**Context from meta-analysis:**
Machine learning in emotional disorder treatment prediction (Clinical Psych Rev, 2025) shows typical AUC = 0.65-0.75 across studies. The submitted paper's AUC 0.82-0.86 exceeds this substantially.

**Possible interpretations:**
1. Genuinely superior biomarker (hopeful but requires validation)
2. Overfitting to sample-specific noise (likely without cross-validation)
3. Sample characteristics (drug-naïve, tight inclusion criteria) reduce heterogeneity
4. Statistical inflation in small samples

**Concerning indicators:**
- 15 different FC features tested, all significant after FDR correction
- Multiple comparisons across anatomical connections increases chance findings
- No cross-validation or bootstrapping to assess stability
- No comparison of prediction accuracy: FC vs clinical variables vs combined

**Required Actions:**
1. Perform cross-validation to assess model stability
2. Test simpler models (e.g., single best predictor vs all 15)
3. Compare FC prediction to clinical-only prediction (baseline Y-BOCS, duration, age, etc.)
4. Report calibration metrics (not just discrimination/AUC)
5. Provide model stability analysis (e.g., bootstrap 95% CIs for predictions)

Without these analyses, the high accuracy could reflect overfitting rather than true predictive power.

---

### 4. **Cross-Sectional Design Limits Mechanistic Insight**

The study collected baseline imaging only, without post-treatment scans. This is a missed opportunity for several reasons:

**Lost insights:**
1. **Causality:** Cannot determine if FC changes mediate treatment response
2. **Mechanism:** Cannot test if SSRI normalizes hyperconnectivity in responders
3. **Specificity:** Cannot compare neural changes between responders and non-responders
4. **Dose-response:** Cannot correlate FC changes with symptom improvement magnitude

**Contemporary standards:**
Pre-post designs are increasingly standard in treatment prediction studies, as they provide both prediction (baseline) and mechanism (change). Several recent NPP papers emphasize longitudinal designs.

**Justification provided:**
Authors note post-treatment imaging was not collected (data availability statement mentions ongoing longitudinal study but data not yet published). This is acknowledged as limitation but not adequately discussed.

**Required Actions:**
1. Expand discussion of why post-treatment imaging would strengthen conclusions
2. Acknowledge inability to test mechanistic hypotheses (e.g., "does SSRI normalize hyperconnectivity?")
3. Mention if follow-up imaging is planned or ongoing
4. Discuss how future longitudinal work could test causal models

This is a minor concern relative to validation issues, but limits the mechanistic depth of the contribution.

---

### 5. **Limited Treatment Generalizability**

All patients received **sertraline** at standardized doses up to 200mg/day. This is appropriate for internal validity but limits generalizability:

**Scope limitations:**
- Does prediction hold for other SSRIs? (fluoxetine, fluvoxamine, escitalopram)
- Does prediction hold for other treatments? (CBT, clomipramine, augmentation strategies)
- Would non-responders to sertraline respond to alternative SSRIs or mechanisms?

**Clinical utility impact:**
If the biomarker only predicts sertraline response, clinical utility is limited since:
- Clinicians often try multiple SSRIs sequentially
- Patient tolerability varies across SSRIs
- Some may prefer/require non-SSRI treatments

**Required Actions:**
1. Acknowledge specificity to sertraline (not "SSRIs" generally)
2. Discuss whether sensorimotor circuit hypothesis would extend to other serotonergic agents
3. Mention whether separate prediction for CBT response would be valuable (different neural mechanisms)
4. Clarify clinical decision pathway (when would FC guide SSRI selection vs alternative treatment?)

Not a fatal flaw, but overgeneralizing from sertraline to "SSRI response" is not fully supported.

---

## Minor Issues and Suggestions for Improvement

### 6. Incomplete Baseline Clinical Correlations

The paper reports that baseline Y-BOCS scores were not significantly correlated with altered FC in most regions (except SomMotA_1_l-VI_l in non-responders). This is mentioned briefly but deserves more attention.

**Why this matters:**
- Demonstrates FC predicts **treatment response** not just **symptom severity**
- Rules out confound that high FC → high severity → poor response
- Suggests circuit-specific mechanism rather than general disease load

**Suggestion:**
Emphasize this as a strength. Create a supplementary table showing:
- Correlations between each significant FC and baseline Y-BOCS (across all patients)
- Correlations within responders and non-responders separately
- Comparison: FC vs baseline severity as predictors

This would strengthen the argument that FC provides information beyond clinical variables.

---

### 7. Robustness of Responder Definition

The study defines response as ≥35% Y-BOCS reduction, which is standard but somewhat arbitrary. The field has used various cutoffs (25%, 35%, 50%) and alternative definitions (remission, clinical improvement scores).

**Sensitivity concern:**
- Would results hold with 25% cutoff (less stringent)?
- Would results hold with 50% cutoff (more stringent)?
- How many patients are near the 35% boundary (misclassification risk)?

**Suggestion:**
Perform sensitivity analysis:
1. Rerun key analyses with 25% and 50% cutoffs
2. Report how many patients fall in 30-40% reduction range (near boundary)
3. Test continuous outcome (percent reduction) vs binary classification

If results are robust to cutoff choice, this strengthens conclusions. If sensitive to cutoff, acknowledging this is important for interpretation.

---

### 8. Cerebellar Findings Underdeveloped

The cerebellar connectivity findings are interesting but incompletely integrated:

**Findings:**
- Responders: Increased SomMotA_1_l - VIIb_l connectivity
- Non-responders: Increased Thal_r - Crus1_r, Decreased SomMotA regions - Lobule VI

**Discussion gaps:**
- Different cerebellar regions have distinct functional profiles (motor vs cognitive vs affective)
- Lobule VI, VIIb: More sensorimotor
- Crus I: More cognitive (part of default mode network territory)
- What do these differential patterns suggest about treatment response mechanisms?

**Suggestions:**
1. Expand discussion of cerebellar heterogeneity
2. Relate findings to recent cerebellar psychiatry literature (Van Overwalle 2024, De Zeeuw et al. 2021)
3. Propose specific mechanistic hypotheses (e.g., "cerebellar forward modeling of sensorimotor predictions?")
4. Mention future directions (cerebellar stimulation as treatment target?)

This could elevate the paper from "we found cerebellar effects" to "cerebellar-sensorimotor integration may be key to SSRI response."

---

### 9. Statistical Reporting: Missing Details

Several statistical details would improve transparency:

**Missing elements:**
1. **Sample size justification:** Was N=54 determined a priori or convenience sample? Power analysis?
2. **Multiple comparisons strategy:** 28 ROIs tested, but only significant clusters reported. Total number of tests conducted?
3. **Model assumptions:** Logistic regression assumes linearity on logit scale. Were assumptions checked?
4. **Calibration:** AUC measures discrimination, but how well-calibrated are predictions? (Hosmer-Lemeshow test?)
5. **Multicollinearity:** Are the 15 FC predictors correlated? VIF values?

**Suggestions:**
Add to methods or supplement:
- Sample size determination and power calculation
- Complete multiple comparisons correction description
- Model diagnostic checks
- Calibration plots for prediction models
- Variance inflation factors for multivariate models

These technical details would satisfy quantitatively-oriented reviewers.

---

### 10. Data and Code Availability

The manuscript states individual-level data cannot be shared until cohort's primary results are published, with derived measures available upon request.

**NPP 2025 direction** (per editorial): Strong emphasis on open science and data sharing.

**Concerns:**
- Makes independent validation difficult
- Limits meta-analytic inclusion
- Reduces transparency

**Suggestions:**
1. Consider sharing summary statistics (group-level FC matrices)
2. Share analysis code (preprocessing and statistical pipelines)
3. Provide timeline for full data release
4. Deposit preprocessed FC matrices in repository (anonymized)

Even partial data sharing would strengthen the paper's impact and facilitate validation efforts by others.

---

## Specific Recommendations for Revision

### Essential Changes (Required for Acceptance):

1. **Recalibrate Claims Throughout**
   - Title: Change "biomarker" to "candidate biomarker"
   - Abstract: Add "pending external validation"
   - Conclusion: Emphasize exploratory nature, need for replication
   - Tone: From "we identified a biomarker" to "we present preliminary evidence"

2. **Add Internal Validation Analyses**
   - Leave-one-out or 10-fold cross-validation
   - Report cross-validated accuracy, sensitivity, specificity
   - Compare full-sample vs cross-validated metrics
   - Assess stability of effect size estimates

3. **Compare to Clinical Predictors**
   - Build logistic regression: baseline Y-BOCS, duration, age, sex as predictors
   - Build combined model: clinical + FC variables
   - Compare AUC: clinical-only vs FC-only vs combined
   - Demonstrate incremental validity of FC over clinical variables

4. **Expand Limitations Discussion**
   - Sample size below recommended standards (N<100)
   - Single-site, single-scanner, single-ethnicity
   - No external validation
   - Cross-sectional design
   - Single treatment (sertraline only)
   - Potential overfitting
   - Each limitation should have 2-3 sentences explaining implications

5. **Add Convergent Evidence Emphasis**
   - Prominently discuss Wang et al. (2025) parallel findings
   - Frame as cross-method validation (zfALFF + FC)
   - Discuss implications of sensorimotor convergence across studies
   - Position within ENIGMA consortium context

### Strongly Recommended (Would Substantially Strengthen):

6. **Robustness Analyses**
   - Test 25% and 50% responder cutoffs
   - Report continuous outcome analysis (correlation with % improvement)
   - Test whether FC predicts remission (Y-BOCS <16) not just response

7. **Model Comparison**
   - Test simpler models (single best predictor vs. full model)
   - Report if combined model improves prediction
   - Assess multicollinearity among FC predictors
   - Provide model selection justification

8. **Cerebellar Integration**
   - Expand discussion of cerebellar functional heterogeneity
   - Connect to recent cerebellar psychiatry literature
   - Propose mechanistic hypotheses for cerebellar involvement
   - Discuss implications for understanding OCD circuits

9. **Future Directions Section**
   - Outline specific validation study design
   - Mention collaboration opportunities or data sharing plans
   - Discuss how findings could inform treatment algorithms
   - Propose mechanistic follow-up studies (e.g., cerebellar stimulation)

10. **Statistical Transparency**
    - Add supplementary methods with full details
    - Provide model diagnostics
    - Report calibration metrics
    - Include variance inflation factors
    - Share analysis code in repository

### Optional Enhancements:

11. **Visual Improvements**
    - Add ROC curves for all significant predictors (currently only top 3 shown)
    - Create decision tree or nomogram for clinical utility visualization
    - Add supplementary figure: FC correlation matrix (among 15 predictors)
    - Create graphical abstract summarizing key finding

12. **Clinical Context**
    - Add paragraph on clinical implementation feasibility
    - Discuss cost-effectiveness considerations
    - Mention barriers to adoption (MRI accessibility, interpretation complexity)
    - Compare to other emerging biomarkers in OCD

---

## Decision Recommendation and Rationale

**Recommendation: MAJOR REVISIONS**

### Rationale for Decision:

**Reasons for Major Revisions (not rejection):**
1. Scientific merit is high (7.5/10)
2. Clinical question is important and timely
3. Methodology is rigorous and state-of-the-art
4. Theoretical framework is well-developed
5. Converging evidence from independent work (Wang et al. 2025)
6. Findings are novel (hyperconnectivity in responders)
7. With appropriate revisions, paper could make valuable contribution

**Reasons against Accept (why revisions required):**
1. Sample size below current NPP standards (N=54 vs N≥100)
2. No external validation despite biomarker claims
3. High prediction accuracy without validation raises overfitting concerns
4. Single-site, single-treatment limits generalizability
5. Claims not calibrated to evidence strength

**Why Not Reject:**
- Rejection would be appropriate if:
  - Methodology were flawed (it is not)
  - Contribution were negligible (it is not)
  - Overfitting concerns were unaddressable (they can be addressed with cross-validation)

- The core science is sound and addresses an important gap
- With recalibrated claims and additional analyses, the paper merits publication
- Exploratory biomarker discovery is valuable even without external validation, if framed appropriately

### Conditional Paths:

**Path 1: Accept after Major Revisions IF:**
- Authors recalibrate claims to match evidence strength
- Internal cross-validation demonstrates model stability
- Comparison to clinical predictors provided
- Limitations discussion substantially expanded
- Convergent evidence with Wang et al. emphasized
- Statistical transparency improved

**Path 2: Reject & Invite Resubmission IF:**
- Authors cannot/will not perform requested analyses
- Cross-validation reveals models do not generalize
- Authors insist on biomarker claims without validation
- Substantial conceptual concerns emerge

**Path 3: Accept with Minor Revisions IF (unlikely):**
- Authors obtain external validation dataset during revision
- Validation confirms prediction accuracy
- All other concerns addressed

---

## Comparison to NPP 2025 Standards

Based on systematic analysis of recent NPP publications (2025 special issue on neuroimaging), this paper:

**Meets Standards:**
- ✓ Methodological rigor (preprocessing, statistics)
- ✓ Clinical relevance and translational focus
- ✓ Theoretical integration across levels
- ✓ Transparent reporting
- ✓ Ethical conduct
- ✓ Novel contribution

**Below Standards:**
- ✗ Sample size (N=54 vs N≥100 recommendation)
- ✗ External validation (increasingly expected)
- ✗ Multi-site collaboration (single-site)
- ✗ Open data sharing (limited availability)

**Mixed:**
- ○ Statistical sophistication (rigorous but traditional; ML approaches more common but appropriate given N)
- ○ Effect size magnitude (high but not validated)
- ○ Generalizability (limited but acknowledged)

**Overall Fit:** The paper represents solid work that would have been strong 5 years ago but falls slightly short of 2025 standards for biomarker validation. However, it remains publishable with appropriate recalibration and additional analyses.

---

## Summary for Authors

This manuscript makes an important contribution to OCD treatment prediction research by identifying sensorimotor circuit connectivity as a candidate biomarker for SSRI response. The work is methodologically rigorous, theoretically well-motivated, and clinically relevant. The finding of hyperconnectivity in responders versus hypoconnectivity in non-responders is novel and aligns with emerging circuit-based approaches to psychiatric heterogeneity.

However, the paper's claims exceed the evidence provided. The absence of external validation, combined with a sample size below current standards and exceptionally high prediction accuracy, creates concern about generalizability and potential overfitting. These concerns are addressable through internal cross-validation, comparison to clinical predictors, and appropriate recalibration of claims.

**Key Message:** Frame this work as important **exploratory discovery** that provides strong foundation for larger validation studies, rather than **definitive biomarker validation**. With this framing and the requested additional analyses, the paper merits publication and will make a valuable contribution to the field.

**Recommended Action:** Major Revisions with emphasis on validation analyses, comparative analyses, and recalibrated claims. With these changes, the paper will be suitable for publication in Neuropsychopharmacology.

---

## Suggested Changes to Key Statements

### Title:
**Current:** "Sensorimotor circuit connectivity as a candidate biomarker for responsiveness to selective serotonin reuptake inhibitors in obsessive–compulsive disorder"

**Suggested:** "Baseline sensorimotor circuit connectivity predicts selective serotonin reuptake inhibitor response in obsessive–compulsive disorder: An exploratory study"

OR: "Sensorimotor circuit connectivity as a candidate biomarker for SSRI response in OCD: Findings requiring validation"

### Abstract Conclusion:
**Current:** "Taken together, baseline sensorimotor-circuit FC differentiates responders from non-responders and holds promise as a neuroimaging biomarker to guide individualized pharmacological treatment in OCD."

**Suggested:** "Taken together, baseline sensorimotor-circuit FC differentiates SSRI responders from non-responders in this exploratory study. These findings, pending external validation in independent cohorts, suggest that sensorimotor connectivity may serve as a candidate biomarker to guide individualized pharmacological treatment in OCD. Replication in larger, multi-site samples is essential to confirm clinical utility."

### Discussion Conclusion:
**Current:** Similar strong biomarker language

**Suggested:** Add paragraph:
"Several limitations must be considered when interpreting these findings. First, our sample size (N=54), while adequate for exploratory discovery, is below the N≥100 recommended for definitive biomarker validation in recent neuroimaging standards [cite Marek & Laumann, 2025]. Second, and most critically, we did not validate findings in an external independent cohort. The high prediction accuracy (AUC 0.82-0.86) is promising but requires replication to rule out overfitting. Third, our single-site design limits generalizability across scanners, populations, and clinical settings. Fourth, we assessed only sertraline response; whether findings extend to other SSRIs or treatment modalities requires investigation. Fifth, the cross-sectional design prevents testing causal mechanisms through post-treatment imaging. These limitations position our findings as hypothesis-generating rather than clinically actionable. Future research should prioritize multi-site validation studies with larger samples, post-treatment imaging to test mechanisms, and comparison to alternative treatments. We are currently pursuing such validation efforts and encourage independent replication by other groups."

---

**End of Review**

**Reviewer Signature:** [Anonymous Expert Reviewer]
**Date:** October 26, 2025
**Recommendation:** **MAJOR REVISIONS**
**Priority:** HIGH (important contribution pending appropriate revisions)
**Estimated Revision Time:** 2-3 months (depending on additional analyses)

