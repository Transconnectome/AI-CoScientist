# Remarks to be Sent to the Author

## Manuscript MS 37480: Sensorimotor Circuit Connectivity as SSRI Response Predictor in OCD

Dear Authors,

Thank you for submitting your manuscript to *Neuropsychopharmacology*. Your work addresses an important clinical question using rigorous methodology, and I particularly appreciate the quality of your study design—the drug-naïve sample is quite valuable, and your preprocessing procedures are exemplary.

After careful review, I'm recommending **Major Revisions**. The fundamental science is sound, but the manuscript needs substantial revisions to align with contemporary expectations for predictive biomarker studies. The good news is that all essential revisions can be accomplished using your existing data.

I've organized my comments into three sections: essential revisions that must be addressed for acceptance, strongly recommended additions that would significantly strengthen the paper, and optional enhancements you might consider. Please address each point systematically in your response letter and revised manuscript.

---

## PART A: ESSENTIAL REVISIONS (Must Be Addressed for Acceptance)

### A1. Add Internal Cross-Validation Analysis

Here's my main concern: you're reporting very high prediction metrics—AUC values of 0.82-0.86 and R²=0.48—without any form of cross-validation. These values almost certainly represent optimistic upper bounds and may not generalize to new patients. This is especially problematic in small samples like yours.

I strongly recommend adding leave-one-out cross-validation for your logistic regression models. Without this, the AUC values are likely inflated—typical in small samples. When you run LOOCV, report the full suite of metrics: AUC with confidence intervals, sensitivity, specificity, and both positive and negative predictive values. Also include balanced accuracy since your groups are somewhat unbalanced (33 vs 21). You should also add permutation testing with at least 1,000 iterations to establish the null distribution for prediction accuracy. Report both your original in-sample metrics and the new cross-validated out-of-sample metrics so readers can see the comparison.

Once you have cross-validated metrics, these should replace the current values throughout the manuscript—particularly in the Abstract where readers will see them first. You'll want to add a new subsection in Results titled "Cross-Validation Analysis" after your current prediction section, and Figure 3 should show both the original and cross-validated ROC curves side by side so readers can see the difference. In the Discussion, interpret the cross-validated performance honestly and acknowledge any drop in accuracy.

The reason this matters: Marek & Laumann (2025, *Neuropsychopharmacology* 50:52-57) demonstrate that small samples without validation show inflated effect sizes—typically 2-3 times larger than true effects. Cross-validation provides an unbiased estimate of how well your model would actually perform on new patients.

In your response letter, describe the cross-validation procedure you implemented, report all cross-validated metrics, and discuss the implications if cross-validated AUC drops substantially from the original values.

---

### A2. Compare Functional Connectivity to Clinical Predictors

Here's the thing about biomarker validation: you have to show that your neuroimaging adds something beyond what clinicians already know from basic patient information. Right now, you're claiming that sensorimotor FC predicts treatment response, but you haven't provided any evidence that neuroimaging adds value beyond readily available clinical information like baseline severity, illness duration, or demographics.

You'll need to build and compare three logistic regression models. Model A should be clinical-only, including baseline Y-BOCS total score, illness duration, age, sex, education level, and BDI score. Model B should be FC-only with your sensorimotor connectivity values from the current analysis. Model C should be the combined model with both clinical variables and FC values. For each model, report AUC with 95% confidence intervals, then conduct DeLong tests comparing AUC values between all pairs of models. To quantify whether neuroimaging actually adds value, calculate Net Reclassification Improvement and Integrated Discrimination Improvement comparing your combined model to the clinical-only baseline. Also report Nagelkerke R² for all three models—this gives readers a sense of how much variance each one explains.

Create a comparison table showing all three models side-by-side and a figure showing ROC curves for all three on the same plot. Add a new Results subsection titled "Comparison to Clinical Predictors" before the Discussion. You'll also need a new table called "Comparative Performance of Clinical, Neuroimaging, and Combined Prediction Models" and a corresponding figure showing the ROC curves together.

In the Discussion, interpret the incremental predictive value honestly. Does FC add significant value beyond clinical predictors? If it doesn't, you need to discuss what that means for clinical translation. Without this comparison, the clinical utility of your biomarker remains completely unestablished.

In your response letter, report all comparative statistics, discuss whether FC adds significant incremental value, and if it doesn't add value beyond clinical predictors, address the implications for clinical translation.

---

### A3. Reframe Claims as Exploratory Discovery

I need you to substantially revise how you're framing this work. The manuscript currently makes definitive biomarker claims that simply aren't supported by your study design—you have 54 participants from a single site with no external validation.

Let's start with the Abstract. On line 15, change "biomarker" to "candidate biomarker." On line 18, add a qualifying phrase like "in an exploratory single-site study." On line 22, change "may serve as" to "represents a promising candidate for." And add to the final sentence something like "pending validation in independent samples."

In the Introduction, reframe your study purpose in the final paragraph as "exploratory discovery" rather than "validation." Add an explicit statement that this study aims to identify candidate neural predictors that require subsequent validation. In Results, change your section heading from "Predicting treatment response" to "Exploratory prediction of treatment response," and throughout the results section, hedge your mechanistic language appropriately.

In the Discussion, add a new paragraph after your opening paragraph that reads something like this:

> "The exploratory nature of this study must be emphasized. Our sample size (N=54) falls below contemporary recommendations for establishing reliable brain-behavior associations (N≥100; Marek & Laumann, 2025), and we lack external validation. These findings should be interpreted as hypothesis-generating and requiring replication rather than definitive evidence of a clinical biomarker. We recommend independent validation in larger, multi-site samples before any clinical implementation."

Throughout the Discussion, replace confident assertions with appropriately hedged language: "demonstrates" becomes "suggests"; "establishes" becomes "provides preliminary evidence for"; "biomarker" becomes "candidate biomarker"; "predicts" becomes "shows potential to predict." Your sample size, lack of external validation, and single-site design all necessitate framing this as exploratory work.

In your response letter, confirm that all claims have been appropriately hedged, describe the specific changes you made to reframe the work, and explain your understanding of the exploratory versus validation distinction.

---

### A4. Remove or Hedge Mechanistic Claims

Your cross-sectional design—baseline scans only, no post-treatment imaging—simply cannot support mechanistic claims about how SSRIs work. You need to identify and revise these claims throughout the manuscript.

In the Abstract, line 24, change "SSRIs may exert therapeutic effects through recalibration of sensory-motor coordination" to something like "Baseline sensorimotor connectivity differences may reflect preserved neural capacity that enables treatment response." In Discussion paragraph 3, remove "These findings suggest SSRIs modulate..." and replace it with "We hypothesize that baseline connectivity differences may reflect different illness subtypes or neural reserve capacity. Longitudinal imaging is needed to test whether FC changes mediate treatment response." Throughout the Discussion, add qualifiers to all mechanistic statements—phrases like "We speculate that..." or "One possible interpretation is..." Make sure you distinguish clearly between prediction (what you can actually conclude from your data) and mechanism (what requires further study).

Add a new paragraph to your Limitations section that reads:

> "Our cross-sectional design precludes causal or mechanistic inference. Baseline FC differences between future responders and non-responders could reflect: (1) different OCD subtypes with inherently different treatment trajectories, (2) variations in neural reserve or compensatory capacity, (3) pre-existing differences in illness severity or chronicity not captured by Y-BOCS scores, or (4) other unmeasured confounds. We cannot determine whether these FC differences causally influence treatment response or merely correlate with response. Longitudinal imaging with pre- and post-treatment scans is required to test whether FC changes mediate therapeutic effects and to distinguish prognostic biomarkers (predicting outcome) from mechanistic biomarkers (explaining how treatment works)."

The key point: cross-sectional baseline differences cannot establish mechanism. This is a prognostic marker study, not a mechanistic study. In your response letter, list all mechanistic claims you identified and how each was revised, and confirm you understand the distinction between prognostic and mechanistic biomarkers.

---

### A5. Expand Limitations Section

Your current limitations section doesn't adequately address the generalizability constraints and methodological limitations of this work. You need to add four new paragraphs.

First, address generalizability. Write something like: "This study was conducted at a single site (Shanghai Mental Health Center) using a single 3T Siemens scanner, which limits generalizability to other scanners and acquisition protocols. All participants were of Chinese ethnicity, and whether these findings extend to other populations (Caucasian, African, Hispanic, etc.) remains unknown. Cross-cultural differences in brain structure, connectivity patterns, and symptom presentation may moderate the predictive utility of sensorimotor FC. Multi-site replication across diverse populations is essential."

Second, address the single medication issue: "Only sertraline (50-200 mg/day) was tested, preventing generalization to other SSRIs (fluoxetine, fluvoxamine, paroxetine, escitalopram) or other pharmacological treatments (clomipramine, SNRIs, augmentation strategies). Different SSRIs have partially distinct pharmacological profiles beyond serotonin reuptake inhibition, and our findings may not extend to all serotonergic agents. The title and claims have been revised to specify sertraline rather than SSRIs generally."

Third, tackle sample size directly: "Our sample size (N=54 total, 33 responders vs. 21 non-responders) falls below contemporary recommendations for establishing reliable brain-behavior associations. Marek and Laumann (2025) recommend N≥100 as a minimum for validation studies, noting that smaller samples show inflated effect sizes and higher false positive rates. The unbalanced group sizes further reduce statistical power and precision for the smaller non-responder group. Our results should be interpreted as exploratory findings requiring replication in larger, adequately powered samples."

Fourth, address overfitting risk: "The originally reported AUC values (0.82-0.86) are exceptionally high for neuroimaging prediction, and cross-validation revealed [INSERT YOUR CROSS-VALIDATED VALUES]. The difference between in-sample and out-of-sample performance indicates some degree of overfitting, as expected in small samples. Even with cross-validation, our single-site sample size limits confidence in effect size estimates. External validation in independent samples is the only definitive way to establish true prediction accuracy."

Replace your existing brief limitations with these four comprehensive paragraphs in the Discussion section, Limitations subsection. Readers need a complete understanding of the generalizability boundaries and methodological constraints of your work.

In your response letter, confirm all four paragraphs have been added, fill in the cross-validated values in paragraph four, and describe any additional limitations you identified beyond those I've specified.

---

### A6. Revise Title and Report Residual Motion

Two separate issues here. First, your title claims "SSRIs" but you only tested sertraline. Please revise the title from "Baseline Sensorimotor Circuit Connectivity Predicts Responsiveness to Selective Serotonin Reuptake Inhibitors in Obsessive-Compulsive Disorder" to "Baseline Sensorimotor Circuit Connectivity Predicts Responsiveness to Sertraline in Obsessive-Compulsive Disorder: An Exploratory Study." Accuracy requires specifying the specific SSRI you tested, and adding "An Exploratory Study" appropriately frames the work.

Second, residual head motion can confound group differences and must be reported. Calculate for each participant: mean framewise displacement, mean DVARS, and percentage of volumes scrubbed. Create a new supplementary table titled "Supplementary Table X: Residual Motion Parameters by Group" with columns for Group (Responders/Non-responders), Mean FD, SD FD, Mean DVARS, SD DVARS, and percentage of scrubbed volumes. Include ANOVA results testing group differences. Add to your Methods section something like: "Mean framewise displacement did not differ between future responders (M=0.XX, SD=0.XX) and non-responders (M=0.XX, SD=0.XX), F(1,52)=X.XX, p=X.XX, ensuring that group differences in connectivity were not confounded by motion artifacts. Detailed motion parameters are provided in Supplementary Table X."

In your response letter, confirm the title has been revised, report the motion parameter statistics, and confirm there are no group differences in residual motion.

---

## PART B: STRONGLY RECOMMENDED REVISIONS (Should Address for Strong Paper)

### B1. Conduct Sensitivity Analyses with Alternative Response Thresholds

The 35% Y-BOCS reduction threshold you're using is conventional but somewhat arbitrary. Testing robustness across different thresholds would strengthen confidence in your findings.

I recommend testing three response definitions: liberal (≥25% Y-BOCS reduction), conventional (≥35% reduction, your current approach), and stringent (≥50% reduction). For each threshold, report the number of responders versus non-responders, the AUC for sensorimotor FC prediction, and the optimal FC threshold that maximizes sensitivity plus specificity. Create a table comparing results across all three thresholds and discuss the implications. Do your findings hold across thresholds or do they depend critically on the 35% cutoff?

Add a subsection in Results titled "Sensitivity to Response Threshold" and create a new supplementary table called "Prediction Performance Across Response Definitions." In the Discussion, interpret whether your findings are robust or threshold-dependent. If findings hold across multiple definitions of response, that increases confidence substantially. If they're highly threshold-dependent, the clinical applicability narrows considerably.

Please indicate in your response whether you conducted this analysis and summarize the key findings.

---

### B2. Add Decision Curve Analysis

AUC and accuracy metrics don't directly answer the clinical question that matters: "At what decision thresholds does FC-guided treatment selection improve outcomes compared to treating everyone or treating no one?"

Consider conducting decision curve analysis for your Clinical-only model, FC-only model, and Combined model. Calculate net benefit at threshold probabilities from 0% to 100%. Create a decision curve plot showing net benefit curves for all three models, plus "Treat all" and "Treat none" reference lines. Identify the threshold ranges where each model provides net benefit, and discuss the clinical interpretation—at what decision thresholds would FC-guided selection actually be beneficial?

Add a subsection in Results titled "Clinical Utility Analysis" and create a new figure called "Decision Curves for Clinical, Neuroimaging, and Combined Models." In the Discussion, interpret at which thresholds neuroimaging adds clinical value. Decision curve analysis quantifies clinical utility by showing net benefit at realistic decision points, which directly informs whether FC assessment would actually improve clinical decision-making in practice.

Please indicate in your response whether you conducted DCA and describe the key findings.

---

### B3. Provide Power Calculations for Future Validation Studies

Readers and future researchers need guidance on what sample sizes would be adequate for validation studies.

Based on your observed effect sizes—specifically your cross-validated AUC values—calculate the required sample size for 80% power to detect the effect at α=0.05, and also for 90% power. Assume both conservative (10% smaller effect) and optimistic (observed effect) scenarios. Report the calculations in your Discussion something like: "Based on our cross-validated AUC of X.XX, a future validation study would require N=XXX participants (80% power, α=0.05) to replicate this effect. Given potential effect size inflation in our exploratory sample, a more conservative sample size of N=XXX (assuming 10% effect reduction) is recommended."

Add this paragraph after your limitations section in the Discussion and describe implications for multi-site validation efforts. This guides resource allocation for validation work and demonstrates awareness of your current study's power limitations.

Please provide power calculations in your response or explain if this is infeasible.

---

### B4. Report Medication Adherence and Dosing Details

Treatment response depends on adequate medication exposure, and variability in adherence or final doses could confound your prediction analyses.

Report the distribution of final sertraline doses achieved—a histogram or frequency table showing dose ranges, mean, median, and range of final doses, plus the percentage of patients reaching each dose level (50mg, 100mg, 150mg, 200mg). If you have adherence data available, report your method of adherence assessment (self-report, pill counts, pharmacy records), the percentage of prescribed doses actually taken, and the number of patients with poor adherence (less than 80%). You should also analyze moderating effects by testing whether dose level or adherence moderates the FC-response relationship—add interaction terms to your regression models if there's sufficient variability.

If adherence data aren't available, acknowledge this as a limitation. In Methods, add details on your dose titration protocol and adherence monitoring. In Results, add a subsection titled "Treatment Exposure and Adherence" and create a new supplementary table called "Final Sertraline Doses and Adherence Rates."

This matters because inadequate medication exposure is a common cause of non-response. If dose or adherence varies substantially and moderates prediction, your FC measure may actually be a proxy for tolerability rather than a pure neurobiological predictor.

Please report dosing and adherence data if available in your response, or acknowledge this as a limitation if the data aren't available.

---

### B5. Analyze Comorbidity as Moderator

Comorbid anxiety and depression are common in OCD and may moderate both treatment response and neural signatures.

Report your comorbidity rates—the percentage with comorbid GAD, social anxiety, panic disorder, the range of BDI scores (you report the mean, but the distribution is important), and other relevant comorbidities that fell below your exclusion thresholds. Test whether comorbidity moderates the FC-response relationship by dichotomizing (any comorbid anxiety versus pure OCD) or using continuous measures (BDI score as a moderator). Add interaction terms like FC × comorbidity predicting response. If you find significant moderation, discuss the implications: Does FC predict response only in pure OCD or also with comorbidity? Should future biomarker studies stratify by comorbidity status?

In Methods, specify how comorbidity was assessed. In Results, add a subsection titled "Comorbidity as Moderator." In the Discussion, interpret heterogeneity in prediction across comorbidity status. OCD is heterogeneous, and comorbidity-specific biomarkers may have greater clinical utility than one-size-fits-all approaches.

Please report comorbidity analyses in your response or explain if the data are unavailable.

---

### B6. Justify Univariate Statistical Approach

Readers may question why you didn't use contemporary machine learning methods, given their prevalence in recent neuroimaging prediction studies.

Add a new paragraph to your Methods, Statistical Analysis section that reads something like:

> "We employed univariate logistic regression rather than multivariate machine learning approaches for three reasons. First, with N=54 participants and relatively few predictors (specific FC connections), complex machine learning models (support vector machines, random forests, deep learning) would face severe overfitting risk despite regularization. The curse of dimensionality dictates that high-dimensional models require substantially larger samples to achieve stable performance. Second, univariate models offer greater interpretability, providing clear effect sizes (odds ratios) and clinical interpretability of FC thresholds. Third, our goal was hypothesis-testing (does sensorimotor FC predict response?) rather than optimal prediction (maximizing accuracy at any interpretability cost). While multivariate approaches may achieve marginally higher accuracy in large samples, the simpler univariate approach is more appropriate for our exploratory sample size and scientific aims. Future studies with larger samples may benefit from comparing univariate and multivariate approaches."

Proactive justification prevents reviewer concerns and demonstrates thoughtful methodological choices. Please confirm in your response that this paragraph has been added and consider whether additional justification is needed.

---

## PART C: OPTIONAL ENHANCEMENTS (Would Strengthen Paper)

### C1. Share Code and Processed Data

Consider depositing your preprocessing scripts, analysis code, and processed FC matrices (anonymized, of course) in a public repository like OSF, GitHub, or the journal's data sharing platform. This enhances reproducibility, allows independent verification, and aligns with open science principles that are increasingly valued in neuroimaging. Please indicate your willingness to share materials and the timeline in your response.

---

### C2. Conduct Symptom Dimension Analysis

If you have the data, consider using the Y-BOCS symptom checklist to categorize patients by dominant symptom dimension (contamination, symmetry, hoarding, etc.) and test whether predictive patterns differ across dimensions. OCD heterogeneity is increasingly recognized, and dimension-specific biomarkers may have greater clinical utility than general approaches. Please indicate in your response if this is feasible with your data.

---

### C3. Add Cerebellar Parcellation Analysis

Given your unexpected cerebellar findings, consider conducting more detailed parcellation that distinguishes sensorimotor cerebellar regions (lobules I-VI) from cognitive regions (Crus I/II, VIIb). This would strengthen your cerebellar findings and provide more specific mechanistic hypotheses about which cerebellar functions relate to treatment response. Please indicate your interest and feasibility in your response.

---

### C4. Compare to Published Prediction Models

Consider creating a systematic comparison table showing study, sample size, method, predictors, and AUC or accuracy for all prior OCD treatment prediction studies. This would contextualize your contribution within the broader literature and help readers assess relative performance. Please indicate in your response if you'll add this to the Discussion.

---

### C5. Propose Multi-Site Validation Protocol

In your Discussion, consider outlining a specific plan for external validation—target sample size, recruitment sites, inclusion criteria, and a pre-registered analysis plan. This demonstrates commitment to validation and provides a clear roadmap for future work. Please indicate in your response whether you have plans or collaborations for validation studies.

---

## SUMMARY OF REVISION REQUIREMENTS

**Must Address (Essential):**
- A1: Add internal cross-validation analysis
- A2: Compare FC to clinical predictors
- A3: Reframe all claims as exploratory
- A4: Remove or hedge mechanistic claims
- A5: Expand limitations section
- A6: Revise title and report motion parameters

**Strongly Encouraged (Recommended):**
- B1: Sensitivity analysis across response thresholds
- B2: Decision curve analysis
- B3: Power calculations for future studies
- B4: Report medication adherence and dosing
- B5: Analyze comorbidity as moderator
- B6: Justify univariate statistical approach

**Optional (Enhancement):**
- C1-C5: Code sharing, symptom dimensions, cerebellar parcellation, literature comparison, validation protocol

---

## RESPONSE INSTRUCTIONS

In your response letter, please address each point systematically.

**For Essential Items (A1-A6):**
Provide a point-by-point response. Specify exactly what was changed with page numbers and line numbers. Provide key statistics or results where applicable. If you cannot address a point, explain why with justification.

**For Recommended Items (B1-B6):**
Indicate whether you conducted each analysis. If yes, summarize key findings. If no, explain the constraints—time, data availability, feasibility.

**For Optional Items (C1-C5):**
Briefly indicate interest and feasibility. No detailed response required if you don't pursue them.

**Response Letter Format:**
```
ESSENTIAL REVISIONS

A1. Internal Cross-Validation
We have conducted LOOCV as requested. Key changes:
- Cross-validated AUC: 0.XX (95% CI: 0.XX-0.XX), compared to original AUC of 0.86
- Updated Abstract lines 20-22 (page 2)
- Added new Results section "Cross-Validation Analysis" (page 12, lines 245-268)
- Updated Figure 3 to include cross-validated ROC curves
[Continue for all essential items...]

RECOMMENDED REVISIONS
[Address each recommended item...]

OPTIONAL ENHANCEMENTS
[Brief indication of interest...]
```

---

## TIMELINE

I'm requesting a revised manuscript within **6 weeks**. The essential analyses (A1-A6) are all feasible with your existing data and should require 2-3 weeks of analyst time. The reframing of claims will require careful review of the full manuscript text.

If you need an extension beyond 6 weeks, please contact the editorial office with justification.

---

## CLOSING REMARKS

Your study addresses an important clinical problem with rigorous methodology. The drug-naïve sample is particularly valuable, and your preprocessing approach is exemplary. The convergence with Wang et al. (2025, *J Affect Disord*) using a different method (zfALFF versus FC) substantially strengthens the biological plausibility of your sensorimotor circuit hypothesis.

The required revisions are substantial but achievable. All essential requirements can be addressed with your existing data. Upon satisfactory revision, your work will make a valuable contribution to the OCD treatment prediction literature and stimulate important validation efforts.

I look forward to receiving your revised manuscript.

Sincerely,

Reviewer