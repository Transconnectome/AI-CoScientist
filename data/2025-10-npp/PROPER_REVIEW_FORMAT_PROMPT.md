# Prompt: Convert to Proper Peer Review Format

## CRITICAL ERROR IN CURRENT VERSION

The current document is written as a **letter to authors** ("Dear Authors, please do this..."), but peer reviews should be written as **third-person evaluations** of the manuscript for the editor.

## CORRECT PEER REVIEW FORMAT

Peer reviews are written in **third-person objective voice**, evaluating the manuscript:

### WRONG (Current):
```
Dear Authors,

Thank you for submitting your manuscript. Please conduct cross-validation...
```

### CORRECT:
```
The manuscript reports prediction metrics (AUC 0.82-0.86) without any form of cross-validation. This is a critical weakness that must be addressed. The authors should conduct leave-one-out cross-validation...
```

## TRANSFORMATION RULES

### 1. Remove Letter Format
- **Delete**: "Dear Authors," salutation
- **Delete**: "Thank you for submitting..." pleasantries
- **Delete**: "Sincerely, Reviewer" closing
- **Keep**: Direct evaluation of manuscript

### 2. Convert to Third-Person Evaluation

**Before (Second Person):**
```
You need to conduct cross-validation and report metrics including AUC, sensitivity, and specificity.
```

**After (Third Person):**
```
The authors must conduct cross-validation. The revised manuscript should report cross-validated AUC, sensitivity, and specificity.
```

### 3. Use Evaluative Language

**Replace:**
- "Please conduct..." → "The authors should..." / "The manuscript requires..." / "This study must include..."
- "You reported..." → "The authors report..." / "The manuscript presents..."
- "Your study..." → "This study..." / "The current work..."
- "I recommend you..." → "I recommend the authors..." / "The manuscript should..."

### 4. Maintain Critical Tone Where Appropriate

**Strong Requirements:**
- "The authors **must** add cross-validation before this work can be accepted."
- "This is a **critical deficiency** that undermines the claims."
- "The manuscript **cannot** make biomarker claims without validation."
- "**Essential revision:** The authors should..."

**Moderate Suggestions:**
- "The authors **should strongly consider** adding decision curve analysis."
- "The manuscript would be **substantially strengthened** by..."
- "I **recommend** the authors conduct sensitivity analyses."

**Optional Enhancements:**
- "The authors **may wish to consider** sharing code."
- "**One potential enhancement** would be symptom dimension analysis."

### 5. Section-Specific Formatting

**Part A: Essential Revisions**
Format as firm requirements:
```
### A1. Internal Cross-Validation is Required

The manuscript reports exceptionally high prediction metrics (AUC 0.82-0.86, R²=0.48) without any form of cross-validation. These values likely represent optimistic upper bounds that will not generalize to new patients. Without validation, the reported performance metrics are not credible.

The authors must conduct leave-one-out cross-validation (LOOCV) for all logistic regression models. The revised manuscript should report cross-validated AUC with 95% confidence intervals, sensitivity, specificity, positive and negative predictive values, and balanced accuracy. Permutation testing (minimum 1,000 iterations) should establish the null distribution. Both in-sample and out-of-sample metrics should be reported for comparison.

These cross-validated values must replace the current metrics in the Abstract. A new Results subsection titled "Cross-Validation Analysis" should be added after the current prediction section. Figure 3 should be updated to show both original and cross-validated ROC curves. The Discussion must interpret the cross-validated performance and acknowledge any drop in accuracy.

Marek & Laumann (2025, Neuropsychopharmacology 50:52-57) demonstrate that small samples without validation show inflated effect sizes (2-3× larger than true effects). Cross-validation is non-negotiable for any predictive biomarker study.
```

**Part B: Recommended Revisions**
Format as strong recommendations:
```
### B1. Sensitivity Analysis Across Response Thresholds

The 35% Y-BOCS reduction threshold, while conventional, is somewhat arbitrary. The manuscript would be strengthened by testing robustness across multiple definitions of response.

I recommend the authors test three response definitions: liberal (≥25% reduction), conventional (≥35%, current), and stringent (≥50% reduction). For each threshold, the revised manuscript should report the number of responders vs. non-responders, AUC for sensorimotor FC prediction, and the optimal FC threshold. A comparison table across all three thresholds would help readers assess whether findings depend critically on the 35% cutoff.

If findings hold across multiple definitions, confidence in the results increases substantially. If highly threshold-dependent, clinical applicability may be narrower than claimed.
```

**Part C: Optional Enhancements**
Format as suggestions:
```
### C1. Code and Data Sharing

The authors may wish to consider depositing preprocessing scripts, analysis code, and processed FC matrices (anonymized) in a public repository such as OSF, GitHub, or the journal's data sharing platform. This would enhance reproducibility and align with open science principles increasingly valued in neuroimaging research.
```

### 6. Response Instructions Section

Keep the response instructions but frame as "what the authors should provide":

```
## REQUIRED ELEMENTS OF AUTHOR RESPONSE

The authors' response letter should address each point systematically:

**For Essential Revisions (A1-A6):**
The authors must provide point-by-point responses specifying exactly what was changed (page numbers, line numbers), key statistics/results, and justification for any points that cannot be addressed.

**For Recommended Revisions (B1-B6):**
The authors should indicate whether each analysis was conducted. If yes, key findings should be summarized. If no, constraints (time, data availability, feasibility) should be explained.

**For Optional Enhancements (C1-C5):**
The authors may briefly indicate interest and feasibility. Detailed response not required if not pursued.
```

## STRUCTURE TO MAINTAIN

- Keep all section headings (Part A, Part B, Part C)
- Keep all itemized revision points (A1-A6, B1-B6, C1-C5)
- Keep all technical requirements and specifications
- Keep all citations and references
- Keep summary sections and timeline

## CRITICAL: PRESERVE ALL CONTENT

- Every technical requirement must remain
- Every citation must remain
- Every specific instruction must remain
- Only the **voice and perspective** changes from "letter to authors" to "evaluation of manuscript"

## OUTPUT REQUIREMENTS

Rewrite the entire document as a proper peer review: third-person evaluation of the manuscript with firm requirements, strong recommendations, and optional suggestions. Remove all letter-to-authors formatting while preserving every technical detail.

---

## ORIGINAL DOCUMENT TO TRANSFORM

[The remarks document will be inserted here]
