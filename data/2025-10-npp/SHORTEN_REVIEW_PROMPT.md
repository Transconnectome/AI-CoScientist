# Prompt: Shorten Review to Realistic Length

## PROBLEM

The current review is ~30,000 characters. Real peer reviews are 5,000-7,000 characters. This needs to be reduced to **1/5 of current length**.

## WHAT TO REMOVE

1. **Example paragraphs in quotes** - Reviewers don't write paragraphs for authors to copy
2. **Detailed formatting instructions** - "In Abstract line 15, change X to Y"
3. **Multiple bullet points** - One clear requirement, not 6 sub-bullets
4. **Repetitive explanations** - Say it once
5. **Where to report details** - Authors know where things go

## WHAT TO KEEP

1. **Core criticism** - What's wrong and why
2. **Essential requirement** - What must be done
3. **Brief rationale** - One sentence why
4. **Clear priority** - Essential vs. recommended vs. optional

## TARGET FORMAT

### Before (Too Long):
```
### A1. Add Internal Cross-Validation Analysis

The manuscript reports very high prediction metrics (AUC 0.82-0.86, R²=0.48) without any form of cross-validation. These values likely represent optimistic upper bounds and may not generalize to new patients.

The authors must conduct leave-one-out cross-validation (LOOCV) for all logistic regression models predicting treatment response, and report cross-validated performance metrics including: (1) AUC with 95% confidence intervals, (2) sensitivity and specificity, (3) positive predictive value (PPV) and negative predictive value (NPV), and (4) balanced accuracy. Add permutation testing (minimum 1,000 iterations) to establish the null distribution for prediction accuracy. Report both original (in-sample) and cross-validated (out-of-sample) metrics for comparison. Add a new Results subsection titled "Cross-Validation Analysis" and update all figures showing prediction performance to include cross-validated values.

In the Abstract, replace current AUC values with cross-validated values. In Results, add a dedicated subsection after the current "Predicting treatment response" section. Update Figure 3 to add cross-validated ROC curves alongside original curves. In Discussion, interpret cross-validated performance and acknowledge any drop in accuracy. Marek & Laumann (2025, Neuropsychopharmacology 50:52-57) demonstrate that small samples without validation show inflated effect sizes (2-3× larger than true effects). Cross-validation provides an unbiased estimate of generalization performance.
```

### After (Concise):
```
**A1. Cross-Validation Required**
The reported AUC values (0.82-0.86) without any validation are likely inflated. The authors must add leave-one-out cross-validation for all prediction models and report cross-validated performance metrics (AUC, sensitivity, specificity, PPV, NPV). Both in-sample and out-of-sample results should be reported. Marek & Laumann (2025) show that N<100 samples without validation inflate effect sizes 2-3×.
```

## RULES FOR EACH SECTION

**Part A (Essential) - 2-3 sentences per item:**
- Problem statement (1 sentence)
- Required action (1 sentence)
- Rationale (1 sentence)

**Part B (Recommended) - 1-2 sentences per item:**
- What would strengthen the paper
- Brief justification

**Part C (Optional) - 1 sentence per item:**
- Just state the option

## REMOVE COMPLETELY

- All quoted example paragraphs (">")
- All "In Abstract line X..." detailed instructions
- All "Create table titled..." formatting details
- All multi-level bullet points
- All "In your response letter..." instructions
- Response format template
- Timeline section
- Closing remarks section

## KEEP STRUCTURE

- Part A/B/C divisions
- Item numbers (A1-A6, B1-B6, C1-C5)
- Essential/Recommended/Optional labels
- Summary of requirements section

## TARGET OUTPUT: 6,000 characters total

- Part A (Essential): ~2,500 chars (6 items × ~400 chars)
- Part B (Recommended): ~2,000 chars (6 items × ~330 chars)
- Part C (Optional): ~500 chars (5 items × ~100 chars)
- Summary: ~1,000 chars

## TRANSFORM NOW

Cut the review to 1/5 length. Be ruthless. Real reviewers don't write novels.

---

## ORIGINAL REVIEW TO SHORTEN

[Review will be inserted here]
