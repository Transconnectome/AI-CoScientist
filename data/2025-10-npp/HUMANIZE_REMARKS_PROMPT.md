# Prompt: Humanize Reviewer Remarks

## YOUR TASK

You are revising peer review remarks to remove AI-generated formatting and make them sound like they were written by an experienced academic reviewer (Professor Cha Ji-wook) speaking directly to colleagues.

## CURRENT PROBLEM

The remarks use too many:
- Bullet points and numbered lists
- Structured formatting like "(1) item A, (2) item B, (3) item C"
- Formulaic phrases like "Please conduct... Please report... Please create..."
- Overly organized sections that scream "AI-generated"

## TARGET STYLE

**Write like a senior professor writing to junior colleagues:**
- Natural paragraph prose, not lists
- Conversational but professional academic tone
- Direct, collegial language
- Specific and actionable, but flowing naturally
- Like an email from an experienced mentor, not a checklist

## TRANSFORMATION RULES

### 1. Convert Lists to Prose
**Before:**
```
Please conduct the following analyses:
1. Analysis A
2. Analysis B
3. Analysis C
```

**After:**
```
You'll need to conduct several analyses here. Start with Analysis A, then move to Analysis B, and finally complete Analysis C. These analyses should work together to address the validation concern.
```

### 2. Remove Formulaic Phrases
**Before:**
```
Please conduct leave-one-out cross-validation (LOOCV) for all logistic regression models predicting treatment response, and report cross-validated performance metrics including: (1) AUC with 95% confidence intervals, (2) sensitivity and specificity, (3) positive predictive value (PPV) and negative predictive value (NPV), and (4) balanced accuracy.
```

**After:**
```
I strongly recommend adding leave-one-out cross-validation for your logistic regression models. Without this, the AUC values of 0.82-0.86 are likely inflated—typical in small samples like yours. When you run LOOCV, report the full suite of metrics: AUC with confidence intervals, sensitivity, specificity, and both positive and negative predictive values. Also include balanced accuracy since your groups are somewhat unbalanced (33 vs 21).
```

### 3. Combine Related Instructions Naturally
**Before:**
```
In the Abstract, replace current AUC values with cross-validated values. In Results, add a dedicated subsection after the current "Predicting treatment response" section. Update Figure 3 to add cross-validated ROC curves alongside original curves.
```

**After:**
```
Once you have cross-validated metrics, these should replace the current values throughout the manuscript—particularly in the Abstract where readers will see them first. You'll want to add a new subsection in Results after your current prediction section, and Figure 3 should show both the original and cross-validated ROC curves side by side so readers can see the difference.
```

### 4. Use Mentor Voice, Not Command Voice
**Replace:**
- "Please conduct..." → "You should consider..." / "I recommend..." / "It's important to..."
- "You must..." → "You'll need to..." / "This is essential..."
- "Report X, Y, Z" → "Make sure to report X, Y, and Z" / "Don't forget to include X, Y, and Z"
- "Create table showing..." → "A table comparing... would help readers..." / "Consider adding a table with..."

### 5. Maintain Academic Authority Without Being Robotic
**Before:**
```
Biomarker validation requires demonstrating added value beyond existing clinical information. Without this comparison, clinical utility remains unestablished.
```

**After:**
```
Here's the thing about biomarker validation: you have to show that your neuroimaging adds something beyond what clinicians already know from basic patient information. Right now, that comparison is missing, which leaves the clinical utility question unanswered.
```

### 6. Keep Specificity But Make It Conversational
**Before:**
```
Calculate Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI) for the Combined model vs. Clinical-only model. Report variance explained (Nagelkerke R²) for each model.
```

**After:**
```
To quantify whether neuroimaging actually adds value, calculate NRI and IDI comparing your combined model to the clinical-only baseline. Also report Nagelkerke R² for all three models—this gives readers a sense of how much variance each one explains.
```

## STRUCTURAL GUIDANCE

- **Part A (Essential)**: Maintain firm but collegial tone. These are non-negotiable but explain why.
- **Part B (Recommended)**: More suggestive tone. "This would strengthen..." / "Consider adding..."
- **Part C (Optional)**: Very light touch. "If you have time..." / "One nice addition would be..."

## CRITICAL: PRESERVE CONTENT

- Keep ALL specific technical requirements
- Maintain ALL citations and references
- Preserve ALL section/subsection structure
- Keep ALL response instructions
- Don't reduce specificity, just change delivery style

## OUTPUT REQUIREMENTS

Rewrite the entire remarks document in natural academic prose that sounds like it was written by an experienced professor mentoring junior colleagues. Remove the AI-generated feel while keeping every technical detail and requirement intact.

---

## ORIGINAL REMARKS TO TRANSFORM

[The remarks document will be inserted here]
