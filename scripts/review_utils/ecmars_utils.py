"""
ECMARS (Evolutionary Consensus Multi-Agent Review System) Utilities.
Ported logic for Scientific Meaningfulness and 3-Tier Decision Making.
"""

def get_ecmars_system_prompt_addendum() -> str:
    """
    Returns the specific instructions for ECMARS-style rigorous evaluation.
    tobe appended to the main system prompt.
    """
    return """
    ### ECMARS EVALUATION FRAMEWORK (MANDATORY)
    
    You must evaluate the paper using the **ECMARS** rigorous criteria:

    #### 1. SCIENTIFIC MEANINGFULNESS CLASSIFICATION
    Classify the paper into ONE of these 4 categories. Be strict.
    - **Paradigm-Shifting**: (Rare, <5%) Field-defining work that fundamentally changes how we think.
    - **Substantial**: (Top 20%) Significant contribution that advances the state-of-the-art meaningfully.
    - **Incremental**: (Common) Sound work but with only minor or expected improvements.
    - **Pseudoscience/Flawed**: Fundamentally flawed methodology, invalid baselines, or misleading results.

    #### 2. REVISION POTENTIAL
    Assess whether the paper's weaknesses are "Addressable" or "Fatal".
    - **Revision Potential Score (0.0 - 1.0)**:
        - **High (0.7-1.0)**: Issues are clear and fixable (e.g., "Add experiment X", "Clarify section Y").
        - **Low (0.0-0.3)**: Issues are structural (e.g., "The core premise is wrong", "Data leakage detected").

    #### 3. 3-TIER DECISION SYSTEM
    You must output a decision based on this logic:
    - **ACCEPT**: Meaningfulness is 'Substantial' or higher AND no major flaws.
    - **MAJOR REVISION**: Paper has potential (High Revision Potential) but needs significant work. *Note: CVPR doesn't usually allow Major Revisions, but use this to indicate "Borderline/Weak Reject with hope".*
    - **REJECT**: Meaningfulness is 'Incremental'/'Pseudoscience' OR Low Revision Potential.

    ---
    **OUTPUT FORMAT REQUIREMENT**:
    Add a special section at the TOP of your review:
    
    # ECMARS Dashboard
    - **Meaningfulness**: [Category]
    - **Revision Potential**: [Score 0.0-1.0] ([Explanation])
    - **Decision**: [ACCEPT / MAJOR REVISION / REJECT]
    
    ---
    """

def get_meaningfulness_rubric() -> str:
    """Detailed rubric for meaningfulness if needed for chain-of-thought."""
    return """
    Rubric for Meaningfulness:
    1. Paradigm-Shifting: Introducing a new problem setting, a newly discovered phenomenon, or a method that makes all previous methods obsolete.
    2. Substantial: Achieving SOTA by a wide margin with a novel insight, or connecting two previously unrelated fields.
    3. Incremental: Combining existing components (A+B), hyperparameter tuning, or small gains on standard benchmarks.
    4. Pseudoscience: Cherry-picking results, hiding baselines, comparing against SOTA from 3 years ago, or theoretical errors.
    """
