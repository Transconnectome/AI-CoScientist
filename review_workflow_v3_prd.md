# World-Class CVPR Reviewer V3 (ECMARS Enhanced)

> **State-of-the-Art Automated Paper Review System** based on Neural-Symbolic Hybrid Architecture (Gemini 3 Pro + ECMARS Logic).

---

## 🚀 System Overview

**Reviewer V3** is an automated scientific peer review system designed to replicate the rigor, depth, and persona of a world-class CPVR/NeurIPS area chair (Dr. Jiook Cha). It combines **Multimodal Visual Analysis** with **Evolutionary Consensus (ECMARS)** decision logic to produce reviews that are:
1.  **Visually Grounded**: Critiques plots, figures, and architecture diagrams using Gemini 3 Pro Vision.
2.  **Contextually Aware**: Uses SOTA References to ground claims in existing literature.
3.  **Scientifically Rigorous**: Applies strict "Scientific Meaningfulness" classification (Paradigm-shifting vs. Pseudoscience).
4.  **Persona-Driven**: Emulates the specific expertise and tone of Dr. Jiook Cha (Neuro-AI Expert).

## 🧠 Key Innovations

### 1. Hybrid Architecture (Neural + Symbolic)
-   **Neural Engine**: `gemini-3-pro-preview` for high-reasoning text generation and visual critique.
-   **Symbolic Logic**: **ECMARS (Evolutionary Consensus Multi-Agent Review System)** logic ported for strict decision-making:
    -   **Meaningfulness Classification**: (Paradigm-Shifting, Substantial, Incremental, Pseudoscience).
    -   **Revision Potential Score**: (0.0 - 1.0) assessing if flaws are fixable.
    -   **3-Tier Decision**: (Accept / Major Revision / Reject).

### 2. Visual Forensics
-   Detects **"Visual Lies"** (e.g., truncated axes, misleading baselines).
-   Critiques **Architecture Diagrams** for clarity and reproducibility.
-   Evaluates **Data Visualization** (e.g., stopping 3D bar charts).

### 3. SOTA Reference Grounding
-   Automatically parses the bibliography.
-   Retrieves abstracts of key references.
-   Ingests them into ChromaDB (`session_{paper_id}_references`) to compare the paper's claims vs. actual SOTA.

---

## 🛠️ Usage Guide

### Requirements
-   Python 3.10+
-   `GOOGLE_API_KEY` (Gemini)
-   `OPENAI_API_KEY` (Optional, for secondary agents)

### Quick Start
To generate a review for a PDF paper:

```bash
# Run the Reviewer V3 Workflow
poetry run python scripts/review_workflow_v2.py \
    --paper "inputs/my_submission.pdf" \
    --out "reviews/v3_output"
```

### Output Format
The system generates a Markdown review (`Review_{paper_name}_{persona}.md`) containing:
1.  **ECMARS Dashboard**: High-level scores (Meaningfulness, Revision Potential, Decision).
2.  **Visual Critique**: Specific analysis of figures/tables.
3.  **Neuro-AI Perspective**: Domain-specific deep dive (if applicable).
4.  **SOTA Comparison**: Novelty assessment grounded in real references.
5.  **Weaknesses & Detailed Feedback**: Actionable steps for improvement.

---

## 🏗️ Technical Architecture

### Component Diagram

```
[Input PDF] 
    │
    ├──> [PDFExtractor] --> (Text) --> [ChromaDB: Session]
    │
    └──> [VisualAnalyzer] --> (Images) --> [Gemini 3 Pro Vision] 
                │                                  │
         (Visual Report) <-------------------------┘
                │
                v
      [ReviewOrchestrator] <--- [ChromaDB: Persona (Dr. Cha)]
                │
                + <-- [ECMARS Logic Module] (Rules & Rubrics)
                │
                v
      [Gemini 3 Pro LLM]
                │
                v
       [Final V3 Review]
```

### Key Files
-   `scripts/review_workflow_v2.py`: Main orchestrator (Manages the flow).
-   `scripts/review_utils/visual_analyzer.py`: Handles PDF->Image conversion and Vision API calls.
-   `scripts/review_utils/ecmars_utils.py`: Contains the ported ECMARS logic/prompts.
-   `scripts/ingest_golden_references_advanced.py`: Reference extraction and ingestion logic.

---

## 📊 ECMARS Categorization Rubric

| Category | Description | Revision Potential | Decision |
| :--- | :--- | :--- | :--- |
| **Paradigm-Shifting** | Field-defining work (Top 5%). | High | **Strong Accept** |
| **Substantial** | Significant SOTA advance (Top 20%). | High | **Accept** |
| **Incremental** | Minor gains, combination of existing methods. | Medium | **Major Revision** |
| **Pseudoscience** | Fatal flaws, data leakage, misleading visualizations. | Low (<0.3) | **Strong Reject** |

---

**Built by**: AI-CoScientist Team
**License**: MIT
