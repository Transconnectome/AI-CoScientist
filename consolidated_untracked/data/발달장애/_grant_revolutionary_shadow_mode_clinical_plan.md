# Shadow-Mode Clinical Validation Outline — Pediatric Neurodevelopment AI

## Purpose
Demonstrate safety and efficacy of AI predictions before any intervention authority is granted, complying with MFDS and WHO AI-in-Health guidance (2025).

## Study Design (Phase 0/Shadow Mode)
- **Population:** 500 pediatric patients (0–6 yrs) spanning ASD, ADHD, cerebral palsy risk groups; stratified by SES and region.  
- **Duration:** 18 months observation; AI runs continuously but outputs are hidden from treating clinicians (logged for analysis).  
- **Endpoints:**  
  - Primary: Predictive accuracy for 12-month developmental outcomes (Vineland, Griffiths).  
  - Secondary: Calibration metrics, subgroup fairness, false-positive counseling protocol stress test.

## Workflow
1. **Data Ingestion:** Real-time multimodal feed (EHR, MRI/DTI, EEG, genomics, wearables).  
2. **AI Prediction Engine:** Runs probabilistic trajectory model + RL recommender but labels each output “For Research Only.”  
3. **Clinician Panel:** Independent experts evaluate AI suggestions retrospectively, rating clinical plausibility.  
4. **Safety Board:** Monthly review of discrepancies; triggers root-cause analysis for high-risk disagreements.

## Transition Criteria to Assisted Mode (Phase 1)
- Brier score improvement ≥20% vs. baseline.  
- Calibration error <2.5% across demographic slices.  
- Clinician plausibility rating ≥4/5 on 80% of cases.  
- Ethics board approval of communication protocol.

## Assisted Mode Overview (Phase 1)
- AI recommendations shown in dashboard; clinicians must accept/modify/ reject with rationale (logged).  
- RL engine still constrained; recommendations require dual sign-off (neurologist + ethicist).  
- Duration: 12 months; primary endpoint is change in time-to-diagnosis and appropriateness of interventions.

## Risk Mitigation
- **Human Override:** Immediate override channels; AI cannot auto-execute therapy changes.  
- **Psychological Safeguards:** Counseling team ready before any risk communication.  
- **Audit Trails:** Immutable logs for MFDS auditing; regular fairness and bias checks.

## Deliverables
1. Shadow-mode protocol (IRB submission by Jan 2026).  
2. Interim safety report (Month 6).  
3. Transition readiness dossier (Month 12).  
4. Assisted-mode launch plan with training curriculum for clinicians.



