# Peer Review Simulation: INCITE NeuroX-Fusion 130B Grant Proposal
## Simulated Grant Review Panel Comments and Scoring

**Review Panel:** NIH/NSF Special Emphasis Panel on AI/ML for Neurodevelopmental Disorders
**Review Date:** 2025-11-30
**Proposal:** INCITE NeuroX-Fusion 130B Foundation Model for Developmental Disorders
**Funding Mechanism:** R01-equivalent (Large-scale, 7-year, $50M)

---

## PANEL COMPOSITION

- **Chair:** Dr. Elena Vasquez (Biostatistics, NIH study section veteran, 20+ years)
- **Reviewer 1:** Dr. Michael Zhang (AI/ML, Foundation Models, Google Brain alum)
- **Reviewer 2:** Dr. Sarah Johnson (Child Psychiatry, ADOS-2 expert, 30+ years clinical trials)
- **Reviewer 3:** Dr. Rajiv Patel (Neuroimaging, ENIGMA consortium lead, 40-site studies)
- **Reviewer 4:** Dr. Lisa Chen (Regulatory Science, former FDA reviewer, 10+ SaMD approvals)

---

## REVIEWER 1: DR. MICHAEL ZHANG (AI/ML Expert)

### Overall Impact Score: **5** (1=Exceptional, 9=Poor) → **TOP 15%**

### Significance: **3** (High)

**Strengths:**
- This is genuinely paradigm-shifting. 130B foundation model for developmental disorders? Nobody else is doing this at scale.
- Multimodal integration (5 modalities) with federated learning across 50 sites - if they pull this off, it's Nature/Science cover material.
- LoRA fine-tuning strategy is elegant and cost-efficient (99% savings claim is realistic based on my own experience).
- Competitive benchmarking is excellent - clear positioning vs. SOTA (CCTF 82.1% → target 90-92%).

**Weaknesses:**
- **CRITICAL: Where is the INCITE NeuroX-Fusion 130B model?** I searched extensively - no ArXiv paper, no GitHub, no INCITE program announcement. Is this real or speculative?
  - If it exists: They need to cite it and show preliminary results ASAP.
  - If it doesn't exist: They're proposing to build a 130B foundation model from scratch, which is a $10-20M, 12-18 month side project. The proposal doesn't budget for this.
- **130B parameter scale is unjustified.** Medical imaging tasks typically saturate at 1-10B parameters (SAM-Med3D, MedSAM). Why 130B? This seems like parameter count inflation without ablation studies showing 130B > 13B.
- **Differential privacy (ε=1.0) performance impact is unknown.** In my experience, DP with ε=1.0 degrades accuracy by 2-5%. Their 90-92% target - is this with or without DP? They don't say.
- **Federated learning assumptions are optimistic.** ComBat harmonization cited (20-30% heterogeneity reduction), but ComBat requires centralized data calibration. Federated alternatives (FedBN) are less validated.

**Questions for PI:**
1. Do you have confirmed INCITE compute allocation? Please provide allocation letter.
2. What are the preliminary results of NeuroX-Fusion 130B on *any* task (even non-DD)?
3. Ablation study: What's the performance of 13B, 40B, 130B models on your pilot data?
4. Differential privacy sensitivity: What's accuracy at ε=0.1, 1.0, 10, ∞ (no DP)?

**Recommendation:** **MAJOR REVISION REQUIRED**
- Without INCITE documentation, I cannot assess technical feasibility.
- If they can produce preliminary results (even simulation on ABIDE), I'd raise this to score 2-3 (Excellent-Outstanding).

---

### Innovation: **2** (Outstanding)

**Strengths:**
- First DD-specific 130B foundation model (no competitor)
- First 50-site global federated learning for DD (SOTA is single-country, <10 sites)
- First end-to-end causal inference (genes→brain→behavior) - I've never seen this integration
- Parameter-efficient fine-tuning (LoRA) is state-of-the-art

**Weaknesses:**
- NeuroX-Fusion 130B architecture is under-specified. "SwiFT 4D + Channel-equivariant + BrainOmni" - how are these 3 models integrated? Ensemble? Modular? Hybrid?
- No technical innovation in the models themselves (SwiFT, BrainOmni are from literature). Innovation is in *application* to DD, not methods.

**Overall:** Despite weaknesses, this is the most innovative DD proposal I've reviewed in 5 years.

---

### Approach: **5** (Good)

**Strengths:**
- Statistical rigor is outstanding (>99% power for primary outcome).
- Bayesian adaptive design with interim analyses is appropriate for long study.
- 5-modality integration is comprehensive.
- pRCT for FDA approval is smart (directly targets clinical translation).

**Weaknesses (MAJOR):**
- **50-site coordination plan is woefully inadequate.** How will they recruit 50 sites across 5 continents? Who manages this? What's the budget per site?
  - I've coordinated 5-site studies - it's brutal. 50 sites requires a dedicated team of 10+ project managers.
- **Site dropout not modeled.** Typical multi-site studies lose 20-30% of sites. They should recruit 65 sites to ensure 50 complete.
- **Cluster-randomization effects ignored.** With 50 sites (clusters), effective sample size is n / [1 + (m-1)×ICC]. If ICC=0.10, n_eff = 3,000 / 6.9 = 435 (not 3,000!). This tanks their power calculations.
- **Missing modality handling is vague.** "Mask token + attention masking" - OK, but what's the performance degradation? If 40% lack genomics (likely), what's the impact?

**Questions:**
1. Have you coordinated even a 10-site study before? What's your track record?
2. What's the assumed site-level ICC? Recalculate power with clustering.
3. Pilot data? Even n=50-100 proof-of-concept would dramatically strengthen feasibility.

**Recommendation:** This is fundable with revisions, but needs much more operational detail.

---

### Investigators: **7** (Fair) **[MAJOR WEAKNESS]**

**Strengths:**
- Team size (10 people) seems appropriate.
- Multidisciplinary expertise implied.

**Weaknesses (DISQUALIFYING):**
- **NO INVESTIGATORS NAMED.** I cannot assess feasibility without knowing who leads this.
  - For a $50M, 7-year, 50-site study, I need to see: PI with 10+ multi-site studies, 20+ high-impact publications, prior $20M+ funding.
- **NO PRELIMINARY DATA.** Zero pilot studies, zero publications from this team on DD.
- **NO LETTERS OF SUPPORT.** No INCITE letter, no site commitment letters, no advisory board.
- **Expertise gaps not addressed.** Who has:
  - Foundation model training experience? (This is top 1% AI/ML talent)
  - 50-site international study management? (Rare skill)
  - FDA De Novo submission? (Requires regulatory specialist)

**This is the biggest red flag.** Without named investigators, this is a **non-starter**.

**Recommendation:** **CANNOT RECOMMEND FOR FUNDING** until investigators are named with CVs, track records, and preliminary data.

---

### Environment: **4** (Very Good)

**Strengths:**
- Aurora supercomputer access (if confirmed) is world-class.
- 50-site global network (if confirmed) is unprecedented.

**Weaknesses:**
- No institutional affiliation stated. Which university?
- No institutional support letter. Does the host institution commit resources?
- INCITE allocation not documented. Is it secured or aspirational?

**Recommendation:** Add institutional support letter, INCITE allocation letter.

---

### Overall Comments for Panel Discussion:

**Summary:** This is the most ambitious and innovative DD proposal I've seen, but it's also the most under-developed in terms of investigator credentials and feasibility details. The science is outstanding (if INCITE model is real), but I have zero confidence it can be executed without seeing:
1. Named PI with track record
2. INCITE allocation letter
3. Preliminary data (even n=50 pilot)

**Current State:** Top 15-20% (fundable with major revisions)

**With Fixes:** Top 3-5% (exceptional, near-certain funding)

**Recommendation to Applicants:** This is a diamond in the rough. Polish it with investigator credentials and pilot data, and it's a slam dunk. Without those, it's a high-risk speculation.

---

## REVIEWER 2: DR. SARAH JOHNSON (Child Psychiatry, Clinical Expert)

### Overall Impact Score: **6** (Good) → **TOP 20-25%**

### Significance: **2** (Outstanding)

**Strengths:**
- **The clinical need is enormous.** I run an autism diagnostic clinic - we have 18-month waitlists. 50% reduction (12→6 months) would be life-changing for families.
- Early diagnosis at 6-12 months (vs. current 24-48 months) could enable intervention during peak neuroplasticity (0-3 years). This is the holy grail of autism research.
- The impact metrics are compelling: 250-500億円 medical cost savings, 150-300億円 market creation.
- I love the pragmatic RCT design (n=500, 10 sites). This is real-world effectiveness, not ivory tower efficacy.

**Weaknesses:**
- **40-60 Nature/Science papers is absurd.** I've been in autism research for 30 years. Large consortia publish 5-10 high-impact papers over a decade, not 40-60. This inflated claim hurts credibility.
- **6-12 month diagnosis claim is misleading.** The proposal says wearables at 6-12 months, but the multimodal approach requires MRI (1-3 month wait) + WES (2-4 months turnaround). Total time is still 3-6 months (not same-day diagnosis).
- **Clinical workflow integration is under-addressed.** How do busy pediatricians use this? Who interprets the AI output? What training is required?

**Questions:**
1. What's the false positive rate of Tier 1 wearable screening? If 50% screen positive but only 20% truly have ASD, you're overwhelming specialists.
2. How do you handle false positives (family anxiety, unnecessary interventions)?
3. What's the clinician training program? ADOS-2 certification takes 40+ hours - what about AI interpretation?

**Recommendation:** This addresses a massive unmet need, but clinical implementation details need work.

---

### Innovation: **3** (High)

**Strengths:**
- Multi-tiered early diagnosis (wearables → imaging → clinical) is innovative.
- pRCT design for FDA approval is smart (Canvas Dx precedent shows pathway).

**Weaknesses:**
- Wearables for 6-12 month ASD prediction is not novel (prior studies exist, though small n).
- MRI/EEG/genomics are established tools (not innovative in themselves, innovation is in integration).

**Overall:** Clinically innovative (multi-tiered approach), methodologically appropriate (not cutting-edge).

---

### Approach: **6** (Good)

**Strengths:**
- Gold-standard ADOS-2 comparator is appropriate.
- Longitudinal design (5 timepoints over 5 years) enables trajectory analysis.
- Statistical power is excellent (>99%).

**Weaknesses (MAJOR):**
- **No discussion of developmental heterogeneity.** ASD at 6 months looks very different from ASD at 24 months. How do you handle:
  - Late-onset ASD (symptoms emerge 18-36 months, not detectable at 6 months)?
  - Developmental regression (typical development → regression at 18-24 months)?
  - Diagnostic stability (some kids diagnosed at 24 months no longer meet criteria at 5 years)?
- **Ethical issues not addressed:**
  - Labeling a 6-month-old as "high risk for ASD" - what's the psychological impact on parents?
  - False positives at 6 months → 18 months of anxiety → child develops typically. This is harmful.
  - Where's the genetic counseling plan? Informed consent for whole-exome sequencing in infants?
- **Incidental findings protocol missing.** WES will discover cancer-predisposition genes, other diseases. ACMG requires reporting - but proposal doesn't address this.

**Questions:**
1. What's your ethical framework for early diagnosis? Have you consulted ethicists, patient advocates?
2. Diagnostic stability: What % of 6-month predictions hold at 24 months? 5 years?
3. Incidental findings: How will you handle discovering BRCA1 mutation in an infant?

**Recommendation:** Add ethical framework, genetic counseling plan, diagnostic stability analysis.

---

### Investigators: **8** (Fair/Poor) **[CRITICAL WEAKNESS]**

**Weaknesses:**
- **NO INVESTIGATORS NAMED.** For a clinical trial of this magnitude, I need to see:
  - PI with ADOS-2/ADI-R gold-standard certification
  - Clinical trials experience (10+ RCTs, IND/IDE submissions)
  - Autism research track record (50+ autism papers)
- **NO PRELIMINARY DATA.** Have you even piloted the wearable screening with 10 infants?
- **NO CLINICAL ADVISORY BOARD.** Who represents patient perspectives (autistic adults, parent advocates)?

**This is a deal-breaker.** I cannot recommend funding a clinical trial with unnamed investigators.

**Recommendation:** **MAJOR REVISION** - Name investigators, show clinical trial experience, add patient advisory board.

---

### Environment: **5** (Good)

**Strengths:**
- 50-site global network (if real) provides diverse populations.
- pRCT across 10 sites (academic, community, rural) is pragmatic.

**Weaknesses:**
- No mention of clinical infrastructure. Do sites have:
  - ADOS-2 certified evaluators? (Requires specialized training)
  - Pediatric MRI scanners? (Requires sedation capabilities)
  - Genomics labs / partnerships?

**Recommendation:** Add site capability assessment, resource-sharing plan.

---

### Overall Comments:

**Summary:** This addresses a critical clinical need with a well-designed pRCT, but the proposal glosses over ethical complexities and clinical implementation barriers. The lack of named investigators is disqualifying.

**Current State:** Top 20-25% (good idea, poor execution)

**With Fixes:** Top 10% (named investigators + ethics + clinical details)

**Advice to Applicants:** Partner with autism advocacy groups (Autism Self-Advocacy Network, Autistic Women & Nonbinary Network) early. Get buy-in from autistic community, or this will face resistance even if scientifically sound.

---

## REVIEWER 3: DR. RAJIV PATEL (Neuroimaging, Multi-Site Expert)

### Overall Impact Score: **5** (Good/High) → **TOP 10-15%**

### Significance: **2** (Outstanding)

**Strengths:**
- 50-site, 5-continent neuroimaging consortium is unprecedented for DD. I've led ENIGMA (40+ sites, 50,000+ participants) - this is comparable scale.
- Inter-site diagnostic accuracy (90-92% target) is a huge leap from current SOTA (CCTF 82.1%). If achieved, this solves the generalization problem plaguing neuroimaging AI.
- Multimodal integration (sMRI + fMRI + EEG + genomics + digital) is comprehensive. Very few studies combine >2 modalities.

**Weaknesses:**
- **50-site coordination is vastly underestimated.** ENIGMA took 5+ years to build, $20M+ in funding, 20+ staff. This proposal allocates... how much? (Budget is vague.)
- **Scanner heterogeneity is the elephant in the room.** 50 sites = 50 different scanners (GE, Siemens, Philips), field strengths (1.5T, 3T), protocols. Even with ComBat harmonization, residual site effects are 10-20%.
- **I'm skeptical of 90-92% inter-site accuracy.** ABIDE studies show 5-10 point drops from intra-site to inter-site. Federated learning helps, but I doubt it eliminates the gap entirely.

**Questions:**
1. What's your multi-site study management experience? Have you coordinated even 5 sites?
2. How will you enforce protocol standardization? (ENIGMA requires rigorous QC - do you have a plan?)
3. What's the site payment model? (ENIGMA sites contribute data for free + co-authorship. Your model?)

**Recommendation:** This is fundable if they can show credible coordination plan and realistic site recruitment strategy.

---

### Innovation: **3** (High)

**Strengths:**
- Largest DD neuroimaging consortium to date (if achieved).
- Federated learning with 50 sites is innovative (current SOTA: 5-10 sites).
- Multi-continental diversity (5 continents) addresses generalizability.

**Weaknesses:**
- Neuroimaging methods are standard (FreeSurfer, fMRI connectivity). Innovation is in scale, not methods.

---

### Approach: **5** (Good)

**Strengths:**
- Leave-one-site-out cross-validation (50-fold) is rigorous.
- ComBat harmonization is appropriate (though federated adaptation needed).
- Power calculations account for multi-site heterogeneity (... wait, do they? See below).

**Weaknesses (MAJOR):**
- **Cluster-randomization power adjustment is MISSING.** With 50 sites (clusters), you must account for intra-cluster correlation (ICC). Typical ICC for multi-site neuroimaging: 0.05-0.15.
  - Design effect = 1 + (m-1)×ICC, where m = 60 (participants/site)
  - If ICC=0.10: Design effect = 1 + 59×0.10 = 6.9
  - Effective n = 3,000 / 6.9 = 435 (not 3,000!)
  - **This drops power from >99% to ~60% for medium effects.**
- **Site recruitment timeline is unrealistic.** They say "50 sites" but give no timeline. ENIGMA took 5 years to recruit 40 sites. Realistically: 2 years to recruit 50 sites (if well-connected PI).
- **Data quality monitoring plan is absent.** Who reviews MRI quality? FreeSurfer segmentation errors? (ENIGMA has rigorous QC - essential for multi-site.)

**Questions:**
1. What's the site-level ICC assumption? (They don't state it.)
2. Recalculate power with design effect. What's the result?
3. What's the data quality monitoring plan? (Automated QC, manual review, exclusion criteria?)

**Recommendation:** **MAJOR REVISION** - Add cluster-adjusted power, site recruitment timeline, QC plan.

---

### Investigators: **7** (Fair) **[MAJOR WEAKNESS]**

**Weaknesses:**
- **NO INVESTIGATORS NAMED.** For 50-site neuroimaging, I need to see:
  - PI with ENIGMA/ABIDE leadership experience
  - Track record: 20+ multi-site papers, 10+ consortia led
  - Neuroimaging expertise: FreeSurfer, fMRI preprocessing, quality control
- **NO PRELIMINARY DATA.** Have you piloted ComBat harmonization on even 2 sites?

**I cannot assess feasibility without knowing the PI.** If it's someone like Paul Thompson (ENIGMA founder), I'd give score 2 (Outstanding). If it's a junior investigator, score 9 (Poor).

**Recommendation:** Name investigators with multi-site track record.

---

### Environment: **4** (Very Good)

**Strengths:**
- Aurora supercomputer for 130B model training is excellent (if secured).
- 50-site network (if built) is world-class infrastructure.

**Weaknesses:**
- No list of committed sites. Who are the 50 sites? Even 5-10 letters of intent would strengthen this.
- No data coordinating center specified. Where will federated servers run? Who manages data quality?

**Recommendation:** Add letters of intent from 5-10 sites, specify data coordinating center.

---

### Overall Comments:

**Summary:** This is a hugely ambitious neuroimaging consortium with potential to transform DD diagnostics. However, the operational details (site coordination, QC, power calculations) are severely under-developed. With a proven multi-site PI and revised power analysis, this is top 5%. As written, it's top 15%.

**Current State:** Top 10-15% (great idea, weak execution plan)

**With Fixes:** Top 3-5% (named PI + cluster power + QC plan)

**Advice:** Talk to Paul Thompson (ENIGMA), Adriana Di Martino (ABIDE). Learn from their hard-won lessons on multi-site coordination. Then rewrite this proposal with their insights.

---

## REVIEWER 4: DR. LISA CHEN (Regulatory Science, Former FDA Reviewer)

### Overall Impact Score: **6** (Good) → **TOP 20%**

### Significance: **3** (High)

**Strengths:**
- FDA De Novo pathway is well-justified (Canvas Dx precedent).
- pRCT with real-world endpoints (time-to-diagnosis) is patient-centered and aligned with 21st Century Cures Act.
- 10-site validation (vs. Canvas Dx 1-site) provides stronger regulatory evidence.

**Weaknesses:**
- **Timeline to FDA clearance is optimistic.** Canvas Dx took ~5-7 years from development to clearance. This proposal is more complex (multimodal, AI/ML) → likely 8-10 years.
- **FDA AI/ML requirements are underspecified.** FDA guidance (2024) requires:
  - Algorithm Change Protocol (how will model be updated post-deployment?)
  - Real-World Performance Monitoring (continuous accuracy tracking)
  - Cybersecurity (SBOM, penetration testing)
  - None of these are addressed in detail.

**Questions:**
1. Have you had pre-submission meetings with FDA to confirm De Novo eligibility?
2. What's your regulatory strategy if FDA requires Class III (PMA) instead of Class II (De Novo)? (Multimodal genomics+imaging may be higher risk.)

**Recommendation:** Add FDA engagement plan (pre-submission meetings), address AI/ML-specific requirements.

---

### Innovation: **4** (Very Good)

**Strengths:**
- Multi-site pRCT for AI diagnostic is innovative (Canvas Dx was single-site).
- Global validation (50 sites, 5 continents) exceeds FDA diversity requirements.

**Weaknesses:**
- pRCT design is standard (pragmatic, but not methodologically novel).

---

### Approach: **6** (Good)

**Strengths:**
- pRCT design is appropriate for regulatory validation.
- Primary endpoint (time-to-diagnosis) is clinically meaningful and patient-centered.
- Secondary endpoints (accuracy, cost-effectiveness) support payer coverage.

**Weaknesses (REGULATORY RED FLAGS):**
- **Risk management (ISO 14971) is MISSING.** FDA requires comprehensive risk analysis:
  - Hazards: False positives, false negatives, model bias, data breaches
  - Mitigation: Safeguards, warnings, post-market surveillance
  - **Where is this in the proposal?**
- **Algorithmic fairness is not addressed.** 21st Century Cures Act requires performance metrics stratified by race, ethnicity, sex. The proposal mentions "10+ ancestries" but no fairness analysis plan.
- **Post-market surveillance plan is absent.** FDA will require real-world performance monitoring for AI/ML devices. How will you track accuracy post-deployment?
- **Cybersecurity plan is vague.** "사이버보안 팀(24/7 모니터링)" is mentioned but not detailed. FDA requires SBOM, vulnerability scanning, incident response plan.

**Questions:**
1. Have you developed an ISO 14971 risk management file?
2. What's your fairness analysis plan? (Equal error rates across demographic subgroups?)
3. Post-market surveillance: How will you detect performance drift, distribution shift?

**Recommendation:** **MAJOR REVISION** - Add risk management, fairness analysis, post-market surveillance, cybersecurity details.

---

### Investigators: **8** (Fair/Poor) **[CRITICAL WEAKNESS]**

**Weaknesses:**
- **NO REGULATORY EXPERTISE NAMED.** For FDA submission, I need to see:
  - Regulatory consultant with De Novo submission experience (5+ successful submissions)
  - Former FDA reviewer or industry regulatory affairs lead
  - Quality management system (QMS) expertise (ISO 13485)
- **NO PRELIMINARY REGULATORY ENGAGEMENT.** Have you had even one pre-submission meeting with FDA?

**Without regulatory expertise, FDA submission will likely fail or face major delays.**

**Recommendation:** Name regulatory expert, show track record, document FDA pre-submission meetings.

---

### Environment: **5** (Good)

**Strengths:**
- Multi-site pRCT infrastructure is appropriate.
- Global diversity exceeds FDA requirements.

**Weaknesses:**
- No Quality Management System (QMS) mentioned. FDA requires ISO 13485 certification for SaMD manufacturers.
- No mention of manufacturing/deployment infrastructure. Who builds the production software? Where is it hosted?

**Recommendation:** Add QMS plan, software deployment infrastructure.

---

### Overall Comments:

**Summary:** This proposal has strong clinical validation (pRCT, multi-site), but regulatory preparation is immature. FDA AI/ML requirements (algorithm change, post-market surveillance, fairness, cybersecurity) are not adequately addressed. With regulatory expert on team and detailed FDA plan, this is fundable.

**Current State:** Top 20% (good clinical science, weak regulatory planning)

**With Fixes:** Top 10% (add regulatory expert + FDA engagement + risk management)

**Advice:** Hire a regulatory consultant NOW (before grant submission). FDA pre-submission meetings are free and incredibly valuable - use them to de-risk your regulatory pathway.

---

## PANEL DISCUSSION SUMMARY

**Chair (Dr. Vasquez):** Let's discuss overall scoring. I'm seeing a range from 5-6 (Good to High), which puts this in the top 10-20% range. Everyone agrees this is innovative and significant, but we have major concerns about investigator credibility, feasibility, and regulatory planning. Let's go around the table.

**Dr. Zhang (AI/ML):** I'm at a score of 5. This is the most exciting AI proposal I've seen for DD, but I can't recommend funding without seeing the INCITE NeuroX-Fusion 130B model or pilot data. If they can show preliminary results - even n=50 - I'd move to score 2-3.

**Dr. Johnson (Clinical):** I'm at 6. The clinical need is huge, but ethical issues and clinical implementation are under-addressed. And where are the investigators? For a $50M clinical trial, I need to see a proven PI with RCT experience.

**Dr. Patel (Neuroimaging):** Also 5. As someone who's coordinated 40-site studies, I know how hard this is. Their 50-site plan is under-developed. But if they have the right PI (ENIGMA-level experience), this is doable. Without seeing the PI, I can't assess feasibility.

**Dr. Chen (Regulatory):** I'm at 6. FDA pathway is plausible (Canvas Dx precedent), but they've underestimated regulatory complexity. Risk management, fairness, post-market surveillance are missing. These are not optional - FDA will require them.

**Chair:** Consensus score is **5-6 (Good to High) → Top 10-20%**. We all agree: This could be top 5% with major revisions. The key fixes are:

1. **Name investigators with track records** (all reviewers flagged this as critical)
2. **Clarify INCITE model status** (real or speculative?)
3. **Add pilot/preliminary data** (n=50-100 proof-of-concept)
4. **Detailed 50-site coordination plan** (recruitment, retention, governance, budget)
5. **Regulatory planning** (risk management, fairness, FDA engagement)

**Panel Recommendation: MAJOR REVISION REQUIRED**

**Summary Statement to Applicants:**
"This is a highly innovative and significant proposal addressing a critical unmet need in developmental disorder diagnosis. The scientific approach is rigorous, and if successful, would be transformative for the field. However, the panel has major concerns about feasibility and investigator expertise that must be addressed before funding can be recommended. Specifically: (1) No investigators are named, making it impossible to assess whether the team has the necessary multi-site coordination, AI/ML, clinical trials, and regulatory expertise to execute this ambitious project. (2) The INCITE NeuroX-Fusion 130B foundation model is described in detail but not cited or validated - it is unclear if this model exists or must be built from scratch. (3) Preliminary data are absent - even a small pilot study (n=50-100) would greatly strengthen feasibility. (4) The 50-site international coordination plan is severely under-developed for a project of this complexity. (5) Regulatory planning for FDA approval is incomplete (missing risk management, fairness analysis, post-market surveillance). With these revisions, this proposal has the potential to be in the top 3-5% and receive exceptional scores. We strongly encourage resubmission with major revisions."

**Funding Decision:** **DEFER** (pending major revisions)

**Revised Score Projection (if fixes implemented):** **2-3 (Excellent to Outstanding) → Top 3-5%**

---

## OVERALL PANEL SCORES SUMMARY

| Reviewer | Role | Overall Impact | Innovation | Approach | Investigators | Environment |
|----------|------|---------------|-----------|----------|---------------|-------------|
| **Dr. Zhang** | AI/ML | 5 (Good/High) | 2 (Outstanding) | 5 (Good) | 7 (Fair) | 4 (Very Good) |
| **Dr. Johnson** | Clinical | 6 (Good) | 3 (High) | 6 (Good) | 8 (Fair/Poor) | 5 (Good) |
| **Dr. Patel** | Neuroimaging | 5 (Good/High) | 3 (High) | 5 (Good) | 7 (Fair) | 4 (Very Good) |
| **Dr. Chen** | Regulatory | 6 (Good) | 4 (Very Good) | 6 (Good) | 8 (Fair/Poor) | 5 (Good) |
| **Panel Consensus** | - | **5-6** | **2-3** | **5-6** | **7-8** | **4-5** |

**Consensus Percentile:** **Top 10-20%** (currently)

**Consensus Percentile (with major revisions):** **Top 3-5%** (projected)

---

## KEY PANEL QUESTIONS FOR PI (MUST ADDRESS IN REVISION)

### Category 1: Investigator Credibility (CRITICAL)

1. **Who is the Principal Investigator?** Provide name, affiliation, CV, publication list, h-index, prior funding.
2. **What is the PI's track record in multi-site studies?** Provide examples of 3-5 prior multi-site studies (n sites, n participants, outcomes).
3. **Who are the Co-Investigators?** Provide names, expertise areas, CVs for:
   - AI/ML expert (foundation models, federated learning)
   - Child psychiatrist (ADOS-2 certified, clinical trials)
   - Genetic epidemiologist (WES, rare variants, GWAS)
   - Neuroimaging expert (multi-site studies, QC, harmonization)
   - Regulatory expert (FDA submissions, De Novo experience)
4. **What preliminary data do you have?** Even n=50-100 pilot showing proof-of-concept would be transformative for this application.

### Category 2: INCITE Model Status (CRITICAL)

5. **Does the INCITE NeuroX-Fusion 130B model exist?** Provide citation, technical report, GitHub link, or ArXiv paper.
6. **If the model exists, what are preliminary results?** Show performance on any task (even general neuroscience, non-DD).
7. **Do you have confirmed INCITE compute allocation?** Provide allocation letter from DOE INCITE program or Aurora supercomputer.
8. **If the model doesn't exist, what's your plan to build it?** Add timeline (12-18 months?), budget ($10-20M?), fallback (use BrainLM if Aurora unavailable?).

### Category 3: Multi-Site Coordination (HIGH PRIORITY)

9. **What is your site recruitment strategy?** How will you identify, recruit, and onboard 50 sites across 5 continents?
10. **What is the site recruitment timeline?** (Realistic: 18-24 months to recruit 50 sites)
11. **What is the site retention plan?** Incentives (co-authorship, funding, site-specific analyses), support (central IRB, training), attrition assumptions (20-30%).
12. **What is the governance structure?** Steering committee, data coordinating center, publication policy.
13. **What is the budget per site?** (ENIGMA: sites contribute data for co-authorship. Your model: pay sites? How much?)
14. **Do you have letters of intent from 5-10 sites?** Even preliminary commitments would strengthen feasibility.

### Category 4: Statistical Power (HIGH PRIORITY)

15. **What is the site-level intra-cluster correlation (ICC)?** Estimate from ABIDE/ADHD-200 multi-site data.
16. **What is the design effect for clustered data?** Recalculate power with design effect = 1 + (m-1)×ICC.
17. **What is the effective sample size after cluster adjustment?** n_eff = n / design_effect
18. **Does power remain >80% for primary outcome after cluster adjustment?** If not, how will you address this (increase n, reduce sites)?

### Category 5: Regulatory Planning (HIGH PRIORITY)

19. **Have you had FDA pre-submission meetings?** If yes, what feedback did you receive? If no, when will you schedule them?
20. **What is your ISO 14971 risk management plan?** Hazard analysis, mitigation strategies, post-market surveillance.
21. **What is your algorithmic fairness plan?** Performance metrics stratified by race, ethnicity, sex, age (21st Century Cures Act requirement).
22. **What is your cybersecurity plan?** SBOM, vulnerability scanning, incident response (FDA AI/ML guidance requirement).
23. **What is your post-market surveillance plan?** How will you monitor real-world performance, detect distribution shift, update the model?

---

## CONCLUSION: PATH FORWARD FOR APPLICANTS

**Current Status: TOP 10-20% (FUNDABLE WITH MAJOR REVISIONS)**

**Action Items for Revision (Priority Order):**

**CRITICAL (Must Address for Any Funding Consideration):**
1. Name Principal Investigator + Co-Investigators with CVs and track records
2. Provide preliminary data (even n=50-100 pilot study)
3. Clarify INCITE NeuroX-Fusion 130B status (cite model or add pre-training aim)
4. Add letters of support (INCITE, sites, advisory board)

**HIGH PRIORITY (Needed for Top 5% Competitiveness):**
5. Add detailed 50-site coordination plan (recruitment, retention, governance, budget)
6. Recalculate statistical power with cluster adjustment (site-level ICC)
7. Add regulatory planning (risk management, fairness, FDA engagement)
8. Add ethical framework (early diagnosis, genetic counseling, incidental findings)

**MEDIUM PRIORITY (Polish for Top 3%):**
9. Add missing modality performance analysis (degradation with 4, 3, 2, 1 modality)
10. Add differential privacy sensitivity analysis (ε=0.1, 1.0, 10)
11. Add clinician usability study (interpretability validation)
12. Add QC plan for multi-site neuroimaging (protocol standardization, data quality monitoring)

**With all CRITICAL + HIGH priority fixes: Projected to reach Top 3-5% (EXCELLENT-OUTSTANDING scores, high funding probability >85%)**

**Panel is enthusiastic about the science and encourages strong resubmission. This could be a field-defining study with proper investigator team and feasibility planning.**
