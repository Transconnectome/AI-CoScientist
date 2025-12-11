# TACTICAL ACTION CHECKLIST
## QuantERA 2025 Proposal - 4-Week Emergency Fix Plan

**Target Score:** 67.5/100 (from current 45/100)
**Target Probability:** 35-40% (from current 15%)
**Deadline Assumption:** 4 weeks from today (adjust as needed)

---

## WEEK 1: FOUNDATION FIXES (Days 1-7)

### Day 1: Team Validation (BLOCKING)
**Priority:** 🔴 CRITICAL
**Effort:** 4 hours
**Owner:** Project Coordinator

**Tasks:**
- [ ] Email SNU/Yonsei PI: Request CV (2 pages max) with:
  - [ ] Full publication list (last 5 years)
  - [ ] H-index and total citations
  - [ ] Previous grant funding (amounts + funders)
  - [ ] QML-specific expertise (3-5 key papers)

- [ ] Email Naples Co-PI: Request CV with:
  - [ ] Fuzzy logic credentials
  - [ ] Evolutionary computation track record
  - [ ] Previous QuantERA or EU projects

- [ ] Email Fraunhofer lead: Request institutional profile with:
  - [ ] QUARK framework documentation
  - [ ] Industrial partnerships list
  - [ ] Safety certification expertise

**Deliverable:** 3 CVs (6 pages total) + institutional profiles
**Deadline:** End of Day 2 (chase if no response)

---

### Day 2-3: Theoretical Cleanup (CRITICAL)
**Priority:** 🔴 CRITICAL
**Effort:** 12 hours
**Owner:** Technical lead

**Tasks:**

**Part A: Remove Quantum Forward-Forward**
- [ ] Section 1.1 (Objectives): Remove "QFF local updates" → Replace with "Hybrid Quantum-Evolutionary Optimization (HQEO)"
- [ ] Section 1.2 (Novelty): Remove QFF claims → Focus on HQGA (cite: Malossini et al. 2008, Talbi et al. 2013)
- [ ] Section 1.3 (Methodology): Replace WP2 description:
  ```
  OLD: "Quantum Forward-Forward for local optimization"
  NEW: "Hybrid Quantum Genetic Algorithms (HQGA) with entangled crossover operators"
  ```
- [ ] Remove all mentions of "zero gradient cost" (unverifiable claim)

**Part B: Fix Fuzzy Quantum Logic**
- [ ] Find-replace: "Fuzzy Quantum Logic" → "Noise-Adaptive Quantum Architectures"
- [ ] Remove citation: "Khushal et al., 2025" (CANNOT BE VERIFIED)
- [ ] Replace with legitimate citations:
  - [ ] Add: Temme et al. (2017) "Error mitigation for short-depth quantum circuits"
  - [ ] Add: Kandala et al. (2019) "Error mitigation extends quantum computer utility"
  - [ ] Add: Cai et al. (2023) "Quantum error mitigation" (Rev. Mod. Phys.)

**Part C: Citation Audit**
- [ ] Verify EVERY citation exists (check arXiv, Google Scholar)
- [ ] Remove or replace any unverifiable references
- [ ] Add DOIs to all citations

**Deliverable:** Clean proposal draft with verified citations
**Deadline:** End of Day 3

---

### Day 4-7: Preliminary Experiment (HIGH PRIORITY)
**Priority:** 🟡 HIGH
**Effort:** 24 hours (distributed)
**Owner:** Quantum computing expert

**Option A: IBM Quantum Experiment (BEST)**

**Setup (Day 4):**
- [ ] Create IBM Quantum account (free tier: 10 min QPU/month)
- [ ] Install Qiskit: `pip install qiskit qiskit-ibm-runtime`
- [ ] Test connection to ibm_kyoto or ibmq_quito (5-qubit backends)

**Experiment (Day 5-6):**
- [ ] Implement multi-chip ensemble for MNIST classification:
  ```python
  # Single-chip baseline: 8 features on 1 QPU (4 qubits)
  # Multi-chip ensemble: 4 features on QPU-A + 4 features on QPU-B
  # Classical fusion: Vote or average
  ```
- [ ] Target metrics:
  - [ ] Single-chip accuracy: ~80-85%
  - [ ] Multi-chip accuracy: >85% (even +2% is evidence)
  - [ ] Statistical test: McNemar's test (p<0.05)

**Analysis (Day 7):**
- [ ] Generate confusion matrices
- [ ] Create comparison table
- [ ] Write 1-paragraph result summary for proposal

**Option B: Qiskit Aer Simulation (FALLBACK)**
- [ ] If IBM Quantum unavailable, use Aer simulator with noise models
- [ ] Less credible but better than nothing
- [ ] Label clearly as "simulation study"

**Deliverable:**
- [ ] Table: "Preliminary Multi-Chip Results (MNIST)"
- [ ] Figure: "Accuracy comparison (Single vs. Multi-chip)"
- [ ] 1 paragraph for Section 1.3 (Methodology)

**Deadline:** End of Week 1

---

## WEEK 2: EUROPEAN INTEGRATION (Days 8-14)

### Day 8-10: European Literature Review
**Priority:** 🟡 HIGH
**Effort:** 12 hours
**Owner:** Literature review lead

**Tasks:**
- [ ] Add 10 European QML papers to bibliography:

**Quantum Communication (2 papers):**
- [ ] QuTech (Netherlands): Pompili et al. (Nature 2021) "Entanglement distribution"
- [ ] TU Delft: Wehner et al. (Science 2018) "Quantum internet"

**Quantum Sensing (2 papers):**
- [ ] Fraunhofer (Germany): Recent quantum sensor publication
- [ ] VTT (Finland): Superconducting qubit work

**Quantum Computing (3 papers):**
- [ ] ORCA Computing (UK): Photonic quantum computing
- [ ] IQM (Finland): Quantum processors
- [ ] Pasqal (France): Neutral atom quantum computing

**QML Specific (3 papers):**
- [ ] European QML reviews or applications
- [ ] Search: "quantum machine learning" + "Europe" on arXiv (2024-2025)

**Integration:**
- [ ] Add to Section 1.2 (Novelty): "Building on European quantum excellence..."
- [ ] Cite in competitive positioning
- [ ] Show awareness of EU landscape

**Deliverable:** Updated bibliography + 2 paragraphs on EU quantum context
**Deadline:** End of Day 10

---

### Day 11-14: European Tone Reframing
**Priority:** 🟡 HIGH
**Effort:** 8 hours
**Owner:** Grant writing lead

**Tasks:**

**Section 1.1 (The Hook) - Rewrite:**
- [ ] OLD: "Paradigm shift from fighting physics to physics-aware"
- [ ] NEW: "Systematic approach to physics-informed QML design"
- [ ] Remove: "Revolutionary," "foundational redesign"
- [ ] Add: "Rigorous validation," "incremental milestones," "collaborative science"

**Section 1.2 (Novelty) - European Framing:**
- [ ] Add paragraph: "European Leadership in Quantum Technologies"
  - [ ] Mention European Quantum Flagship
  - [ ] Cite QuTech, VTT, Fraunhofer contributions
  - [ ] Position as "building on European strengths"

**Section 2.1 (Impact) - Align with QuantERA Priorities:**
- [ ] Emphasize: "Practical applications for industry"
- [ ] Add: "Technology transfer pathways"
- [ ] De-emphasize: "Foundational breakthroughs"
- [ ] Frame as: "Enabling NISQ-era commercial deployment"

**Section 3 (Implementation) - Collaborative Spirit:**
- [ ] Add: "Cross-border methodology exchanges"
- [ ] Specify: "Quarterly videoconferences + 2 in-person workshops"
- [ ] Show: "Joint PhD co-supervision (SNU-Naples, Naples-Fraunhofer)"

**Deliverable:** Reframed proposal with European collaborative tone
**Deadline:** End of Week 2

---

## WEEK 3: SCOPE & RISK (Days 15-21)

### Day 15-17: Scope Reduction
**Priority:** 🟡 HIGH
**Effort:** 10 hours
**Owner:** Technical lead + PM

**Tasks:**

**Reduce from 4 to 2 Breakthroughs:**

**KEEP:**
1. ✅ **Multi-Chip Ensembles** (validated concept, preliminary data from Week 1)
2. ✅ **Quantum State Space Models** (novel, feasible, good literature support)

**REMOVE:**
3. ❌ ~~Quantum Forward-Forward~~ (no theoretical foundation - needs 12-18 months pre-work)
4. ❌ ~~Fuzzy Quantum Diffusion~~ (buzzword confusion - reframe as part of Objective 1)

**Restructure Work Packages:**
- [ ] WP1: Multi-Chip Ensembles (Months 1-24, Korea+Italy lead)
- [ ] WP2: Q-SSM Temporal Models (Months 1-24, Korea lead)
- [ ] WP3: QUARK Certification & Robustness (Months 12-36, Germany lead)
- [ ] WP4: Grand Challenge Validation (Months 12-36, All partners)

**Update Objectives:**
- [ ] Objective 1: Multi-Chip Scalability (target: >15% improvement over single-chip)
- [ ] Objective 2: Q-SSM Linear Complexity (target: O(L) scaling validated on EEG data)
- [ ] Objective 3: QUARK Certification (target: Industry reliability standard published)

**Deliverable:** Restructured work plan (focused scope)
**Deadline:** End of Day 17

---

### Day 18-19: Risk Mitigation Section
**Priority:** 🟡 MEDIUM
**Effort:** 6 hours
**Owner:** Project manager

**Tasks:**

**Add Section 3.5: Risk Management**

**Risk Register Template:**
```markdown
| Risk ID | Description | Probability | Impact | Mitigation | Contingency |
|---------|-------------|-------------|--------|------------|-------------|
| R1 | Multi-chip comm overhead > benefit | Medium | High | Benchmark threshold (>15%) | Single-chip optimization |
| R2 | Q-SSM doesn't scale linearly | Medium | Medium | Ablation studies (L=100,500,1000) | Hybrid quantum-classical |
| R3 | NISQ noise too severe | Low | High | Error mitigation (ZNE, PEC) | Focus on logical qubits |
| R4 | Consortium coordination delays | Medium | Low | Monthly sync + shared GitLab | Clear milestone ownership |
| R5 | QPU access limitations | High | Medium | Multi-vendor strategy (IBM+Rigetti) | Simulation fallback |
```

**For each risk, specify:**
- [ ] Early warning indicators (metrics to track)
- [ ] Responsible partner
- [ ] Decision point (when to trigger contingency)

**Deliverable:** Risk management section (2 pages)
**Deadline:** End of Day 19

---

### Day 20-21: Budget Justification
**Priority:** 🟡 MEDIUM
**Effort:** 6 hours
**Owner:** Finance lead

**Tasks:**

**Add Section 3.6: Budget Breakdown**

**Template per Partner:**
```markdown
### Partner 1: SNU/Yonsei (Korea) - €400K

| Category | Item | Cost | Justification |
|----------|------|------|---------------|
| Personnel | 2 PhD students (36 months) | €180K | WP1 implementation + WP2 theory |
| Personnel | 1 Postdoc (24 months) | €100K | Q-SSM algorithm development |
| Equipment | QPU access (IBM Quantum) | €60K | 200 hours QPU time (@€300/hr) |
| Travel | 4 workshops + partner visits | €25K | EU consortium coordination |
| Consumables | HPC computing, cloud storage | €20K | Simulation + data storage |
| Publication | Open access fees (3 papers) | €15K | Dissemination |
| **Total** | | **€400K** | |
```

**Repeat for:**
- [ ] Partner 2: Naples (€350K)
- [ ] Partner 3: Fraunhofer (€400K)
- [ ] **Project Total:** ~€1.15M (check QuantERA limits per country)

**Verify against national annexes:**
- [ ] Korea (NRF): Check max funding
- [ ] Italy (MUR): Check eligibility rules
- [ ] Germany (BMBF/DFG): Check budget categories

**Deliverable:** Detailed budget table (3 pages)
**Deadline:** End of Day 21

---

## WEEK 4: POLISH & SUBMISSION (Days 22-28)

### Day 22-24: Final Rewrite
**Priority:** 🟡 MEDIUM
**Effort:** 12 hours
**Owner:** Grant writing lead

**Tasks:**

**Tone Polish:**
- [ ] Remove ALL instances of: "revolutionary," "paradigm shift," "foundational redesign"
- [ ] Replace with: "systematic," "rigorous," "validated," "incremental"
- [ ] Change passive voice to active: "We will demonstrate..." not "It is proposed..."

**Consortium Strengthening:**
- [ ] Add "Consortium Expertise Matrix" table:
  ```
  | Partner | QI Physics | ML/AI | Fuzzy Logic | Certification | Industry Links |
  |---------|------------|-------|-------------|---------------|----------------|
  | SNU/Yonsei | ✓✓✓ | ✓✓ | | | ✓ (Samsung, LG) |
  | Naples | | ✓✓ | ✓✓✓ | | ✓ (Italian AI firms) |
  | Fraunhofer | ✓ | ✓ | | ✓✓✓ | ✓✓✓ (BMW, Siemens) |
  ```

**Impact Quantification:**
- [ ] Add specific KPIs:
  - [ ] 6 peer-reviewed publications (2 per partner)
  - [ ] 2 PhD theses
  - [ ] 1 QUARK reliability standard (ISO submission)
  - [ ] 3 industry PoCs (Fraunhofer network)
  - [ ] 50% PhD students are women (diversity target)

**Deliverable:** Polished full proposal draft
**Deadline:** End of Day 24

---

### Day 25-26: Mock Review
**Priority:** 🟡 LOW (but valuable)
**Effort:** 8 hours
**Owner:** External reviewers (if available)

**Tasks:**

**Internal Mock Review:**
- [ ] Find 2-3 quantum computing colleagues (not on project)
- [ ] Send proposal with evaluation rubric
- [ ] Request: "Grade 0-5 on Excellence, Impact, Implementation"
- [ ] Specific questions:
  - [ ] "Would YOU fund this with your money?"
  - [ ] "What's the weakest section?"
  - [ ] "Is the team credible?"

**Incorporate Feedback:**
- [ ] Address top 3 criticisms
- [ ] Strengthen weakest section
- [ ] Add clarifications

**Deliverable:** Revised draft based on mock review
**Deadline:** End of Day 26

---

### Day 27-28: Final Checks & Submission
**Priority:** 🔴 CRITICAL
**Effort:** 6 hours
**Owner:** Project coordinator

**Tasks:**

**Citation Verification (CRITICAL):**
- [ ] For EVERY reference, check:
  - [ ] DOI resolves correctly
  - [ ] Authors match
  - [ ] Publication year is correct
- [ ] Remove any unverifiable citations
- [ ] Add missing DOIs

**Formatting:**
- [ ] Check page limits (QuantERA typically 15 pages max)
- [ ] Font: Arial 11pt or similar
- [ ] Margins: 2cm all sides
- [ ] Figures: High resolution, readable legends
- [ ] Tables: Clear headers, aligned columns

**Submission Checklist:**
- [ ] Part B: Research Proposal (main document)
- [ ] CVs: All PIs (2 pages each)
- [ ] Budget forms (per national annex requirements)
- [ ] Letters of Intent (if required by RFO)
- [ ] Ethics statement (if human data used)
- [ ] Data management plan

**Pre-Submission Check:**
- [ ] Spell check (UK English for EU submissions)
- [ ] Reference check (all citations valid)
- [ ] Budget totals match across documents
- [ ] All PI signatures obtained

**Submit via QuantERA portal:**
- [ ] Upload all documents
- [ ] Verify PDF rendering
- [ ] Submit before 17:00 CET deadline
- [ ] Save confirmation email

**Deliverable:** Submitted proposal
**Deadline:** December 5, 2025 17:00 CET (or actual deadline)

---

## QUALITY GATES (Go/No-Go Decisions)

### End of Week 1:
**Check:** Do we have team CVs + preliminary data?
- ✅ YES → Continue to Week 2
- ❌ NO → ESCALATE: Cannot submit without team validation

### End of Week 2:
**Check:** Is European framing complete + citations verified?
- ✅ YES → Continue to Week 3
- ❌ NO → Extend Week 2 by 2-3 days

### End of Week 3:
**Check:** Is scope reduced + risk section added?
- ✅ YES → Continue to Week 4
- ❌ NO → Consider scope reduction further OR extend timeline

### Day 26 (2 days before deadline):
**Check:** Mock review score ≥ 60/100?
- ✅ YES → Final polish and submit
- ❌ NO → DECISION: Submit anyway (35% chance) OR withdraw

---

## SUCCESS METRICS

**Minimum Viable Proposal (60/100):**
- ✅ Team CVs added (minimum credibility)
- ✅ Preliminary data OR simulation (some evidence)
- ✅ Citations verified (no ethics flags)
- ✅ Risk section added (basic due diligence)

**Target Proposal (67.5/100):**
- ✅ All above PLUS
- ✅ Real QPU experiment (stronger evidence)
- ✅ European literature integrated (better positioning)
- ✅ Scope reduced to 2 breakthroughs (realistic)
- ✅ Detailed budget justification (professional)

**Stretch Goal (75/100 - Funding Threshold):**
- ✅ All above PLUS
- ✅ Mock review by external expert (validation)
- ✅ Professional figure design (polished)
- ✅ Industry LoI from Fraunhofer partner (commitment signal)

---

## RISK INDICATORS (Stop-Work Triggers)

**RED FLAGS - Consider Withdrawal:**
- ⛔ Cannot obtain PI CVs by Day 3 → No team credibility
- ⛔ All citations fail verification → Ethics integrity issue
- ⛔ Mock review score <50/100 → Fundamental proposal failure
- ⛔ Consortium partner withdraws → Ineligibility

**YELLOW FLAGS - Adjust Scope:**
- ⚠️ IBM Quantum experiment fails → Use simulation (weaker)
- ⚠️ Cannot complete European literature → Focus on strengths
- ⚠️ Budget exceeds national limits → Reduce personnel or QPU access

---

## DAILY STANDUP TEMPLATE

**Each morning, answer:**
1. What did I complete yesterday? (checklist items ✅)
2. What am I doing today? (specific tasks)
3. What's blocking me? (escalate immediately)
4. Are we on track for target score? (re-estimate weekly)

**Every Friday, review:**
- Week progress: X/Y checklist items completed
- Revised score estimate: Current projection
- Next week priorities: Top 3 must-dos
- Risk status: Any new blockers?

---

## FINAL DECISION POINT (Day 26)

**After mock review, calculate:**
```
Estimated Score = (Excellence × 0.35) + (Impact × 0.25) + (Implementation × 0.40)

IF Score ≥ 65/100:
  → SUBMIT (competitive proposal, 35-40% funding probability)

IF Score = 55-64/100:
  → DECISION: Submit (long-shot, 20-30%) OR Withdraw & resubmit 2027

IF Score < 55/100:
  → WITHDRAW (better to preserve reputation, target QuantERA 2027)
```

---

## APPENDIX: QUICK REFERENCE

**Key Contacts:**
- SNU/Yonsei PI: [Email]
- Naples Co-PI: [Email]
- Fraunhofer Lead: [Email]
- QuantERA Helpdesk: Maurice.Tia@anr.fr (+33 1 72 73 06 90)

**Important Links:**
- QuantERA Portal: https://www.quantera.eu
- IBM Quantum: https://quantum.ibm.com
- Qiskit Docs: https://qiskit.org/documentation
- arXiv QML Papers: https://arxiv.org/list/quant-ph/recent

**File Locations:**
- Proposal Draft: `/data/QuantERA/Revised_Research_Proposal_v1.md`
- Processed Papers: `/data/QuantERA/processed_output/`
- Red Team Analysis: `/data/QuantERA/EVIDENCE_BASED_RED_TEAM_ANALYSIS_2025.md`

---

**Last Updated:** 2025-12-05
**Version:** 1.0
**Status:** READY FOR EXECUTION

**END OF TACTICAL ACTION CHECKLIST**
