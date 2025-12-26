# QuantERA 2025 PHY-QML: Quick Reference Card
## Red Team 4.0/10 → Blue Team 7.5-8.0/10 Recovery Plan

**Last Updated:** 2025-12-04
**Status:** Ready for 4-week improvement sprint

---

## THE SITUATION (1-MINUTE BRIEF)

| Metric | Current (Red Team) | Target (Blue Team) | Path |
|--------|-------------------|-------------------|------|
| **Score** | 4.0/10 (Bottom 60%) | 7.5-8.0/10 (Top 15-25%) | +3.5-4.0 points |
| **Probability** | 15% | 40-50% | 3× improvement |
| **Rank** | ~120-140th of 200 | ~40-60th of 200 | +60-100 ranks |
| **Expected Value** | €480K | €1.44M | +€960K |
| **Timeline** | N/A (don't submit) | 4-6 weeks sprint | Feasible |

**Red Team Verdict:** "Fundable vision, but fatal execution gaps"
**Blue Team Counter:** "All gaps are fixable in 4 weeks with existing infrastructure"

---

## THE 10 CRITIQUES: AT A GLANCE

| # | Critique | Impact | Fix Time | Our Response |
|---|----------|--------|----------|--------------|
| 1️⃣ | Zero preliminary data | ⚠️ -2.0 pts | 2 weeks | Generate 3 pilots (Multi-Chip, QFF, Q-SSM) |
| 2️⃣ | Phantom team (no CVs) | ⚠️ -2.0 pts | 5 days | Add h-index 35-42, 15-20 papers/PI |
| 3️⃣ | Overpromise (4 breakthroughs) | 🟨 -1.5 pts | 2 days | Reframe as 2 core + 2 apps |
| 4️⃣ | Budget handwaving | 🟨 -1.0 pts | 2 days | Line-item breakdown €3.2M |
| 5️⃣ | Hardware uncertainty | 🟩 -0.5 pts | 1 day | "IBM Network confirmed" |
| 6️⃣ | No risk mitigation | 🟩 -0.7 pts | 1 day | 7-risk matrix with fallbacks |
| 7️⃣ | Vague quantum advantage | 🟩 -0.6 pts | Embedded | Quantify: +3-4% over baselines |
| 8️⃣ | Market timing risk | 🟦 -0.5 pts | N/A | Accept, position as complementary |
| 9️⃣ | Consortium siloing | 🟦 -0.5 pts | 1 day | Add collaboration history |
| 🔟 | Fuzzy Logic forced | 🟦 -0.3 pts | 1 day | Ablation study justification |

**Total Fixable:** 7/10 issues (Top 7 = +7.3 pts potential)

---

## 4-WEEK SPRINT: BATTLE PLAN

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   WEEK 1-2  │   WEEK 3    │   WEEK 4    │  WEEK 5-6   │
│   PILOTS    │ TEAM CREDS  │ BUDGET/RISK │ INTEGRATION │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Multi-Chip  │ Compile CVs │ Line-item   │ Merge all   │
│ QFF Barren  │ Joint pubs  │ Risk matrix │ Internal    │
│ Q-SSM EEG   │ Letters     │ Scope frame │ External    │
│ → 3 figures │ → 8 pages   │ → Justified │ → Submit    │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Priority:   │ Priority:   │ Priority:   │ Priority:   │
│ CRITICAL    │ HIGH        │ MEDIUM      │ LOW         │
│ Impact:     │ Impact:     │ Impact:     │ Impact:     │
│ +2.0 pts    │ +1.5 pts    │ +1.7 pts    │ +0.5 pts    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**Resource Need:** 2-3 FTE × 4 weeks = 8-12 person-weeks
**Budget:** ~€5-10K (GPU rental for pilots, can use existing DGX)

---

## 3 PILOT STUDIES: SPECIFICATIONS

### Pilot 1: Multi-Chip Ensemble (SNU, 3 days)
```
Task: Prove 2-chip ensemble > single-chip
Dataset: MNIST (publicly available)
Method: VQC (4 qubits) × 2 chips + DD-RAPTOR fusion
Target: 93% (multi) vs. 87% (single) = +6%
Tool: Qiskit + existing ensemble architecture
Owner: Prof. Cha + 1 PhD student
```

### Pilot 2: QFF Barren Plateau (Naples, 3 days)
```
Task: Prove QFF bypasses gradient vanishing
Dataset: Random circuit (6 qubits, 10 layers, known BP)
Method: Local goodness objectives (layer-wise)
Target: QFF loss <0.1, SPSA/Adam stuck >0.3
Tool: PennyLane
Owner: Prof. Acampora + 1 PostDoc
```

### Pilot 3: Q-SSM Long Sequences (Yonsei, 4 days)
```
Task: Prove Q-SSM > Mamba on long EEG
Dataset: CHB-MIT seizure data (publicly available)
Method: Quantum SSM (6 qubits) vs. Mamba baseline
Target: Q-SSM 90% vs. Mamba 87% on L>1000
Tool: Qiskit + DD-RAPTOR EEG pipelines
Owner: Prof. Yoo + 1 PhD student
```

---

## SCORE PROJECTION: DETAILED BREAKDOWN

### Excellence (50% weight)
| Sub-criterion | Before | After | Gain | Why |
|---------------|--------|-------|------|-----|
| Breakthrough | 6/10 | 8/10 | +2.0 | Pilots prove feasibility |
| Novelty | 7/10 | 8/10 | +1.0 | Scope clarified (2 core foundational) |
| Methodology | 5/10 | 7/10 | +2.0 | Validated, detailed protocols |
| **Subtotal** | 3.13 | 3.88 | +0.75 | **24% improvement** |

### Impact (30% weight)
| Sub-criterion | Before | After | Gain | Why |
|---------------|--------|-------|------|-----|
| Expected Impact | 7/10 | 8/10 | +1.0 | Quantum advantage quantified |
| Dissemination | 8/10 | 8/10 | 0 | Already strong |
| **Subtotal** | 2.25 | 2.40 | +0.15 | **7% improvement** |

### Implementation (20% weight)
| Sub-criterion | Before | After | Gain | Why |
|---------------|--------|-------|------|-----|
| Work Plan | 5/10 | 7/10 | +2.0 | Risk mitigation, realistic timeline |
| Consortium | 6/10 | 7/10 | +1.0 | Team CVs, collaboration documented |
| **Subtotal** | 1.10 | 1.40 | +0.30 | **27% improvement** |

**TOTAL: 4.03 → 7.68 (+3.65 points, 91% improvement)**

---

## COMPETITIVE POSITIONING

### Us vs. "Hypothetical Winner"

```
                  WINNER           US (AFTER FIXES)
              ┌──────────────┐  ┌──────────────┐
Team h-index  │ 85 (Elite)   │  │ 35-42 (Est.) │
              └──────────────┘  └──────────────┘
                    ▲                   ▼
                 THEY WIN          WE LOSE
                    
Prelim. Data  │ Nature 2024  │  │ 3 Pilots     │
              └──────────────┘  └──────────────┘
                    ▲                   ▼
                 THEY WIN          COMPETITIVE
                    
Innovation    │ Incremental  │  │ Foundational │
              └──────────────┘  └──────────────┘
                    ▼                   ▲
                  WE WIN            WE WIN
                    
Scope         │ 1 domain     │  │ 3 domains    │
              └──────────────┘  └──────────────┘
                    ▼                   ▲
                  WE WIN            WE WIN
                    
Risk          │ Low (Safe)   │  │ Medium       │
              └──────────────┘  └──────────────┘
                    ▲                   ▼
              DEPENDS ON REVIEWER
```

**Outcome:** 40-50% win rate (depends on reviewer risk tolerance)

---

## DECISION TREE

```
                    ┌─────────────────────┐
                    │  Deadline Check     │
                    └──────┬──────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         >4 weeks?                 <4 weeks?
              │                         │
              ▼                         ▼
    ┌─────────────────┐      ┌──────────────────┐
    │ Can dedicate    │      │ Can generate 2   │
    │ 2-3 FTE for     │      │ pilots in 2 wks? │
    │ 4 weeks?        │      └────┬─────────────┘
    └────┬────────────┘           │
         │                   ┌─────┴─────┐
    ┌────┴────┐             │           │
   YES       NO            YES          NO
    │         │             │           │
    ▼         ▼             ▼           ▼
┌────────┐ ┌─────┐    ┌────────┐  ┌─────────┐
│ 4-WEEK │ │DEFER│    │ 2-WEEK │  │  DEFER  │
│ SPRINT │ │2026 │    │ MINIMUM│  │  2026   │
└───┬────┘ └─────┘    └───┬────┘  └─────────┘
    │                     │
    ▼                     ▼
 7.5-8.0/10           6.5-7.0/10
 40-50% prob          25-30% prob
 COMPETITIVE          RISKY
```

---

## 72-HOUR KICKOFF CHECKLIST

### Hour 0-24: Mobilization
- [ ] Leadership approves 4-week sprint (GO/NO-GO decision)
- [ ] Allocate 2-3 FTE (0.5-0.75 per partner)
- [ ] Kickoff meeting: Align on roles, timeline, priorities
- [ ] Access existing infrastructure: DD-RAPTOR, QML-RAPTOR, DGX

### Hour 24-48: Week 1 Launch
- [ ] **SNU:** Multi-Chip MNIST setup (dataset, quantum circuits, ensemble code)
- [ ] **Naples:** QFF Barren Plateau setup (benchmark circuit, PennyLane env)
- [ ] **Yonsei:** Q-SSM EEG setup (CHB-MIT data, baseline Mamba model)
- [ ] **Fraunhofer:** CV template created, PI info gathering started

### Hour 48-72: Execution Starts
- [ ] First pilot experiments running (Multi-Chip training launched)
- [ ] Daily stand-ups scheduled (15 min check-ins)
- [ ] Risk tracking: If pilots fail by Day 7, abort and defer to 2026
- [ ] Success metric: At least 1 pilot shows promising results by Day 7

---

## KEY CONTACTS & RESOURCES

### Internal Team
- **SNU (Prof. Cha):** Multi-Chip lead, overall coordinator
- **Yonsei (Prof. Yoo):** Q-SSM lead, evolutionary algorithms
- **Naples (Prof. Acampora):** QFF lead, fuzzy systems
- **Fraunhofer (Dr. Lorenz):** QUARK integration, budget coordination

### External Resources
- **IBM Quantum Network:** SNU institutional access (confirmed 2022+)
- **AWS Braket:** Pay-as-you-go QPU cloud (no partnership needed, €20-30K)
- **Qiskit Aer:** Classical simulation up to 20 qubits (free, open-source)
- **CHB-MIT Dataset:** Public EEG seizure data (PhysioNet, free)
- **DD-RAPTOR:** Existing multi-modal fusion infrastructure (reusable)
- **QML-RAPTOR:** 35+ paper knowledge base (domain expertise proof)

### Support Documents
1. **Red Team Analysis:** `/data/QuantERA/RED_TEAM_CRITICAL_ANALYSIS.md`
2. **Blue Team Defense:** `/data/QuantERA/BLUE_TEAM_DEFENSE_STRATEGY.md`
3. **Executive Summary:** `/data/QuantERA/EXECUTIVE_SUMMARY_RED_VS_BLUE.md`
4. **This Card:** `/data/QuantERA/QUICK_REFERENCE_CARD.md`

---

## SUCCESS DEFINITIONS

### Minimum Viable (4 weeks)
✅ 3 pilot studies with figures
✅ Team CVs compiled
✅ Budget breakdown created
✅ Risk matrix added
🎯 **Result:** 7.5-8.0/10, 40-50% probability

### Optimal (6 weeks)
✅ All minimum items PLUS:
✅ External expert review
✅ 4 letters of support
✅ Fully polished submission
🎯 **Result:** 8.0-8.5/10, 50-60% probability

### Stretch (12 months → 2026)
✅ Multi-Chip paper in Quantum
✅ QFF paper in QST
✅ IBM partnership MOU
🎯 **Result:** 9.0-9.5/10, 80-90% probability

---

## THE BOTTOM LINE (30-SECOND PITCH)

**Problem:** Red Team scored us 4.0/10 (Bottom 60%, 15% probability)

**Root Cause:** Presentation gaps (no pilot figures, no CVs, vague budget), NOT fundamental flaws

**Solution:** 4-week sprint leveraging existing AI Co-Scientist infrastructure
- Week 1-2: Generate 3 pilots proving concepts work (+2.0 pts)
- Week 3: Document team credentials h-index 35-42 (+1.5 pts)
- Week 4: Add budget/risk rigor (+1.7 pts)

**Outcome:** 7.5-8.0/10 (Top 15-25%, 40-50% probability)

**Investment:** 8-12 person-weeks, ~€5-10K compute

**ROI:** €960K expected value gain (3× probability improvement)

**Decision:** EXECUTE 4-WEEK SPRINT → SUBMIT → COMPETE

**Risk if we don't:** 0% probability, missed €3.2M opportunity

**Timeline:** Kickoff in 72 hours, submit in 4-6 weeks

---

## EMERGENCY CONTACTS

**Project Lead:** Professor Cha (SNU)
**Budget/Admin:** Project Manager (TBD)
**Technical Support:** AI Co-Scientist team
**External Review:** QuantERA 2024 winner (friendly contact, TBD)

**Red Team Analyst:** Claude (Anthropic AI)
**Blue Team Strategist:** Claude (Anthropic AI)
**Document Date:** 2025-12-04

---

**PRINT THIS CARD → PIN TO WALL → EXECUTE PLAN → WIN FUNDING**

