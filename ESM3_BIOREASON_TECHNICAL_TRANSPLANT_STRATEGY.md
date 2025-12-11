# ESM3 & BioReason: Technical Transplantation Strategy for Korean Developmental Disorder Research

**Generated**: 2025-12-08
**Purpose**: Deep technical analysis of ESM3 and BioReason strategies with concrete implementation roadmap for Korean brain-genomics foundation model
**Target**: Samsung Future Technology Grant - Korean Developmental Disorder Foundation Model

---

## Executive Summary

This document provides **actionable technical strategies** extracted from ESM3 (protein language model) and BioReason (DNA-LLM integration) that can be directly transplanted to build a **Korean Brain-Genomics Foundation Model** for developmental disorder research.

**Key Insight**: Both ESM3 and BioReason demonstrate that **cross-modal reasoning** (protein ↔ language, DNA ↔ language) can be achieved through:
1. **Frozen foundation model** + **trainable LLM** architecture
2. **Contextualized embeddings** prepended to queries
3. **Multi-stage training** (supervised fine-tuning → reinforcement learning)
4. **Synthetic data augmentation** for scarce annotations

**Direct Application**: We will adapt these proven techniques to create **Brain ↔ Genetics ↔ Language** reasoning for Korean developmental disorder diagnosis and prognosis prediction.

---

## Part 1: ESM3 Technical Deep Dive

### 1.1 Architecture Components

#### **Multi-Track Transformer Design**
```
ESM3 Architecture:
┌─────────────────────────────────────────┐
│  Input Layer (3 Modalities)             │
│  ├─ Sequence Track (amino acids)        │
│  ├─ Structure Track (3D coordinates)    │
│  └─ Function Track (GO annotations)     │
├─────────────────────────────────────────┤
│  Tokenization Layer                     │
│  ├─ Sequence: standard AA tokens        │
│  ├─ Structure: VQ-VAE (1024 codebook)   │
│  └─ Function: discrete function tokens  │
├─────────────────────────────────────────┤
│  Embedding Fusion                       │
│  └─ Concatenate + Linear projection     │
├─────────────────────────────────────────┤
│  Transformer Blocks (48 layers)         │
│  ├─ Standard Self-Attention             │
│  └─ Geometric Attention (structure)     │
├─────────────────────────────────────────┤
│  Output Heads (3 Modalities)            │
│  ├─ Sequence reconstruction             │
│  ├─ Structure prediction (VQ tokens)    │
│  └─ Function classification             │
└─────────────────────────────────────────┘
```

**Transplantation to Brain-Genomics**:
```
Korean Neuro-Foundation Model:
┌─────────────────────────────────────────┐
│  Input Layer (3 Modalities)             │
│  ├─ Brain Track (fMRI/DTI timeseries)   │
│  ├─ Genomics Track (WGS variants)       │
│  └─ Behavior Track (clinical scores)    │
├─────────────────────────────────────────┤
│  Tokenization Layer                     │
│  ├─ Brain: SwiFT 4D patches (128 tokens)│
│  ├─ Genomics: k-mer tokenization (512)  │
│  └─ Behavior: discrete clinical tokens  │
├─────────────────────────────────────────┤
│  Embedding Fusion                       │
│  └─ Cross-modal attention alignment     │
├─────────────────────────────────────────┤
│  Transformer Blocks (24-48 layers)      │
│  ├─ Temporal Self-Attention (fMRI)      │
│  ├─ Genomic Context Attention           │
│  └─ Cross-Modal Fusion Attention        │
├─────────────────────────────────────────┤
│  Output Heads (3 Tasks)                 │
│  ├─ Diagnosis prediction (ASD/ADHD)     │
│  ├─ Prognosis trajectory (future brain) │
│  └─ Biomarker identification            │
└─────────────────────────────────────────┘
```

#### **Geometric Attention Mechanism**

**ESM3 Implementation**:
- Processes 3D protein structure as **distance matrices**
- Amino acids with spatial proximity get **higher attention scores**
- Stacked with sequence attention for dual context

**Brain-Genomics Adaptation**:
```python
class TemporalGeometricAttention(nn.Module):
    """
    Adapted from ESM3's Geometric Attention
    For 4D fMRI: (x, y, z, time)
    """
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.spatial_attention = SpatialAttention(d_model, n_heads)
        self.temporal_attention = TemporalAttention(d_model, n_heads)

    def forward(self, brain_voxels, timestamps):
        # Spatial proximity attention (like ESM3 protein structure)
        spatial_dist = compute_voxel_distance_matrix(brain_voxels)
        spatial_attn = self.spatial_attention(brain_voxels, spatial_dist)

        # Temporal attention for longitudinal trajectory
        temporal_attn = self.temporal_attention(spatial_attn, timestamps)

        return temporal_attn
```

**Implementation Timeline**: Week 3-4 (after SwiFT baseline established)

---

### 1.2 Training Strategies

#### **Masked Language Modeling (MLM) for Multimodal Data**

**ESM3 Approach**:
```
Training Objective:
For each protein:
1. Extract sequence, structure, function
2. Randomly mask 15-30% of tokens across ALL modalities
3. Predict masked tokens using cross-modal context

Example:
Input:  [SEQ: MET-ALA-<MASK>-GLY] [STRUCT: <MASK>-alpha-helix-beta] [FUNC: <MASK>]
Target: [SEQ: VAL] [STRUCT: loop] [FUNC: kinase_activity]
```

**Brain-Genomics Adaptation**:
```python
class NeuroGenomicMLM(nn.Module):
    """Multimodal Masked Language Modeling for Brain-Genomics"""

    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, brain_patches, genomic_kmers, behavior_scores):
        # Randomly mask tokens across all modalities
        brain_masked, brain_targets = self.mask_modality(brain_patches)
        genomic_masked, genomic_targets = self.mask_modality(genomic_kmers)
        behavior_masked, behavior_targets = self.mask_modality(behavior_scores)

        # Cross-modal prediction
        # Brain can predict genomics, genomics can predict brain
        brain_pred = self.brain_head(genomic_masked, behavior_masked)
        genomic_pred = self.genomic_head(brain_masked, behavior_masked)
        behavior_pred = self.behavior_head(brain_masked, genomic_masked)

        # Combined loss
        loss = (
            F.mse_loss(brain_pred, brain_targets) +
            F.cross_entropy(genomic_pred, genomic_targets) +
            F.bce_loss(behavior_pred, behavior_targets)
        )
        return loss
```

**Dataset Requirements**:
- 3,000+ Korean pediatric cohort (from proposal)
- 2,500+ DTI/fMRI scans
- Whole-genome sequencing (WGS) for subset (estimate 500-1000 samples)
- Clinical scores (ADOS, ADI-R, K-WPPSI)

**Training Schedule**:
- **Week 5-8**: Pre-training on public datasets (ABCD Study: 11,000 children, UK Biobank brain imaging)
- **Week 9-12**: Transfer learning on Korean cohort

---

#### **VQ-VAE for Continuous Structure Compression**

**ESM3 Strategy**:
- Protein 3D structures are **continuous** (atom coordinates)
- Use **Vector Quantized VAE** to compress into **discrete tokens** (1024 codebook)
- Two-stage training: (1) Train encoder+codebook with small decoder, (2) Freeze encoder, train large decoder

**Brain Imaging Adaptation**:
```python
class BrainStructureVQVAE(nn.Module):
    """
    Compress 4D fMRI (96×96×96×150 timepoints) → 128 discrete tokens
    Enables efficient transformer processing
    """
    def __init__(self, num_embeddings=2048, embedding_dim=256):
        super().__init__()
        # Stage 1: Efficient encoder
        self.encoder = CNN3D(in_channels=1, out_channels=embedding_dim)
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder_small = CNN3DTranspose(embedding_dim, out_channels=1)

        # Stage 2: Large decoder (trained after freezing encoder)
        self.decoder_large = TransformerDecoder(embedding_dim, num_layers=12)

    def encode(self, fmri_volume):
        z = self.encoder(fmri_volume)  # [B, 256, 12, 12, 12]
        # Quantize to nearest codebook vector
        distances = torch.cdist(z.flatten(2), self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices)
        return z_q, indices

    def decode(self, z_q, use_large_decoder=False):
        if use_large_decoder:
            return self.decoder_large(z_q)
        return self.decoder_small(z_q)
```

**Training Protocol**:
```bash
# Stage 1: Train encoder + codebook (Week 5-6)
python train_vqvae.py \
    --stage 1 \
    --dataset ABCD_fMRI \
    --codebook-size 2048 \
    --epochs 50 \
    --lr 1e-4

# Stage 2: Freeze encoder, train large decoder (Week 7-8)
python train_vqvae.py \
    --stage 2 \
    --checkpoint stage1_best.pt \
    --freeze-encoder \
    --decoder-layers 12 \
    --epochs 30 \
    --lr 5e-5
```

---

#### **Preference Tuning with Biological Constraints**

**ESM3 Innovation**:
- Create **preference pairs**: (good protein, bad protein)
- "Good" = high pTM score (structure prediction confidence), low cRMSD (deviation from target)
- "Bad" = opposite properties
- Train model to assign **higher likelihood** to good samples

**Brain-Genomics Adaptation**:
```python
class ClinicalPreferenceTuning:
    """
    Preference pairs for developmental disorder prediction
    """
    def create_preference_pairs(self, predictions, clinical_outcomes):
        """
        Good prediction: Correctly identified ASD at 3 years old
        Bad prediction: Missed diagnosis (false negative) or false positive
        """
        good_samples = []
        bad_samples = []

        for pred, outcome in zip(predictions, clinical_outcomes):
            if pred.diagnosis_confidence > 0.8 and pred.label == outcome.label:
                # High confidence + correct diagnosis
                good_samples.append((pred.brain_state, pred.genomic_profile))
            elif pred.diagnosis_confidence > 0.8 and pred.label != outcome.label:
                # High confidence + wrong diagnosis (very bad)
                bad_samples.append((pred.brain_state, pred.genomic_profile))

        return list(zip(good_samples, bad_samples))

    def preference_loss(self, model, good_batch, bad_batch):
        """DPO (Direct Preference Optimization) loss"""
        good_logprob = model.log_likelihood(good_batch)
        bad_logprob = model.log_likelihood(bad_batch)

        # Model should assign higher probability to good samples
        loss = -torch.log(torch.sigmoid(good_logprob - bad_logprob))
        return loss.mean()
```

**Dataset Construction**:
- Use **longitudinal validation**: Cases where 3-year prediction matched 10-year outcome = "good"
- Cases where early prediction was incorrect = "bad"
- Estimated 500 preference pairs from 3,000 cohort (assuming 20-year follow-up)

---

### 1.3 Loss Functions & Optimization

**ESM3 Multi-Task Loss**:
```python
total_loss = (
    λ_seq * sequence_reconstruction_loss +
    λ_struct * structure_prediction_loss +
    λ_func * function_classification_loss +
    λ_pref * preference_tuning_loss
)
```

**Brain-Genomics Multi-Task Loss**:
```python
class NeuroGenomicLoss(nn.Module):
    def __init__(self,
                 λ_brain=1.0,      # Brain trajectory prediction
                 λ_genomic=0.5,    # Genomic variant interpretation
                 λ_diagnosis=2.0,  # Diagnosis accuracy (highest weight)
                 λ_prognosis=1.5,  # Prognosis prediction
                 λ_biomarker=0.8): # Biomarker identification
        super().__init__()
        self.weights = [λ_brain, λ_genomic, λ_diagnosis, λ_prognosis, λ_biomarker]

    def forward(self, predictions, targets):
        L_brain = F.mse_loss(predictions.future_brain, targets.future_brain)
        L_genomic = F.cross_entropy(predictions.variant_effects, targets.variant_labels)
        L_diagnosis = focal_loss(predictions.diagnosis_logits, targets.diagnosis)  # Handle class imbalance
        L_prognosis = trajectory_loss(predictions.trajectory, targets.clinical_trajectory)
        L_biomarker = sparse_attention_loss(predictions.attention_weights)  # Encourage sparse biomarkers

        total = sum(w * L for w, L in zip(self.weights, [L_brain, L_genomic, L_diagnosis, L_prognosis, L_biomarker]))
        return total
```

**Optimizer Configuration** (following ESM3 scale):
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),  # ESM3 uses (0.9, 0.98) instead of default (0.9, 0.999)
    weight_decay=0.01,
    eps=1e-6
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,
    eta_min=1e-6
)
```

---

### 1.4 Synthetic Data Augmentation Strategy

**ESM3 Challenge**: Limited experimental protein structures (~200K in PDB) vs. billions of sequences
**ESM3 Solution**: Generate synthetic structures using AlphaFold2, synthetic functions via homology

**Our Challenge**: Limited Korean pediatric fMRI+WGS pairs (~500-1000) vs. need for 10K+ samples
**Our Solution**:

```python
class SyntheticDataGenerator:
    """Generate synthetic brain-genomic pairs for augmentation"""

    def __init__(self, real_data_loader):
        self.real_brain = []
        self.real_genomics = []
        for brain, genomic in real_data_loader:
            self.real_brain.append(brain)
            self.real_genomics.append(genomic)

    def generate_synthetic_pairs(self, n_samples=5000):
        """
        Strategy 1: MixUp in latent space
        """
        synthetic_pairs = []
        for _ in range(n_samples):
            # Sample two random real samples
            i, j = random.sample(range(len(self.real_brain)), 2)

            # Mix in latent space (α = 0.3-0.7 for realistic interpolation)
            α = random.uniform(0.3, 0.7)
            brain_mixed = self.mix_latent(self.real_brain[i], self.real_brain[j], α)
            genomic_mixed = self.mix_genomic_variants(self.real_genomics[i], self.real_genomics[j], α)

            synthetic_pairs.append((brain_mixed, genomic_mixed))

        return synthetic_pairs

    def mix_genomic_variants(self, genomic_a, genomic_b, alpha):
        """
        Mix two genomic profiles by randomly selecting variants
        """
        # For each gene locus, probabilistically select variant from A or B
        mixed_variants = {}
        for locus in set(genomic_a.keys()) | set(genomic_b.keys()):
            if random.random() < alpha:
                mixed_variants[locus] = genomic_a.get(locus, 'REF')
            else:
                mixed_variants[locus] = genomic_b.get(locus, 'REF')
        return mixed_variants
```

**Augmentation Pipeline**:
1. **Real data**: 1,000 Korean fMRI+WGS pairs
2. **Public transfer**: 10,000 ABCD fMRI (no genomics) + 5,000 UK Biobank (genomics, limited fMRI)
3. **Synthetic**: 5,000 MixUp pairs + 3,000 GANs-generated pairs
4. **Total training**: ~24,000 samples (24× augmentation)

---

## Part 2: BioReason Technical Deep Dive

### 2.1 Cross-Modal Connector Architecture

**BioReason Key Innovation**:
**DNA foundation model (frozen) → Contextualized embeddings → Prepended to LLM input**

```
BioReason Architecture:
┌─────────────────────────────────────────┐
│  DNA Input (1024 bp sequence)           │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│  DNA Foundation Model (FROZEN)          │
│  - Pre-trained on genomic sequences     │
│  - Output: [1024, 768] embeddings       │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│  Cross-Modal Connector (TRAINABLE)      │
│  - Linear projection: 768 → 4096        │
│  - Layer norm + activation              │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│  Concatenate with Text Query            │
│  [DNA_embed_1, ..., DNA_embed_128,      │
│   "Question:", query_token_1, ...]      │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│  LLM (Qwen-7B, TRAINABLE)               │
│  - Unified attention over DNA + text    │
│  - Generate reasoning steps + answer    │
└─────────────────────────────────────────┘
```

**Code Implementation**:
```python
class BioReasonConnector(nn.Module):
    """
    Adapts frozen DNA embeddings to LLM input space
    """
    def __init__(self, dna_dim=768, llm_dim=4096):
        super().__init__()
        self.dna_encoder = load_frozen_dna_model()  # HyenaDNA or Nucleotide Transformer
        self.connector = nn.Sequential(
            nn.Linear(dna_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, dna_sequence, text_query, llm_tokenizer):
        # 1. Encode DNA (frozen)
        with torch.no_grad():
            dna_embeddings = self.dna_encoder(dna_sequence)  # [B, 1024, 768]

        # 2. Project to LLM space
        dna_embeddings_proj = self.connector(dna_embeddings)  # [B, 1024, 4096]

        # 3. Tokenize text query
        text_tokens = llm_tokenizer(text_query, return_tensors='pt')
        text_embeddings = self.llm.get_input_embeddings()(text_tokens['input_ids'])

        # 4. Concatenate [DNA embeddings] + [text embeddings]
        combined = torch.cat([dna_embeddings_proj, text_embeddings], dim=1)

        return combined
```

---

### 2.2 Brain-Genomics Connector (Adapted from BioReason)

**Our Adaptation**:
**Brain foundation model (SwiFT) + Genomic foundation model (HyenaDNA) → Joint connector → Clinical LLM**

```python
class NeuroGenomicConnector(nn.Module):
    """
    Multi-modal connector for brain imaging + genomics → language reasoning
    """
    def __init__(self,
                 brain_dim=768,      # SwiFT output dimension
                 genomic_dim=256,    # HyenaDNA dimension
                 llm_dim=4096):      # Qwen or LLaMA-3 dimension
        super().__init__()

        # Frozen foundation models
        self.brain_encoder = load_swift_model(pretrained=True)
        self.genomic_encoder = load_hyenaDNA_model(pretrained=True)

        # Trainable connectors
        self.brain_connector = nn.Sequential(
            nn.Linear(brain_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU()
        )

        self.genomic_connector = nn.Sequential(
            nn.Linear(genomic_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU()
        )

        # Cross-modal fusion (BioReason doesn't have this - our innovation)
        self.cross_attention = nn.MultiheadAttention(llm_dim, num_heads=16)

        # LLM for reasoning
        self.llm = load_clinical_llm('meta-llama/Llama-3-8B')

    def forward(self, fmri_volume, genomic_sequence, clinical_question):
        # 1. Encode brain (frozen)
        with torch.no_grad():
            brain_embeddings = self.brain_encoder(fmri_volume)  # [B, 128, 768]

        # 2. Encode genomics (frozen)
        with torch.no_grad():
            genomic_embeddings = self.genomic_encoder(genomic_sequence)  # [B, 512, 256]

        # 3. Project to LLM space
        brain_proj = self.brain_connector(brain_embeddings)      # [B, 128, 4096]
        genomic_proj = self.genomic_connector(genomic_embeddings)  # [B, 512, 4096]

        # 4. Cross-modal fusion (our innovation beyond BioReason)
        # Brain attends to genomics, genomics attends to brain
        brain_fused, _ = self.cross_attention(brain_proj, genomic_proj, genomic_proj)
        genomic_fused, _ = self.cross_attention(genomic_proj, brain_proj, brain_proj)

        # 5. Concatenate all modalities + text
        text_embeddings = self.llm.embed_text(clinical_question)
        combined = torch.cat([
            brain_fused,       # [B, 128, 4096]
            genomic_fused,     # [B, 512, 4096]
            text_embeddings    # [B, query_len, 4096]
        ], dim=1)

        # 6. Generate reasoning
        reasoning_output = self.llm.generate(
            inputs_embeds=combined,
            max_new_tokens=512,
            return_dict_in_generate=True
        )

        return reasoning_output
```

**Example Usage**:
```python
connector = NeuroGenomicConnector()

# Input data
fmri = load_fmri_scan('patient_001_3years_old.nii.gz')  # 4D volume
genomic_seq = load_wgs_variants('patient_001_variants.vcf')  # VCF file → sequence
question = """
Given this child's brain imaging at age 3 and their genetic profile,
predict the likelihood of autism spectrum disorder diagnosis by age 7.
Provide step-by-step biological reasoning linking genetics → brain changes → behavioral symptoms.
"""

# Generate clinical reasoning
output = connector(fmri, genomic_seq, question)
print(output.text)
```

**Expected Output Format** (following BioReason's `<think>` token approach):
```
<think>
Step 1: Genetic Analysis
- Detected pathogenic variant in SHANK3 gene (chr22:51135320 C>T)
- This variant is known to disrupt synaptic function in prefrontal cortex

Step 2: Brain Imaging Correlation
- Observed reduced fMRI activation in bilateral prefrontal cortex (BA9/46)
- White matter integrity (FA) decreased in uncinate fasciculus (DTI)
- These regions are critical for social cognition and executive function

Step 3: Developmental Trajectory Prediction
- Based on normative brain charts, current connectivity is 2.3 SD below age-matched peers
- Historical data shows 78% of children with this genetic-brain profile develop ASD by age 7
- Predicted ADOS-2 score trajectory: Current 8 → Age 7: 14-18 (clinical threshold: >10)

Step 4: Mechanistic Link
SHANK3 variant → Synaptic dysfunction → Reduced prefrontal activation →
Social communication deficits → ASD diagnosis
</think>

<answer>
Likelihood of ASD diagnosis by age 7: 78% (95% CI: 65-88%)
Recommended interventions: Early behavioral therapy, monitor prefrontal development via annual fMRI
</answer>
```

---

### 2.3 Training Pipeline (BioReason's 3-Stage Approach)

**Stage 1: Supervised Fine-Tuning (SFT)**
```python
# Dataset: KEGG pathway reasoning (BioReason used 1,449 examples)
# Our adaptation: Clinical reasoning dataset

class ClinicalReasoningDataset:
    """
    Korean developmental disorder reasoning dataset
    """
    def __init__(self):
        self.examples = [
            {
                'brain_scan': 'patient_001_fmri.nii.gz',
                'genomics': 'patient_001_variants.vcf',
                'question': 'Predict developmental outcome at age 5',
                'reasoning_steps': [
                    'Identified CHD8 de novo mutation',
                    'Observed macrocephaly in structural MRI (98th percentile)',
                    'Reduced amygdala-PFC connectivity',
                    'These features correlate with high-functioning ASD phenotype'
                ],
                'answer': 'High-functioning ASD with normal IQ, requires social skills therapy'
            },
            # ... 500-1000 examples from 20-year longitudinal cohort
        ]

# Training
python train_sft.py \
    --dataset korean_clinical_reasoning.json \
    --base-model meta-llama/Llama-3-8B \
    --brain-encoder swift_pretrained.pt \
    --genomic-encoder hyenaDNA_pretrained.pt \
    --epochs 10 \
    --batch-size 4 \
    --lr 5e-5 \
    --output-dir ./checkpoints/sft
```

**Stage 2: Reinforcement Learning (GRPO - Group Relative Policy Optimization)**

BioReason's RL setup:
- **Reward function**: Correctness (2.0 if answer correct, 0.0 otherwise)
- **Algorithm**: GRPO (more stable than PPO for small datasets)

```python
class ClinicalRewardModel:
    """
    Reward model for clinical prediction accuracy
    """
    def __init__(self, validation_cohort):
        self.validation_data = validation_cohort  # Known outcomes (age 7-20)

    def compute_reward(self, prediction, ground_truth):
        """
        Multi-component reward:
        1. Diagnosis accuracy: +2.0 if correct, -1.0 if wrong
        2. Confidence calibration: Bonus if high confidence + correct
        3. Reasoning coherence: NLI score between steps
        4. Clinical safety: Penalty for harmful recommendations
        """
        rewards = {}

        # Diagnosis accuracy
        if prediction.diagnosis == ground_truth.diagnosis:
            rewards['accuracy'] = 2.0
            if prediction.confidence > 0.8:
                rewards['calibration'] = 0.5  # Well-calibrated
        else:
            rewards['accuracy'] = -1.0
            if prediction.confidence > 0.8:
                rewards['calibration'] = -0.5  # Overconfident wrong prediction (bad)

        # Reasoning coherence (use NLI model)
        coherence_score = self.check_logical_coherence(prediction.reasoning_steps)
        rewards['coherence'] = coherence_score  # 0.0 to 1.0

        # Clinical safety check
        if self.contains_harmful_recommendation(prediction.recommendation):
            rewards['safety'] = -5.0  # Large penalty
        else:
            rewards['safety'] = 0.0

        total_reward = sum(rewards.values())
        return total_reward, rewards

# Training
python train_rl.py \
    --sft-checkpoint ./checkpoints/sft/best.pt \
    --algorithm grpo \
    --reward-model clinical_reward.py \
    --validation-cohort korean_longitudinal_validation.json \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-5 \
    --output-dir ./checkpoints/rl
```

**Stage 3: Evaluation & Iteration**
```python
# BioReason achieved: 88% → 97% accuracy on KEGG disease pathway prediction
# Our target: 70% → 85%+ accuracy on developmental disorder prediction

python evaluate.py \
    --checkpoint ./checkpoints/rl/best.pt \
    --test-set korean_test_cohort.json \
    --metrics diagnosis_accuracy,auc_roc,calibration_error \
    --output-file evaluation_report.json
```

---

### 2.4 Dataset Construction Strategy (Following BioReason)

**BioReason Dataset**:
- **KEGG**: 1,449 examples spanning 37 diseases
- **8:1:1 split**: Train (1,159) / Val (145) / Test (145)
- **Reasoning traces**: Genetic variant → Molecular pathway → Disease phenotype

**Our Dataset Construction**:

**Step 1: Retrospective Cohort Analysis (Months 1-3)**
```
Korean 20-Year Longitudinal Cohort:
├─ Total: 3,000 children
├─ ASD: 450 (15%)
├─ ADHD: 300 (10%)
├─ Typical: 2,250 (75%)

Data Collection Timeline:
├─ Baseline (Age 0-3): Brain MRI, genetic screening
├─ Follow-up (Age 5, 7, 10): Clinical diagnosis, cognitive testing
├─ Long-term (Age 15-20): Final diagnosis, functional outcome

Available Multimodal Pairs:
├─ Brain + Genetics + Diagnosis: ~800 cases
├─ Brain only: ~2,500 cases
├─ Genetics only: ~1,200 cases
```

**Step 2: Reasoning Trace Generation (Months 4-5)**
```python
# Use expert clinicians + AI assistance to create reasoning traces

class ReasoningTraceGenerator:
    """
    Semi-automated generation of clinical reasoning traces
    """
    def generate_trace(self, patient_data, clinical_expert):
        """
        Input: Patient multimodal data + expert annotations
        Output: Structured reasoning steps
        """

        # Step 1: Genetic analysis
        genetic_findings = self.analyze_variants(patient_data.wgs)
        genetic_step = f"Identified {len(genetic_findings)} risk variants: {genetic_findings}"

        # Step 2: Brain imaging findings
        brain_findings = self.analyze_brain_scan(patient_data.fmri)
        brain_step = f"Brain connectivity analysis: {brain_findings}"

        # Step 3: Integration (requires expert validation)
        integration_step = clinical_expert.validate_integration(
            genetic=genetic_findings,
            brain=brain_findings,
            diagnosis=patient_data.final_diagnosis
        )

        # Step 4: Prediction rationale
        prediction_step = f"Based on genetic-brain profile, predict {patient_data.final_diagnosis} " \
                         f"with {patient_data.confidence}% confidence"

        reasoning_trace = [genetic_step, brain_step, integration_step, prediction_step]
        return reasoning_trace

# Generate for all 800 multimodal cases
for patient in korean_cohort:
    trace = generator.generate_trace(patient, expert_panel)
    dataset.append({
        'brain': patient.fmri,
        'genomics': patient.wgs,
        'reasoning': trace,
        'answer': patient.final_diagnosis
    })
```

**Step 3: Data Split**
```
Training Set (640 cases, 80%):
├─ ASD: 350
├─ ADHD: 150
├─ Typical development: 140

Validation Set (80 cases, 10%):
├─ ASD: 44
├─ ADHD: 19
├─ Typical: 17

Test Set (80 cases, 10%):
├─ ASD: 44
├─ ADHD: 19
├─ Typical: 17
```

---

## Part 3: COMICAL (Brain Imaging-Genomics Integration)

### 3.1 Architecture: Contrastive Learning for Cross-Modal Association

**COMICAL Insight**: Use CLIP-style contrastive learning to align brain imaging and genomics in a shared embedding space.

```
COMICAL Architecture:
┌────────────────────┐         ┌────────────────────┐
│  Brain Imaging     │         │  Genetic Variants  │
│  (fMRI/structural) │         │  (5,603 SNPs)      │
└─────────┬──────────┘         └─────────┬──────────┘
          │                              │
          ↓                              ↓
┌────────────────────┐         ┌────────────────────┐
│ Vision Transformer │         │ Genomic Transformer│
│ (Brain Encoder)    │         │ (custom tokenizer) │
└─────────┬──────────┘         └─────────┬──────────┘
          │                              │
          ↓                              ↓
┌────────────────────────────────────────────────────┐
│      Shared Embedding Space (512-dim)              │
│  Contrastive Loss: Maximize similarity of          │
│  (brain, genomic) pairs from same patient          │
└─────────┬──────────────────────────────────────────┘
          ↓
┌────────────────────────────────────────────────────┐
│  Cross-Modal Attention (many-to-many associations) │
│  - Brain region A ↔ Genomic variant B              │
│  - Discovers novel biomarkers                      │
└────────────────────────────────────────────────────┘
```

**Code Implementation**:
```python
class COMICAL_Neuro(nn.Module):
    """
    Contrastive Learning for Brain-Genomics Association
    Adapted from COMICAL (IBM Research, 2024)
    """
    def __init__(self,
                 brain_encoder='ViT-B/16',
                 genomic_encoder='transformer',
                 embedding_dim=512):
        super().__init__()

        # Brain encoder (ViT for 3D fMRI)
        self.brain_encoder = ViT3D(
            image_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            dim=768,
            depth=12,
            heads=12
        )

        # Genomic encoder (transformer with custom tokenization)
        self.genomic_encoder = GenomicTransformer(
            vocab_size=5,  # A, T, C, G, N
            max_length=10000,  # 10kb context window
            dim=768,
            depth=12,
            heads=12
        )

        # Projection heads to shared space
        self.brain_proj = nn.Linear(768, embedding_dim)
        self.genomic_proj = nn.Linear(768, embedding_dim)

        # Temperature for contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, brain_batch, genomic_batch):
        # Encode modalities
        brain_features = self.brain_encoder(brain_batch)  # [B, 768]
        genomic_features = self.genomic_encoder(genomic_batch)  # [B, 768]

        # Project to shared space
        brain_embed = F.normalize(self.brain_proj(brain_features), dim=-1)
        genomic_embed = F.normalize(self.genomic_proj(genomic_features), dim=-1)

        # Contrastive loss (InfoNCE)
        logits = torch.matmul(brain_embed, genomic_embed.T) / self.temperature
        labels = torch.arange(len(brain_batch), device=logits.device)

        loss_brain = F.cross_entropy(logits, labels)
        loss_genomic = F.cross_entropy(logits.T, labels)
        loss = (loss_brain + loss_genomic) / 2

        return loss, brain_embed, genomic_embed

    def find_associations(self, brain_embed, genomic_embed, top_k=10):
        """
        Discover brain-genomic associations
        Returns top-k genomic variants associated with each brain region
        """
        similarity_matrix = torch.matmul(brain_embed, genomic_embed.T)
        top_associations = torch.topk(similarity_matrix, k=top_k, dim=-1)
        return top_associations
```

**Training Script**:
```bash
# Pre-training on UK Biobank (40,426 samples)
python train_comical.py \
    --dataset uk_biobank \
    --brain-modality fmri,structural \
    --genomic-variants snp_array \
    --batch-size 256 \
    --epochs 100 \
    --lr 1e-4 \
    --output-dir ./checkpoints/comical_pretrain

# Fine-tuning on Korean cohort (800 samples)
python finetune_comical.py \
    --pretrained-checkpoint ./checkpoints/comical_pretrain/best.pt \
    --dataset korean_developmental_disorder \
    --batch-size 32 \
    --epochs 50 \
    --lr 5e-5 \
    --output-dir ./checkpoints/comical_korean
```

---

### 3.2 COMICAL Results & Transplantation

**COMICAL Performance** (UK Biobank, 10 neurological diseases):
- Discovered **significant associations** between genetic markers and imaging-derived phenotypes
- **Predicted unseen clinical outcomes** from learned representations
- **Transfer learning** enabled across diseases

**Our Adaptation Targets**:
```
Transfer Learning Pipeline:
1. Pre-train on UK Biobank (40K subjects, adult brains)
2. Domain adaptation to pediatric brains (ABCD Study, 11K children)
3. Fine-tune on Korean cohort (800 subjects with longitudinal data)
4. Evaluate on held-out test set (80 subjects)

Expected Performance:
├─ Baseline (no transfer): AUC 0.68-0.72
├─ With pre-training: AUC 0.78-0.82
├─ With GRPO fine-tuning: AUC 0.83-0.87
└─ Target (world-class): AUC > 0.85, Specificity > 80%
```

---

## Part 4: Concrete Implementation Roadmap

### Phase 1: Foundation Setup (Weeks 1-4)

**Week 1-2: Infrastructure & Data Preparation**
```bash
# Tasks
1. Set up DGX A100 cluster (4× A100 80GB)
2. Install foundation model frameworks
   - ESM: pip install fair-esm
   - BioReason: git clone https://github.com/bowang-lab/BioReason
   - COMICAL: git clone https://github.com/IBM/comical
3. Prepare Korean dataset
   - Convert DICOM → NIfTI for brain imaging
   - VCF variant calling from WGS data
   - Create metadata CSV with clinical labels

# Code
git clone https://github.com/your-org/korean-neuro-foundation.git
cd korean-neuro-foundation

# Install dependencies
pip install torch transformers fair-esm nibabel pybedtools

# Data preprocessing
python scripts/preprocess_brain_imaging.py \
    --input-dir /data/korean_cohort/dicoms \
    --output-dir /data/processed/fmri \
    --modality fmri,structural

python scripts/preprocess_genomics.py \
    --input-vcf /data/korean_cohort/variants/*.vcf \
    --output-dir /data/processed/genomics \
    --reference hg38
```

**Week 3-4: Baseline Model Training**
```bash
# Train SwiFT (4D fMRI transformer) on ABCD public dataset
python train_swift_baseline.py \
    --dataset ABCD \
    --n-samples 11000 \
    --epochs 50 \
    --batch-size 16 \
    --output-dir ./checkpoints/swift_baseline

# Train HyenaDNA on 1000 Genomes Project
python train_hyenaDNA_baseline.py \
    --dataset 1000genomes \
    --context-length 10000 \
    --epochs 30 \
    --batch-size 64 \
    --output-dir ./checkpoints/hyenaDNA_baseline
```

**Deliverable**: Functional baseline models, processed Korean dataset

---

### Phase 2: Multi-Modal Integration (Weeks 5-8)

**Week 5-6: Cross-Modal Connector Implementation**
```python
# File: src/models/neuro_genomic_connector.py

class NeuroGenomicConnector(nn.Module):
    """Main model following BioReason architecture"""
    # (Code from Section 2.2 above)
    pass

# Training script
python train_connector.py \
    --brain-encoder ./checkpoints/swift_baseline/best.pt \
    --genomic-encoder ./checkpoints/hyenaDNA_baseline/best.pt \
    --llm meta-llama/Llama-3-8B \
    --dataset korean_clinical_reasoning.json \
    --epochs 10 \
    --batch-size 4 \
    --lr 5e-5
```

**Week 7-8: Contrastive Pre-Training (COMICAL approach)**
```bash
# Pre-train on UK Biobank
python train_comical_pretrain.py \
    --dataset uk_biobank_imaging_genetics \
    --n-samples 40000 \
    --epochs 100 \
    --batch-size 256 \
    --gpus 4

# Fine-tune on Korean data
python train_comical_finetune.py \
    --pretrained ./checkpoints/comical_ukb/best.pt \
    --dataset korean_cohort \
    --n-samples 800 \
    --epochs 50 \
    --batch-size 32
```

**Deliverable**: Integrated multi-modal model, contrastive embeddings

---

### Phase 3: Reasoning & Reinforcement Learning (Weeks 9-12)

**Week 9-10: Clinical Reasoning Dataset Creation**
```python
# Semi-automated reasoning trace generation
python scripts/generate_reasoning_traces.py \
    --input-cohort korean_longitudinal_data.csv \
    --expert-annotations expert_panel_annotations.json \
    --output reasoning_dataset.json \
    --n-examples 800

# Validate with clinical experts
python scripts/validate_reasoning.py \
    --dataset reasoning_dataset.json \
    --expert-panel 3_clinicians \
    --output validated_reasoning.json
```

**Week 11-12: Reinforcement Learning Fine-Tuning**
```bash
# Stage 1: Supervised Fine-Tuning
python train_sft.py \
    --dataset validated_reasoning.json \
    --base-model ./checkpoints/connector/best.pt \
    --epochs 10 \
    --batch-size 4 \
    --lr 5e-5 \
    --output-dir ./checkpoints/sft

# Stage 2: GRPO Reinforcement Learning
python train_grpo.py \
    --sft-checkpoint ./checkpoints/sft/best.pt \
    --reward-model clinical_reward.py \
    --validation-cohort korean_validation.json \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-5 \
    --output-dir ./checkpoints/grpo
```

**Deliverable**: Production-ready reasoning model

---

### Phase 4: Evaluation & Deployment (Weeks 13-16)

**Week 13-14: Comprehensive Evaluation**
```bash
# Evaluation metrics
python evaluate_model.py \
    --checkpoint ./checkpoints/grpo/best.pt \
    --test-set korean_test_cohort.json \
    --metrics accuracy,auc_roc,calibration,reasoning_coherence \
    --output-file evaluation_report.json

# Biomarker discovery
python discover_biomarkers.py \
    --model ./checkpoints/grpo/best.pt \
    --attention-analysis \
    --output-file discovered_biomarkers.csv
```

**Week 15-16: Clinical Validation & Deployment**
```python
# Deploy as clinical decision support system
python deploy_clinical_system.py \
    --model ./checkpoints/grpo/best.pt \
    --endpoint https://clinical-api.hospital.kr \
    --authentication oauth2 \
    --monitoring prometheus

# Prospective validation study (3-month trial)
# - Enroll 50 new patients (age 2-3)
# - AI prediction vs. clinician prediction
# - Follow-up at age 5 for ground truth
```

**Deliverable**: Validated clinical system, biomarker catalog

---

## Part 5: Resource Requirements

### 5.1 Computational Resources

**Training Infrastructure**:
```
Hardware:
├─ GPU: 4× NVIDIA A100 80GB (DGX A100 system)
├─ CPU: 128 cores (2× AMD EPYC 7763)
├─ RAM: 1TB DDR4
├─ Storage: 100TB NVMe SSD (for imaging data)

Software Stack:
├─ OS: Ubuntu 22.04 LTS
├─ CUDA: 12.1
├─ PyTorch: 2.1.0
├─ Transformers: 4.35.0
├─ Custom libraries: FSL (fMRI), GATK (genomics)

Estimated Compute:
├─ Pre-training (UK Biobank): 1,000 A100-hours
├─ Fine-tuning (Korean cohort): 200 A100-hours
├─ GRPO reinforcement learning: 100 A100-hours
└─ Total: ~1,300 A100-hours (~$3,900 on cloud, $0 on owned DGX)
```

---

### 5.2 Data Requirements

**Dataset Summary**:
```
Public Pre-training Data (Free):
├─ ABCD Study: 11,000 children, fMRI + structural MRI
├─ UK Biobank: 40,000 adults, brain imaging + genomics
├─ 1000 Genomes: 2,504 individuals, whole-genome sequences
└─ Total: ~50TB imaging + 10TB genomics

Korean Proprietary Data:
├─ Longitudinal cohort: 3,000 children (20-year follow-up)
├─ Multimodal pairs: 800 (fMRI + WGS + diagnosis)
├─ Brain imaging only: 2,500 scans
├─ Genomics only: 1,200 samples
└─ Total: ~15TB imaging + 2TB genomics

Synthetic Augmentation (Generated):
├─ MixUp pairs: 5,000
├─ GAN-generated: 3,000
└─ Total: ~8,000 synthetic samples
```

---

### 5.3 Personnel Requirements

**Team Composition** (Months 1-12):
```
Core Team:
├─ Principal Investigator (PI): 1× (20% FTE) - Strategy & oversight
├─ AI/ML Engineers: 2× (100% FTE) - Model development
├─ Computational Neuroscientist: 1× (80% FTE) - Brain imaging expertise
├─ Bioinformatician: 1× (80% FTE) - Genomics pipeline
├─ Clinical Research Coordinator: 1× (50% FTE) - Data curation
└─ DevOps Engineer: 1× (50% FTE) - Infrastructure

Advisory Panel:
├─ Pediatric Neurologist: 2× (10% FTE each) - Clinical validation
├─ Geneticist: 1× (10% FTE) - Variant interpretation
└─ Bioethicist: 1× (5% FTE) - Ethics compliance

Total FTE: ~5.5
```

---

### 5.4 Budget Estimate (12 Months)

```
Personnel (60% of total):
├─ AI/ML Engineers (2×): $200K
├─ Scientists (2×): $180K
├─ Coordinators/DevOps (2.5×): $120K
└─ Subtotal: $500K

Compute & Infrastructure (25%):
├─ DGX A100 (amortized): $50K
├─ Cloud storage (100TB): $30K
├─ Software licenses: $20K
└─ Subtotal: $100K

Data Acquisition & Processing (10%):
├─ Sequencing costs (WGS, 200 samples): $60K
├─ MRI scan time: $20K
└─ Subtotal: $80K

Miscellaneous (5%):
├─ Conference travel: $10K
├─ Publication fees: $5K
└─ Subtotal: $15K

TOTAL: ~$700K USD (₩900M KRW)
```

---

## Part 6: Success Metrics & Milestones

### 6.1 Technical Milestones

**Month 3**:
- ✅ Baseline SwiFT model trained on ABCD (AUC > 0.70)
- ✅ HyenaDNA genomic encoder functional
- ✅ Korean dataset preprocessed and validated

**Month 6**:
- ✅ Cross-modal connector integrated (BioReason architecture)
- ✅ COMICAL contrastive pre-training complete (UK Biobank)
- ✅ Initial clinical reasoning dataset (500 traces)

**Month 9**:
- ✅ Supervised fine-tuning complete
- ✅ GRPO reinforcement learning operational
- ✅ Preliminary validation: AUC > 0.80

**Month 12**:
- ✅ Final model: AUC > 0.85, Specificity > 80%
- ✅ Biomarker catalog: 20+ novel genetic-brain associations
- ✅ Clinical deployment prototype ready

---

### 6.2 Scientific Deliverables

**Publications** (Target: 2-3 high-impact papers):
1. **Nature Medicine / Cell**: "Multimodal Foundation Model for Developmental Disorder Prediction"
   - First brain-genomics foundation model for pediatric neurodevelopment
   - 20-year longitudinal validation

2. **NeurIPS / ICML**: "Cross-Modal Reasoning for Clinical AI: Integrating Brain Imaging and Genomics"
   - Novel connector architecture
   - GRPO application to medical reasoning

3. **Bioinformatics**: "Discovered Genetic-Brain Biomarkers for Early ASD Detection"
   - 20+ novel biomarkers from COMICAL analysis
   - Validated in zebrafish models

**Open-Source Contributions**:
- GitHub repository: `korean-neuro-foundation`
- Pre-trained model weights (upon publication)
- Clinical reasoning dataset (anonymized, IRB-approved)

---

## Part 7: Risk Mitigation & Contingency Plans

### 7.1 Technical Risks

**Risk 1**: Insufficient Korean training data (800 samples too small)
**Mitigation**:
- Heavy reliance on transfer learning (UK Biobank 40K → Korean 800)
- Synthetic data augmentation (MixUp, GANs)
- Federated learning to incorporate multi-center data

**Risk 2**: Model overfitting to small dataset
**Mitigation**:
- Strong regularization (dropout 0.3, weight decay 0.01)
- Cross-validation with 5 folds
- External validation on independent cohort (international collaboration)

**Risk 3**: Poor cross-modal alignment (brain ≠ genomics)
**Mitigation**:
- COMICAL contrastive pre-training explicitly learns alignment
- Use intermediate phenotypes (e.g., brain-based endophenotypes correlate with genetics)
- Multi-task learning with auxiliary tasks (e.g., age prediction, sex classification)

---

### 7.2 Clinical Risks

**Risk**: AI predictions not trusted by clinicians
**Mitigation**:
- Explainable AI: Provide reasoning traces (following BioReason `<think>` approach)
- Attention visualization: Show which brain regions + genes drove prediction
- Prospective validation study with clinician-in-the-loop

**Risk**: Ethical concerns (genetic discrimination, privacy)
**Mitigation**:
- Federated learning (data never leaves hospital)
- Differential privacy during training
- IRB approval + informed consent
- No deployment without clinical validation

---

## Part 8: Competitive Advantage & Innovation

### 8.1 Why This Will Succeed (vs. International Competition)

**Unique Assets**:
1. **20-year longitudinal data**: No other cohort has this depth
2. **Multimodal pairs**: fMRI + WGS + behavior (rare globally)
3. **Homogeneous population**: Korean genetics reduce confounding
4. **Clinical access**: Direct collaboration with Samsung Medical Center

**Technical Innovation**:
1. **First pediatric brain-genomics foundation model** (adults exist, children don't)
2. **Cross-modal reasoning** (beyond association → mechanistic understanding)
3. **GRPO for clinical reasoning** (BioReason applied to neuroscience)

**Clinical Impact**:
1. **Pre-symptomatic diagnosis**: Predict ASD at age 3 (before symptoms)
2. **Personalized prognosis**: Individual trajectory prediction
3. **Actionable biomarkers**: Guide early intervention

---

### 8.2 Comparison to SOTA

| System | Modalities | Reasoning | Pediatric | Korean Data | Performance |
|--------|-----------|-----------|-----------|-------------|-------------|
| **Med-Gemini** | Text + Image | Yes | No | No | AUC 0.85 (general) |
| **BrainLM** | fMRI only | No | Partial | No | - |
| **COMICAL** | Brain + Genetics | No (association only) | No | No | Discovery |
| **BioReason** | Genetics + Text | Yes | No | No | 97% (KEGG) |
| **Our Model** | Brain + Genetics + Text | Yes | **Yes** | **Yes** | Target: AUC > 0.85 |

---

## Conclusion: Action Plan Summary

### Immediate Next Steps (Weeks 1-2)

1. **Secure compute resources**: DGX A100 access confirmed
2. **Download public datasets**:
   - ABCD Study (11K children): https://nda.nih.gov/abcd
   - UK Biobank (40K imaging): https://www.ukbiobank.ac.uk/
   - 1000 Genomes (genomics): https://www.internationalgenome.org/
3. **Preprocess Korean cohort**:
   ```bash
   python scripts/preprocess_pipeline.py \
       --brain-dir /data/korean_cohort/imaging \
       --genomics-dir /data/korean_cohort/vcf \
       --output-dir /data/processed
   ```
4. **Set up code repository**:
   ```bash
   git clone https://github.com/bowang-lab/BioReason
   git clone https://github.com/IBM/comical
   git clone https://github.com/evolutionaryscale/esm
   ```
5. **Hire AI/ML engineer** (critical path)

### 3-Month Sprint (Weeks 1-12)

**Month 1**: Foundation setup, baseline models
**Month 2**: Cross-modal integration, contrastive pre-training
**Month 3**: Clinical reasoning, GRPO fine-tuning, initial validation

**Deliverable**: Working prototype with AUC > 0.80

### 6-Month Target (Weeks 1-24)

**Month 4-5**: Biomarker discovery, external validation
**Month 6**: Clinical deployment, prospective study initiation

**Deliverable**: Production system, submitted Nature Medicine paper

---

## References & Code Repositories

### Papers
1. **ESM3**: Hayes et al. (2024). "Simulating 500 million years of evolution with a language model." *Science*. https://www.science.org/doi/10.1126/science.ads0018
2. **BioReason**: Fei et al. (2025). "BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model." *NeurIPS 2025*. https://arxiv.org/abs/2505.23579
3. **COMICAL**: IBM Research (2024). "A Multimodal Foundation Model for Discovering Genetic Associations with Brain Imaging Phenotypes." *medRxiv*. https://www.medrxiv.org/content/10.1101/2024.11.02.24316653v1
4. **Med-Gemini**: Google DeepMind (2024). "Advancing medical AI with Med-Gemini." https://arxiv.org/html/2406.00631v1
5. **BrainLM**: Abdallah et al. (2024). "BrainLM: A Foundation Model for Brain Activity Recordings." *ICLR 2024*.

### Code Repositories
- **ESM**: https://github.com/evolutionaryscale/esm
- **BioReason**: https://github.com/bowang-lab/BioReason
- **COMICAL**: https://github.com/IBM/comical
- **SwiFT (4D fMRI)**: https://github.com/Saurabh7/SwiFT
- **HyenaDNA**: https://github.com/HazyResearch/hyena-dna

### Datasets
- **ABCD Study**: https://nda.nih.gov/abcd (11,000 children, fMRI)
- **UK Biobank**: https://www.ukbiobank.ac.uk/ (40,000 brain imaging + genetics)
- **1000 Genomes**: https://www.internationalgenome.org/ (2,504 WGS samples)
- **BioReason KEGG Dataset**: https://huggingface.co/bowang-lab/BioReason

---

**Document Status**: Ready for implementation
**Next Update**: After Month 3 milestone review
**Contact**: AI-CoScientist Team (ai-coscientist@samsung.com)
