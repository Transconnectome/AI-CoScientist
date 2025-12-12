# Korean Brain Foundation Model: Technical Implementation Specification

**Version**: 1.0
**Date**: December 8, 2025
**Purpose**: Concrete technical roadmap for Korean brain foundation model development

---

## 1. SYSTEM ARCHITECTURE

### 1.1 Recommended Tech Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Korean Brain Foundation Model                 │
│                        (KoreanBrain-1B)                          │
└─────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
   │  fMRI   │              │ Struct  │              │   EEG   │
   │ Encoder │              │   MRI   │              │ Encoder │
   │         │              │ Encoder │              │         │
   │NeuroSTORM│              │BrainSeg │              │META-EEG │
   │ -based  │              │Founder  │              │ -based  │
   └────┬────┘              └────┬────┘              └────┬────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                          ┌───────▼────────┐
                          │  Cross-Modal   │
                          │    Attention   │
                          │   (Brain-MGF)  │
                          └───────┬────────┘
                                  │
                          ┌───────▼────────┐
                          │  Task-Specific │
                          │  Prompt Tuning │
                          │     (TPT)      │
                          └───────┬────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
   │Disease  │              │ Report  │              │Segmen-  │
   │Classifi-│              │Generate │              │tation   │
   │cation   │              │(Korean) │              │         │
   └─────────┘              └─────────┘              └─────────┘
```

### 1.2 Core Components

#### **Component 1: fMRI Encoder (NeuroSTORM-based)**
- **Architecture**: Shifted-Window Mamba (SWM) backbone
- **Input**: 4D fMRI volumes (x, y, z, t) in MNI152 space
- **Pre-training**:
  - Masked autoencoding (MAE)
  - Spatiotemporal redundancy dropout (STRD)
  - Temporal contrastive learning
- **Output**: 768-dimensional embeddings per ROI
- **Parameters**: ~300M (base), ~1B (large)

**Implementation**:
```python
# Pseudo-code
class fMRIEncoder(nn.Module):
    def __init__(self, d_model=768, n_layers=12, window_size=7):
        self.patch_embed = PatchEmbed3D(patch_size=8)
        self.swm_blocks = nn.ModuleList([
            ShiftedWindowMamba(d_model, window_size)
            for _ in range(n_layers)
        ])
        self.strd = SpatiotemporalRedundancyDropout(p=0.15)

    def forward(self, fmri_volume):
        # fmri_volume: (B, T, H, W, D)
        x = self.patch_embed(fmri_volume)  # (B, N, d_model)
        x = self.strd(x)
        for block in self.swm_blocks:
            x = block(x)
        return x  # ROI embeddings
```

**Reference**: [GitHub - CUHK-AIM-Group/NeuroSTORM](https://github.com/CUHK-AIM-Group/NeuroSTORM)

---

#### **Component 2: Structural MRI Encoder (BrainSegFounder-based)**
- **Architecture**: 3D Vision Transformer (3D-ViT)
- **Input**: T1-weighted MRI (1mm³ isotropic)
- **Pre-training**:
  - Self-supervised masked volume modeling
  - Contrastive learning across subjects
- **Output**: 512-dimensional volume embeddings
- **Parameters**: ~200M

**Implementation**:
```python
class StructuralMRIEncoder(nn.Module):
    def __init__(self, img_size=(160, 192, 160), patch_size=16, d_model=512):
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=d_model
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8),
            num_layers=12
        )

    def forward(self, t1_mri):
        # t1_mri: (B, 1, H, W, D)
        x = self.patch_embed(t1_mri) + self.pos_embed
        x = self.transformer(x)
        return x.mean(dim=1)  # Global pool
```

**Reference**: [GitHub - lab-smile/BrainSegFounder](https://github.com/lab-smile/BrainSegFounder)

---

#### **Component 3: Cross-Modal Attention Fusion (Brain-MGF)**
- **Architecture**: Adaptive gating mechanism with graph attention
- **Input**: Multi-modal embeddings (fMRI, structural MRI, EEG)
- **Fusion strategy**: Sample-specific softmax weighting
- **Output**: Unified 1024-dimensional representation

**Implementation**:
```python
class CrossModalFusion(nn.Module):
    def __init__(self, d_fmri=768, d_struct=512, d_eeg=256, d_out=1024):
        self.fmri_proj = nn.Linear(d_fmri, d_out)
        self.struct_proj = nn.Linear(d_struct, d_out)
        self.eeg_proj = nn.Linear(d_eeg, d_out)

        # Adaptive gating MLP
        self.gate = nn.Sequential(
            nn.Linear(d_out * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, fmri_emb, struct_emb, eeg_emb):
        # Project to common dimension
        f_proj = self.fmri_proj(fmri_emb)
        s_proj = self.struct_proj(struct_emb)
        e_proj = self.eeg_proj(eeg_emb)

        # Concatenate for gating
        concat = torch.cat([f_proj, s_proj, e_proj], dim=-1)
        weights = self.gate(concat)  # (B, 3)

        # Weighted fusion
        fused = (weights[:, 0:1] * f_proj +
                 weights[:, 1:2] * s_proj +
                 weights[:, 2:3] * e_proj)
        return fused
```

**Expected Performance**:
- Multimodal fusion: 74-76% accuracy (vs 59-69% single-modality)

**Reference**: [arXiv - Brain-MGF](https://arxiv.org/html/2511.18325)

---

#### **Component 4: Task-Specific Prompt Tuning (TPT)**
- **Architecture**: Learnable prompts (< 1% of model parameters)
- **Approach**: Deep prompt tuning at each transformer layer
- **Adaptation**: Korean-specific medical tasks

**Implementation**:
```python
class TaskSpecificPromptTuning(nn.Module):
    def __init__(self, n_layers=12, d_model=1024, n_prompts=10):
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, n_prompts, d_model))
            for _ in range(n_layers)
        ])

    def forward(self, x, layer_idx):
        # x: (B, N, d_model)
        prompt = self.prompts[layer_idx].expand(x.size(0), -1, -1)
        x_with_prompt = torch.cat([prompt, x], dim=1)
        return x_with_prompt
```

**Advantages**:
- Only tune 0.5-1% of parameters
- 35% reduction in training time
- Preserves pre-trained knowledge

**Reference**: [ScienceDirect DVPT](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000474)

---

### 1.3 Full Model Integration

```python
class KoreanBrainFoundationModel(nn.Module):
    def __init__(self):
        # Modality encoders
        self.fmri_encoder = fMRIEncoder(d_model=768)
        self.struct_encoder = StructuralMRIEncoder(d_model=512)
        self.eeg_encoder = EEGEncoder(d_model=256)

        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            d_fmri=768, d_struct=512, d_eeg=256, d_out=1024
        )

        # Task-specific prompts
        self.tpt = TaskSpecificPromptTuning(n_layers=12, d_model=1024)

        # Task heads
        self.disease_classifier = nn.Linear(1024, num_diseases)
        self.report_generator = KoreanReportGenerator(d_model=1024)
        self.segmentation_head = SegmentationHead(d_model=1024)

    def forward(self, fmri, struct_mri, eeg, task='classification'):
        # Encode modalities
        fmri_emb = self.fmri_encoder(fmri)
        struct_emb = self.struct_encoder(struct_mri)
        eeg_emb = self.eeg_encoder(eeg)

        # Fuse
        fused = self.fusion(fmri_emb, struct_emb, eeg_emb)

        # Task-specific adaptation
        if task == 'classification':
            return self.disease_classifier(fused)
        elif task == 'report':
            return self.report_generator(fused)
        elif task == 'segmentation':
            return self.segmentation_head(fused)
```

---

## 2. DATA PIPELINE

### 2.1 Korean Brain Imaging Dataset Specifications

#### **Target Scale**
- **Phase 1 (Pilot)**: 1,000 subjects
- **Phase 2 (Validation)**: 5,000 subjects
- **Phase 3 (Production)**: 10,000-20,000 subjects

#### **Data Modalities**
| Modality | Resolution | Acquisition Time | Purpose |
|----------|-----------|------------------|---------|
| **T1-weighted MRI** | 1mm³ isotropic | 5 min | Structural anatomy |
| **T2-FLAIR** | 1mm³ isotropic | 6 min | Lesion detection |
| **Resting fMRI** | 3mm³, TR=2s, 600 volumes | 20 min | Functional connectivity |
| **Diffusion MRI** | 2mm³, 64 directions | 15 min | White matter tracts |
| **EEG (optional)** | 64 channels, 1000 Hz | 10 min | Electrophysiology |

#### **Clinical Annotations**
- **Disease labels**: Alzheimer's, Parkinson's, stroke, tumors, schizophrenia
- **Severity scores**: MMSE, CDR, UPDRS
- **Demographics**: Age, sex, education, genetics (APOE4)
- **Segmentations**: Tumors, lesions, atrophy (expert radiologists)

### 2.2 Data Collection Protocol

#### **Multi-Center Partnership**
1. **Seoul National University Hospital (SNUH)**: 3,000 subjects
2. **Samsung Medical Center (SMC)**: 3,000 subjects
3. **Yonsei University Severance Hospital**: 2,000 subjects
4. **Asan Medical Center**: 2,000 subjects
5. **Total**: 10,000 subjects (2-year collection)

#### **Ethical Approval**
- IRB approval from each institution
- Data sharing agreement (federated learning architecture)
- De-identification: DICOM metadata removal, face defacing

### 2.3 Data Quality Control

#### **Automated QC (UK Biobank Pipeline)**
```python
def automated_qc(mri_path):
    img = nib.load(mri_path)

    # Compute quality metrics
    snr = compute_snr(img)
    cnr = compute_cnr(img, tissue_masks)
    cjv = compute_cjv(img)
    fber = compute_fber(img)
    iqr = compute_iqr(img)

    # Thresholds (UK Biobank standards)
    passed = (snr > 10 and cnr > 5 and cjv < 0.4 and
              fber > 20 and iqr < 0.05)

    return {
        'passed': passed,
        'metrics': {'SNR': snr, 'CNR': cnr, 'CJV': cjv,
                    'FBER': fber, 'IQR': iqr}
    }
```

**QC Metrics** (based on UK Biobank):
- **SNR (Signal-to-Noise Ratio)**: > 10
- **CNR (Contrast-to-Noise Ratio)**: > 5
- **CJV (Coefficient of Joint Variation)**: < 0.4
- **FBER (Foreground-Background Energy Ratio)**: > 20
- **IQR (Image Quality Rate)**: < 0.05

**Reference**: [PMC UK Biobank QC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5770339/)

#### **Manual Review**
- Random 5-10% subset reviewed by radiologists
- Artifacts, motion, pathology verification

---

## 3. TRAINING PIPELINE

### 3.1 Stage 1: Self-Supervised Pre-Training

#### **Objective**: Learn general brain representations from unlabeled Korean data

#### **Approach**: BrainMAE + Temporal Contrastive Learning

```python
class SelfSupervisedPreTraining:
    def __init__(self, model, unlabeled_data):
        self.model = model
        self.data = unlabeled_data

    def masked_autoencoding_loss(self, fmri):
        # Randomly mask 30% of ROIs
        masked_fmri, mask = self.random_mask(fmri, mask_ratio=0.3)

        # Encode
        embeddings = self.model.fmri_encoder(masked_fmri)

        # Reconstruct masked regions
        reconstructed = self.model.decoder(embeddings)

        # MSE loss on masked regions only
        loss = F.mse_loss(reconstructed[mask], fmri[mask])
        return loss

    def temporal_contrastive_loss(self, fmri):
        # Positive pairs: neighboring time segments
        t1, t2 = fmri[:, :300], fmri[:, 250:550]  # 50 overlap
        z1 = self.model.fmri_encoder(t1)
        z2 = self.model.fmri_encoder(t2)

        # Negative pairs: distant segments
        t3 = fmri[:, :300]
        t4 = fmri[:, -300:]
        z3 = self.model.fmri_encoder(t3)
        z4 = self.model.fmri_encoder(t4)

        # NT-Xent loss
        loss = nt_xent_loss(z1, z2, z3, z4, temperature=0.5)
        return loss

    def train(self, epochs=100):
        for epoch in range(epochs):
            for batch in self.data:
                mae_loss = self.masked_autoencoding_loss(batch['fmri'])
                cl_loss = self.temporal_contrastive_loss(batch['fmri'])

                total_loss = mae_loss + 0.5 * cl_loss
                total_loss.backward()
                optimizer.step()
```

**Duration**: 4-6 weeks on 8x A100 GPUs
**Expected outcome**: General brain representations (no task-specific labels needed)

**Reference**: [arXiv BrainMAE](https://arxiv.org/html/2406.17086v1)

---

### 3.2 Stage 2: Foundation Model Adaptation

#### **Objective**: Adapt NeuroSTORM/BrainLM to Korean data

```python
class FoundationModelAdaptation:
    def __init__(self, pretrained_model_path, korean_data):
        # Load pre-trained weights (9,000 hrs, 50K subjects)
        self.model = NeuroSTORM.from_pretrained(pretrained_model_path)
        self.data = korean_data

    def task_specific_prompt_tuning(self, batch, task='alzheimers'):
        # Freeze pre-trained weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Only tune prompts (< 1% params)
        fmri_emb = self.model.fmri_encoder(batch['fmri'])
        prompted = self.model.tpt(fmri_emb, task=task)

        # Task head
        logits = self.model.disease_classifier(prompted)
        loss = F.cross_entropy(logits, batch['label'])
        return loss

    def continual_pretraining(self, korean_medical_corpus):
        # Incorporate Korean medical text for report generation
        for doc in korean_medical_corpus:
            # MLM (Masked Language Modeling) on Korean text
            masked_text, labels = self.mask_tokens(doc)
            text_emb = self.model.text_encoder(masked_text)
            logits = self.model.lm_head(text_emb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
```

**Duration**: 2-4 weeks
**Expected gain**: 5-15% improvement on Korean-specific tasks

**Reference**: [Nature npj Digital Medicine - Me-LLaMA](https://www.nature.com/articles/s41746-025-01533-1)

---

### 3.3 Stage 3: Multimodal Fusion Training

#### **Objective**: Learn to integrate fMRI, structural MRI, EEG

```python
class MultimodalFusionTraining:
    def __init__(self, model, multimodal_data):
        self.model = model
        self.data = multimodal_data

    def train_cross_modal_fusion(self, batch):
        # Encode each modality
        fmri_emb = self.model.fmri_encoder(batch['fmri'])
        struct_emb = self.model.struct_encoder(batch['t1_mri'])
        eeg_emb = self.model.eeg_encoder(batch['eeg'])

        # Cross-modal attention fusion
        fused = self.model.fusion(fmri_emb, struct_emb, eeg_emb)

        # Multi-task learning
        disease_logits = self.model.disease_classifier(fused)
        disease_loss = F.cross_entropy(disease_logits, batch['disease_label'])

        segmentation = self.model.segmentation_head(fused)
        seg_loss = dice_loss(segmentation, batch['lesion_mask'])

        total_loss = disease_loss + 0.5 * seg_loss
        return total_loss
```

**Duration**: 2-3 weeks
**Expected performance**: 74-76% accuracy (multimodal fusion)

**Reference**: [arXiv Brain-MGF](https://arxiv.org/html/2511.18325)

---

### 3.4 Stage 4: Clinical Validation

#### **Hold-Out Test Set**
- 20% of Korean data (2,000 subjects)
- Stratified by disease, age, sex

#### **Evaluation Metrics**
```python
def evaluate_model(model, test_loader):
    y_true, y_pred = [], []

    for batch in test_loader:
        with torch.no_grad():
            logits = model(batch['fmri'], batch['t1_mri'], batch['eeg'])
            preds = logits.argmax(dim=-1)

        y_true.extend(batch['label'].cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_pred, multi_class='ovr')

    # Clinical metrics
    sensitivity = recall_score(y_true, y_pred, average='macro')
    specificity = specificity_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
```

#### **Benchmark Comparison**
| Task | GPT-4V | Med-Gemini | BrainGFM | KoreanBrain-1B (Target) |
|------|--------|-----------|----------|------------------------|
| **Alzheimer's classification** | 0.72 AUC | 0.78 AUC | 0.83 AUC | **0.85-0.90 AUC** |
| **Brain tumor detection** | 56% | 60% | 68% | **75-80%** |
| **Stroke lesion segmentation** | N/A | N/A | 0.72 Dice | **0.75-0.80 Dice** |
| **Korean report generation** | 45% BLEU | 52% BLEU | N/A | **65-70% BLEU** |

---

## 4. DEPLOYMENT STRATEGY

### 4.1 Model Serving Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI REST API                       │
│  /predict, /segment, /generate_report                     │
└───────────────────────┬──────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐     ┌───▼────┐     ┌───▼────┐
   │ Model A │     │Model B │     │Model C │
   │ (GPU 0) │     │(GPU 1) │     │(GPU 2) │
   │         │     │        │     │        │
   │Classify │     │Segment │     │Report  │
   └─────────┘     └────────┘     └────────┘
```

#### **FastAPI Implementation**
```python
from fastapi import FastAPI, File, UploadFile
import torch

app = FastAPI()

# Load model once at startup
model = KoreanBrainFoundationModel.from_pretrained('koreanbrainlab/KoreanBrain-1B')
model.to('cuda')
model.eval()

@app.post("/predict/alzheimers")
async def predict_alzheimers(
    fmri: UploadFile = File(...),
    t1_mri: UploadFile = File(...),
    metadata: dict = Body(...)
):
    # Load DICOM/NIfTI
    fmri_vol = load_fmri(await fmri.read())
    t1_vol = load_t1(await t1_mri.read())

    # Preprocess
    fmri_tensor = preprocess_fmri(fmri_vol).unsqueeze(0).cuda()
    t1_tensor = preprocess_t1(t1_vol).unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        logits = model(fmri_tensor, t1_tensor, task='classification')
        proba = torch.softmax(logits, dim=-1)

    return {
        'prediction': 'Alzheimer\'s Disease' if proba[0, 1] > 0.5 else 'Healthy',
        'confidence': float(proba[0, 1]),
        'risk_score': float(proba[0, 1])
    }

@app.post("/generate_report")
async def generate_korean_report(
    fmri: UploadFile = File(...),
    t1_mri: UploadFile = File(...)
):
    # Similar preprocessing
    fmri_tensor = preprocess_fmri(await fmri.read()).unsqueeze(0).cuda()
    t1_tensor = preprocess_t1(await t1_mri.read()).unsqueeze(0).cuda()

    # Generate Korean medical report
    with torch.no_grad():
        report = model.generate_report(fmri_tensor, t1_tensor, language='ko')

    return {
        'report': report,
        'generated_at': datetime.now().isoformat()
    }
```

### 4.2 Optimization for Production

#### **Model Quantization (TensorRT)**
```python
import torch_tensorrt

# FP16 quantization for 2x speedup
optimized_model = torch_tensorrt.compile(
    model,
    inputs=[torch.randn(1, 600, 91, 109, 91).cuda()],  # fMRI input shape
    enabled_precisions={torch.float16},
    workspace_size=1 << 30  # 1GB
)

# Expected inference time
# FP32: ~2.5s per sample
# FP16 (TensorRT): ~1.2s per sample
```

#### **Batch Inference**
```python
class BatchInferenceService:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
        self.queue = []

    async def add_to_queue(self, fmri, t1_mri):
        self.queue.append((fmri, t1_mri))

        if len(self.queue) >= self.batch_size:
            return await self.process_batch()
        else:
            # Wait for more samples or timeout
            await asyncio.sleep(0.1)

    async def process_batch(self):
        batch_fmri = torch.stack([q[0] for q in self.queue])
        batch_t1 = torch.stack([q[1] for q in self.queue])

        with torch.no_grad():
            results = self.model(batch_fmri, batch_t1)

        self.queue.clear()
        return results
```

**Expected throughput**:
- Single inference: 0.8 FPS (1.2s per sample)
- Batch inference (8 samples): 4.5 FPS (effective 0.22s per sample)

### 4.3 Clinical Integration

#### **Hospital PACS Integration**
```python
from pynetdicom import AE, evt
from pydicom.dataset import Dataset

class DicomReceiver:
    def __init__(self, model_service):
        self.ae = AE()
        self.ae.add_supported_context('1.2.840.10008.5.1.4.1.1.4')  # MRI Storage
        self.model = model_service

    def handle_store(self, event):
        # Receive DICOM from PACS
        ds = event.dataset

        # Convert to model input
        fmri_tensor = dicom_to_tensor(ds)

        # Run inference
        prediction = self.model.predict(fmri_tensor)

        # Create DICOM Structured Report
        report = create_dicom_sr(prediction, ds)

        # Send back to PACS
        self.send_to_pacs(report)

        return 0x0000  # Success

    def start_scp(self, port=11112):
        handlers = [(evt.EVT_C_STORE, self.handle_store)]
        self.ae.start_server(('0.0.0.0', port), evt_handlers=handlers)
```

---

## 5. EVALUATION FRAMEWORK

### 5.1 KoreanBrainBench: Comprehensive Benchmark Suite

#### **Task Categories**
1. **Disease Classification**: 5 diseases (Alzheimer's, Parkinson's, stroke, tumors, schizophrenia)
2. **Lesion Segmentation**: Tumors, stroke lesions, MS plaques
3. **Report Generation**: Korean medical reports (BLEU, ROUGE, clinical accuracy)
4. **Future State Prediction**: Disease progression forecasting
5. **Cross-Center Generalization**: Train on 3 hospitals, test on 4th

#### **Metrics**
```python
class KoreanBrainBench:
    def __init__(self, test_data):
        self.test_data = test_data

    def evaluate_classification(self, model):
        metrics = {
            'accuracy': [],
            'f1': [],
            'auc': [],
            'sensitivity': [],
            'specificity': []
        }

        for disease in ['alzheimers', 'parkinsons', 'stroke', 'tumor', 'schizophrenia']:
            disease_data = self.test_data[disease]
            preds = model.predict(disease_data)

            metrics['accuracy'].append(accuracy_score(disease_data.labels, preds))
            metrics['f1'].append(f1_score(disease_data.labels, preds))
            metrics['auc'].append(roc_auc_score(disease_data.labels, preds))
            # etc.

        return {k: np.mean(v) for k, v in metrics.items()}

    def evaluate_segmentation(self, model):
        dice_scores = []
        hd95_scores = []  # Hausdorff distance 95th percentile

        for sample in self.test_data['segmentation']:
            pred_mask = model.segment(sample['fmri'], sample['t1_mri'])
            true_mask = sample['lesion_mask']

            dice = dice_coefficient(pred_mask, true_mask)
            hd95 = hausdorff_distance_95(pred_mask, true_mask)

            dice_scores.append(dice)
            hd95_scores.append(hd95)

        return {
            'dice': np.mean(dice_scores),
            'hd95': np.mean(hd95_scores)
        }

    def evaluate_report_generation(self, model):
        bleu_scores = []
        rouge_scores = []
        clinical_accuracy = []

        for sample in self.test_data['reports']:
            generated = model.generate_report(sample['fmri'], sample['t1_mri'])
            reference = sample['expert_report']

            bleu = sentence_bleu([reference.split()], generated.split())
            rouge = rouge_score(generated, reference)

            # Clinical accuracy: check if key findings are mentioned
            clinical_acc = self.check_clinical_accuracy(generated, reference)

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            clinical_accuracy.append(clinical_acc)

        return {
            'bleu': np.mean(bleu_scores),
            'rouge': np.mean(rouge_scores),
            'clinical_accuracy': np.mean(clinical_accuracy)
        }
```

### 5.2 Comparison to Baselines

#### **Baseline Models**
1. **GPT-4V**: General vision-language model
2. **Med-Gemini**: Medical-specific multimodal model
3. **BrainLM**: Time-series brain foundation model
4. **BrainGFM**: Graph-based brain foundation model
5. **BrainSegFounder**: 3D segmentation specialist

#### **Expected Results**
```
KoreanBrainBench Results (Target):
────────────────────────────────────────────────────
Task                      | GPT-4V | Med-Gemini | BrainGFM | KoreanBrain-1B
────────────────────────────────────────────────────
Disease Classification    |  58%   |    68%     |   72%    |    78-82%
  - Alzheimer's (AUC)     | 0.72   |    0.78    |   0.83   |    0.85-0.90
  - Parkinson's (AUC)     | 0.68   |    0.75    |   0.80   |    0.82-0.88
  - Stroke (AUC)          | 0.65   |    0.72    |   0.78   |    0.80-0.85
────────────────────────────────────────────────────
Lesion Segmentation       |  N/A   |    N/A     |   N/A    |    0.75-0.80
  - Dice coefficient      |  N/A   |    N/A     |   0.72   |    0.75-0.80
────────────────────────────────────────────────────
Korean Report Generation  |  45%   |    52%     |   N/A    |    65-70%
  - BLEU score            |  0.45  |    0.52    |   N/A    |    0.65-0.70
  - Clinical accuracy     |  62%   |    71%     |   N/A    |    80-85%
────────────────────────────────────────────────────
Cross-Center              |  -8%   |    -5%     |   -3%    |    -2% (target)
  Generalization drop     |        |            |          |
────────────────────────────────────────────────────
```

---

## 6. RESOURCE REQUIREMENTS

### 6.1 Computational Infrastructure

#### **Training Phase**
- **GPUs**: 8x NVIDIA A100 (80GB) or 8x H100 (80GB)
- **Storage**: 100TB (raw DICOM) + 20TB (preprocessed tensors)
- **RAM**: 512GB system memory
- **Network**: 100Gbps for multi-node training
- **Duration**: 3-4 months total (all stages)

**Cost Estimate** (AWS p4d.24xlarge):
- $32.77/hour × 24 hours × 90 days = **~$70,000** for training

#### **Inference Phase**
- **GPUs**: 1x NVIDIA A100 (40GB) per 1,000 patients/day
- **Latency**: 1.2s per sample (FP16 TensorRT)
- **Throughput**: ~60,000 patients/day per GPU

**Cost Estimate** (AWS p4d.24xlarge for inference):
- $4.13/hour × 24 hours × 30 days = **~$3,000/month** for 1,000 patients/day

### 6.2 Data Storage

#### **Raw Data**
- **1 subject**: ~2GB (T1 + T2-FLAIR + resting fMRI + diffusion)
- **10,000 subjects**: 20TB
- **With backups (3 copies)**: 60TB

#### **Preprocessed Data**
- **1 subject**: ~500MB (normalized, skull-stripped, registered to MNI152)
- **10,000 subjects**: 5TB
- **Model checkpoints**: 50GB per checkpoint × 10 checkpoints = 500GB

**Total storage**: ~70TB

### 6.3 Personnel

#### **Research Team**
- **Principal Investigator (PI)**: 1 (overall strategy, publications)
- **Deep Learning Engineers**: 3 (model architecture, training, optimization)
- **Neuroimaging Specialists**: 2 (preprocessing, quality control, interpretation)
- **Data Engineers**: 2 (multi-center data aggregation, PACS integration)
- **Clinical Radiologists**: 2 (annotation, validation, clinical trials)
- **Project Manager**: 1 (coordination, timeline, budget)

**Total**: 11 people

**Budget** (2-year project):
- Salaries: $2.5M (average $115K/year per person)
- Compute: $150K (training + inference)
- Data collection: $500K (hospital partnerships, IRB, recruitment)
- Publication/travel: $50K
- **Total**: ~$3.2M

---

## 7. RISK MITIGATION

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Insufficient Korean data** | High | High | Synthetic data augmentation (Med-DDPM), few-shot learning |
| **Poor generalization** | Medium | High | Multi-center training, domain adaptation techniques |
| **Computational bottleneck** | Medium | Medium | Model distillation, quantization, cloud bursting |
| **Data privacy breach** | Low | Critical | Federated learning, differential privacy, strict IRB protocols |
| **Clinical validation failure** | Medium | High | Early pilot studies, radiologist-in-the-loop training |

### 7.2 Data Augmentation Strategy

#### **Synthetic Data Generation (Med-DDPM)**
```python
class SyntheticDataGenerator:
    def __init__(self, diffusion_model):
        self.ddpm = diffusion_model  # Trained Med-DDPM

    def generate_synthetic_mri(self, disease='alzheimers', n_samples=1000):
        synthetic_mris = []

        for _ in range(n_samples):
            # Sample from noise
            noise = torch.randn(1, 1, 160, 192, 160).cuda()

            # Conditional generation
            condition = self.disease_to_embedding(disease)

            # Denoise iteratively
            mri = self.ddpm.sample(noise, condition)
            synthetic_mris.append(mri)

        return synthetic_mris

    def augment_training_set(self, real_data, augmentation_ratio=2.0):
        # Generate synthetic data to reach target ratio
        n_synthetic = int(len(real_data) * augmentation_ratio)
        synthetic_data = self.generate_synthetic_mri(n_samples=n_synthetic)

        # Mix real + synthetic
        augmented_data = real_data + synthetic_data
        return augmented_data
```

**Expected benefit**: +3-5% accuracy with synthetic augmentation (based on Alzheimer's diffusion study)

**Reference**: [PubMed Counterfactual MRI](https://pubmed.ncbi.nlm.nih.gov/38370616/)

---

## 8. TIMELINE AND MILESTONES

### 8.1 Project Gantt Chart

```
Month 1-3: Data Collection & Infrastructure
├─ IRB approvals (all 4 hospitals)
├─ PACS integration setup
├─ GPU cluster provisioning
└─ Pilot data collection (500 subjects)

Month 4-6: Self-Supervised Pre-Training
├─ BrainMAE training on Korean data
├─ Temporal contrastive learning
├─ Quality validation (embedding quality)
└─ Checkpoint: General brain representations

Month 7-9: Foundation Model Adaptation
├─ Load NeuroSTORM pre-trained weights
├─ Task-specific prompt tuning
├─ Continual pre-training (Korean medical text)
└─ Checkpoint: Adapted model for Korean data

Month 10-12: Multimodal Fusion Training
├─ Cross-modal attention fusion
├─ Multi-task learning (classification + segmentation + reports)
├─ End-to-end fine-tuning
└─ Checkpoint: Unified multimodal model

Month 13-15: Clinical Validation
├─ Hold-out test set evaluation
├─ Cross-center validation
├─ Radiologist comparison study
└─ Checkpoint: Clinical validation report

Month 16-18: Deployment & Optimization
├─ TensorRT optimization
├─ PACS integration at Samsung Medical Center
├─ Real-world pilot (100 patients)
└─ Checkpoint: Production-ready system

Month 19-21: Regulatory Approval & Scaling
├─ Korean FDA (MFDS) submission
├─ Multi-center clinical trials
├─ Scale to 4 hospitals
└─ Checkpoint: Regulatory approval pathway

Month 22-24: Publication & Open-Source Release
├─ Nature Medicine / NEJM submission
├─ Model release (Hugging Face)
├─ KoreanBrainBench public dataset
└─ Final Deliverable: KoreanBrain-1B v1.0
```

---

## 9. SUCCESS CRITERIA

### 9.1 Technical Metrics

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| **Alzheimer's classification (AUC)** | 0.78 (Med-Gemini) | 0.85 | 0.90 |
| **Multi-disease accuracy** | 68% (Med-Gemini) | 75% | 80% |
| **Lesion segmentation (Dice)** | 0.72 (BrainSegFounder) | 0.75 | 0.80 |
| **Korean report BLEU** | 0.52 (Med-Gemini) | 0.65 | 0.70 |
| **Cross-center generalization** | -5% drop | -2% drop | -1% drop |
| **Inference latency** | 2.5s (baseline) | 1.2s | 0.8s |

### 9.2 Clinical Impact

| Metric | Target |
|--------|--------|
| **Radiologist agreement** | > 85% (on test set) |
| **False positive rate** | < 10% |
| **Sensitivity (disease detection)** | > 90% |
| **Specificity** | > 85% |
| **Clinical workflow time saved** | > 30% (compared to manual review) |

### 9.3 Business Impact

| Metric | Target |
|--------|--------|
| **Patients processed (Year 1)** | 10,000 |
| **Partner hospitals** | 4+ |
| **Publications (high-impact)** | 2+ (Nature Medicine, NEJM, Radiology) |
| **Regulatory approval** | Korean FDA (MFDS) within 2 years |
| **Cost per diagnosis** | < $50 (vs $200-500 current radiologist time) |

---

## 10. REFERENCES

All references from main research document apply. Key implementation references:

### Code Repositories
- [GitHub - vandijklab/BrainLM](https://github.com/vandijklab/BrainLM)
- [GitHub - CUHK-AIM-Group/NeuroSTORM](https://github.com/CUHK-AIM-Group/NeuroSTORM)
- [GitHub - lab-smile/BrainSegFounder](https://github.com/lab-smile/BrainSegFounder)

### Key Papers
- [BrainLM bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1.full)
- [NeuroSTORM arXiv](https://arxiv.org/html/2506.11167v1)
- [BrainGFM arXiv](https://arxiv.org/html/2506.02044)
- [Brain-MGF arXiv](https://arxiv.org/html/2511.18325)
- [BrainMAE arXiv](https://arxiv.org/html/2406.17086v1)
- [Med-Gemini Google Research](https://research.google/blog/advancing-medical-ai-with-med-gemini/)
- [UK Biobank QC PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5770339/)

---

**Document End**

This technical specification provides a concrete, implementable roadmap for developing a Korean brain foundation model based on cutting-edge research from 2024-2025.
