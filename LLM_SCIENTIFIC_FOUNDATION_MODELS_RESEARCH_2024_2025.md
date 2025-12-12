# LLM-Based Scientific Foundation Models and Multimodal Integration: Comprehensive Research Analysis (2024-2025)

**Generated**: December 8, 2025
**Purpose**: Rigorous technical research to inform Korean brain foundation model development strategy

---

## Executive Summary

This research compiles recent advances (2024-2025) in LLM-based scientific foundation models, multimodal integration strategies, and their applications to neuroimaging and brain data. The analysis focuses on **concrete implementations, architectures, benchmarks, and limitations** to replace conceptual approaches with evidence-based technical strategies.

**Key Finding**: The field has rapidly matured with multiple specialized brain foundation models (BrainLM, BrainGFM, NeuroSTORM, BrainSegFounder) achieving state-of-the-art performance through:
1. Large-scale pre-training on 6,700-28,650+ hours of fMRI data
2. Transformer-based architectures with domain-specific adaptations
3. Self-supervised learning (MAE, contrastive learning) to overcome data scarcity
4. Multimodal fusion strategies combining structural/functional imaging
5. Few-shot and zero-shot learning for rare disorders

---

## 1. BRAIN FOUNDATION MODELS: STATE-OF-THE-ART (2024-2025)

### 1.1 Major Models and Architectures

#### **BrainLM** (Vandijk Lab, 2024)
- **Scale**: 6,700 hours of fMRI recordings from UK Biobank and HCP datasets
- **Architecture**: Transformer-based masked autoencoder
  - Available in 111M and 650M parameter versions
  - Parcellated embeddings → masked → reconstructed via Transformer autoencoder
  - Captures spatiotemporal dynamics of large-scale brain activity
- **Capabilities**:
  - Fine-tuning: Predicts clinical variables and future brain states
  - Zero-shot: Identifies functional networks, generates interpretable latent representations, supports perturbation simulation
- **Code**: Available on Hugging Face (vandijklab/brainlm) and GitHub
- **Reference**: [BrainLM bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1.full)

#### **NeuroSTORM** (CUHK AIM Group, 2025)
- **Scale**: 28.65 million fMRI frames (>9,000 hours) from 50,000+ subjects
  - Multiple centers, ages 5-100
- **Architecture**: Neuroimaging Foundation Model with Spatial-Temporal Optimized Representation Modeling
  - **Shifted-Window Mamba (SWM)**: Linear-time state-space modeling + shifted-window mechanisms
  - Reduces complexity and GPU memory usage
  - **Spatiotemporal Redundancy Dropout (STRD)**: Learns inherent fMRI characteristics, improves robustness
  - **Task-specific Prompt Tuning (TPT)**: For downstream adaptation
- **Benchmarks**: Comprehensive fMRI benchmark with 5 tasks
  1. Age and Gender Prediction
  2. Phenotype Prediction
  3. Disease Diagnosis
  4. fMRI Retrieval
  5. Task fMRI State Classification
- **Code**: [GitHub - CUHK-AIM-Group/NeuroSTORM](https://github.com/CUHK-AIM-Group/NeuroSTORM)
- **Reference**: [arXiv - Towards general-purpose foundation model for fMRI](https://arxiv.org/html/2506.11167v1)

#### **BrainGFM** (Brain Graph Foundation Model, 2024)
- **Scale**: 27 neuroimaging datasets, 25 neurological/psychiatric disorders
  - 2 atlas types (functional/anatomical) across 8 parcellations
  - 25,000+ subjects, 60,000+ fMRI scans, 400,000+ graph samples
- **Architecture**: Graph Transformer with domain-specific enhancements
  - **Pretext Tasks**: Graph Contrastive Learning (GCL) + Graph Masked Autoencoders (GMAE)
  - **Graph Prompts + Language Prompts**: Integrated into model design
  - **Structural Encodings**: fMRI-specific enhancements
- **Performance**: Surpasses connectome/FC-based models (BrainMass, BrainNPT) and matches/exceeds time-series models (BrainLM)
- **Reference**: [arXiv - Brain Graph Foundation Model](https://arxiv.org/html/2506.02044)

#### **BrainSegFounder** (2024)
- **Scale**: 41,400 participants with multimodal brain MRI
- **Architecture**: 3D Vision Transformer with novel two-stage pretraining
  - 3D convolutional networks for comprehensive neuroimage segmentation
  - Self-supervised training on unlabeled multimodal MRI
- **Benchmarks**:
  - BraTS (Brain Tumor Segmentation) challenge
  - ATLAS v2.0 (Anatomical Tracings of Lesions After Stroke)
  - **Performance**: Surpasses previous winning solutions using fully supervised learning
- **Code**: [GitHub - lab-smile/BrainSegFounder](https://github.com/lab-smile/BrainSegFounder)
- **Reference**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002263)

#### **MindFormer** (2024)
- **Purpose**: Multi-subject brain decoding via fMRI
- **Architecture**: Transformer for fMRI-conditioned feature generation
  - Novel training strategy based on IP-Adapter
  - Extracts semantically meaningful features from fMRI signals
  - **Subject-specific token + linear layer**: Captures individual fMRI signal differences
  - Learnable subject token identifies data origin
- **Application**: Generates feature vectors for conditioning Stable Diffusion model
- **Reference**: [arXiv - MindFormer](https://arxiv.org/html/2405.17720v1)

### 1.2 Comparative Analysis

| Model | Data Scale | Architecture Type | Key Innovation | Best For |
|-------|-----------|------------------|----------------|----------|
| **BrainLM** | 6,700 hrs | Time-series Transformer MAE | Zero-shot inference | General fMRI analysis, future state prediction |
| **NeuroSTORM** | 9,000+ hrs, 50K subjects | Shifted-Window Mamba | Linear-time state-space + STRD | Multi-task benchmarks, large-scale |
| **BrainGFM** | 400K graph samples | Graph Transformer | Graph prompts + GCL/GMAE | Connectome analysis, disorder classification |
| **BrainSegFounder** | 41,400 subjects | 3D ViT | Two-stage self-supervised | Segmentation tasks (tumors, lesions) |
| **MindFormer** | HCP dataset | IP-Adapter Transformer | Subject-specific tokens | Brain decoding, generative models |

**Performance Hierarchy**:
- **Graph-based (BrainGFM)** > Time-series (BrainLM) for disorder classification
- **3D ViT (BrainSegFounder)** achieves SOTA on segmentation benchmarks
- **NeuroSTORM** shows best multi-task versatility across 5 benchmark categories

---

## 2. GENERAL-PURPOSE LLMs IN MEDICAL NEUROIMAGING

### 2.1 GPT-4V, Gemini, and Med-Gemini Performance

#### **Med-Gemini** (Google, 2024-2025)
- **Performance Benchmarks**:
  - Surpasses GPT-4V by **44.5% average relative margin** on 7 multimodal medical benchmarks
  - NEJM Image Challenges: State-of-the-art
  - Multimodal USMLE-style questions: SOTA
- **Brain MRI Accuracy**:
  - Multiple Sclerosis: 60%
  - Brain tumors: 56%
  - Brain hemorrhage: 54%
  - Ischemic stroke: 46%
- **Clinical Impact**: AI assistance increased radiologist accuracy from 47.2% to 56.0%
- **Reference**: [Google Research - Med-Gemini](https://research.google/blog/advancing-medical-ai-with-med-gemini/)

#### **GPT-4V, Grok, Gemini Comparative Study** (2024)
- **Neuroradiology Performance**:
  - **Images alone**: Radiologists (42.0%) >> GPT-4o (3.8%), Gemini 1.5 Pro (7.5%)
  - **Images + text**: Radiologists (48.0%) > Gemini (38.7%) > GPT-4o (34.0%)
  - **Text alone**: Gemini (44.7%) > GPT-4o (34.0%) > Radiologists (16.4%)
- **Key Insight**: Multimodal models excel at clinical context integration but struggle with pure visual diagnosis
- **Limitations**: Frequent hallucinations, difficulty integrating multimodal inputs effectively
- **Gold Standard**: CNN/ViT models achieve 98-99% accuracy for MRI pathology prediction
- **Reference**: [MDPI Diagnostics](https://www.mdpi.com/2075-4418/15/11/1320)

#### **Medical VLM-24B** (John Snow Labs, 2024)
- **Scale**: 24 billion parameters
- **Performance**: 82.9% average across medical specialties
- **Architecture**: Vision-language model optimized for English medical text
- **Clinical Focus**: Unprecedented accuracy and reliability in medical contexts
- **Reference**: [John Snow Labs Medical VLM](https://www.johnsnowlabs.com/introducing-medical-vlm-24b-our-first-medical-vision-language-model/)

### 2.2 Vision-Language Models for Medical Imaging (2024)

#### **Key Models and Achievements**
1. **Med-VLFM (Pclmed)**: Winner of ImageCLEFmedical 2024 Caption Prediction Challenge
2. **LViT**: Vision Transformer + language guidance for precise, context-aware segmentation
3. **3D Vision-Language Models**: Integration of 3D ViT, 3D Perceiver, 3D ResNet with Gemini/Llama/Phi-3

#### **Technical Architecture**
- **Multimodal Fusion**: Transformer-based architectures for image-text integration
- **3D Imaging**:
  - 3D CNNs and Transformer networks extract volumetric features
  - Text encoders process data tokens
  - Vision-Language Foundation Models leverage computer vision + LLMs
- **Applications**: Cross-modal retrieval, disease diagnosis, automated report generation
- **Reference**: [Nature npj AI](https://www.nature.com/articles/s44387-025-00015-9)

#### **Growth Metrics**
- Literature count increased **exponentially** from 2019-2024
- 2024 saw rapid acceleration in VLM + medical imaging publications
- **Challenges**: Limited data, privacy concerns, lack of standardized evaluation metrics

---

## 3. CROSS-MODAL ATTENTION MECHANISMS

### 3.1 Brain-MGF: EEG-fMRI Multimodal Fusion (2024)

#### **Architecture**
- **Graph Construction**:
  - Partial-correlation edges
  - Pearson-profile node features
  - Separate graphs for each modality
- **Fusion Strategy**:
  - Adaptive softmax gate with sample-specific weights
  - Captures context-dependent contributions
  - Three GraphConv blocks with LeakyReLU + dropout
  - Global mean pooling → subject-level embeddings

#### **Performance**
- **Multimodal fusion**: 74.02% accuracy
- **fMRI alone**: 68.84%
- **EEG alone**: 59.64%
- **Rest conditions**: 76.00% accuracy, 85.83% ROC-AUC
- **Reference**: [arXiv - Brain-MGF](https://arxiv.org/html/2511.18325)

### 3.2 Other Cross-Modal Architectures

#### **CMAF-Net** (Cross-Modal Attention Fusion, 2024)
- **Application**: Incomplete multi-modal brain tumor segmentation
- **Architecture**: Deep neural network with cross-modal attention
- **Key Innovation**: Handles missing modalities through attention-based feature correlation
- **Reference**: [PMC CMAF-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC11250309/)

#### **MCSP** (Multi-modal Cross-domain Self-supervised Pre-training)
- **Innovation**: Cross-modal self-supervised loss for fMRI-EEG fusion
- **Approach**: Knowledge distillation within domains + cross-modal feature convergence
- **Reference**: [ScienceDirect MCSP](https://www.sciencedirect.com/science/article/abs/pii/S089360802400995X)

#### **MultiViT for Alzheimer's Disease**
- **Architecture**: Cross-attention mechanisms fuse ViT-encoded features
  - Gray matter volume + functional network connectivity
- **Performance**: AUC of 0.833
- **Reference**: [IEEE MultiViT](https://ieeexplore.ieee.org/iel7/10385250/10385251/10385864.pdf)

### 3.3 Complementary Information Integration

**Key Principle**: fMRI + EEG provide complementary views
- **fMRI**: Large-scale haemodynamic coupling (seconds), higher spatial resolution
- **EEG**: Millisecond-scale electrophysiological coherence, fast neuronal synchrony
- **Outcome**: Comprehensive characterization linking haemodynamic + electrophysiological patterns

---

## 4. SELF-SUPERVISED LEARNING FOR BRAIN IMAGING

### 4.1 BrainMAE Framework (2024)

#### **Architecture**
- **Brain Masked Auto-Encoder** for fMRI time-series
- **Components**:
  1. **Region-aware graph attention**: Captures relationships between brain ROIs
  2. **Masked autoencoding framework**: Pre-training strategy
- **Advantages**:
  - Captures rich temporal dynamics
  - Maintains resilience to fMRI noise
- **Performance**: Outperforms baselines by significant margins across 4 downstream tasks
- **Reference**: [arXiv BrainMAE](https://arxiv.org/abs/2406.17086)

### 4.2 Graph Contrastive Learning with Diffusion Augmentation (GCDA, 2024)

#### **Innovation**
- **Graph diffusion augmentation** preserves BOLD signal integrity
- **Application**: fMRI analysis + brain disease detection
- **Purpose**: Alleviates small-sample-size problem
- **Tasks**: Orthostatic hypotension, Alzheimer's disease detection
- **Reference**: [PMC GCDA](https://pmc.ncbi.nlm.nih.gov/articles/PMC11875923/)

### 4.3 Temporal Contrastive Learning

#### **Strategy**
- **Positive pairs**: End-middle (neighboring) fMRI segments
- **Negative pairs**: Beginning-end (distant) segments
- **Learning objective**: Internal spatiotemporal patterns within fMRI

#### **Performance**
- **With only 12 subjects**: 69.7 ± 4.4% classification accuracy
- **Tasks**: Motor and Relational classification
- **Advantage**: Pre-trained models >> randomly initialized training
- **Reference**: [PMC SSL Brain Disorders](https://pmc.ncbi.nlm.nih.gov/articles/PMC12324563/)

### 4.4 Summary of SSL Approaches

| Method | Pretext Task | Key Innovation | Best Application |
|--------|-------------|----------------|------------------|
| **BrainMAE** | Masked autoencoding | Region-aware graph attention | fMRI time-series analysis |
| **GCDA** | Graph contrastive learning | Diffusion augmentation | Small-sample disease detection |
| **CGL** | Functional connectivity graphs | Subject-specific patterns | Cross-subject transfer |
| **Temporal CL** | Temporal segment pairing | Spatiotemporal patterns | Limited training data |

---

## 5. FEW-SHOT AND ZERO-SHOT LEARNING

### 5.1 Few-Shot Learning Approaches

#### **Deep Triplet Networks for Brain Imaging**
- **Datasets**: 7 MRI sequences (T1, T2, post-contrast T1, T2-FLAIR, PD, PASL, MRA) + CT + FDG-PET
- **Approach**: Extracts relevant imaging features from limited training examples
- **Source**: Public datasets with healthy + diseased individuals
- **Reference**: [Springer Few-Shot Learning](https://link.springer.com/chapter/10.1007/978-3-030-33391-1_21)

#### **Expert-Guided Few-Shot Learning (2024)**
- **Innovation**: Integrates radiologist spatial annotations into training
- **Evaluation**: BraTS (brain tumor MRI) + chest X-ray datasets
- **Performance**: Consistent improvements in accuracy + visual interpretability
- **Reference**: [arXiv Expert-Guided FSL](https://arxiv.org/html/2509.08007)

#### **GAN-Based Domain Adaptation**
- **Applications**:
  - Unbalanced label handling
  - Synthetic data augmentation
  - Cross-center domain adaptation
- **Reference**: [Wiley NMR Biomedicine](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/nbm.5143)

### 5.2 Zero-Shot Learning for EEG-BCI

#### **META-EEG Framework (2024)**
- **Architecture**: Gradient-based meta-learning + intermittent freezing
- **Purpose**: Zero-calibration solution for inter-subject variability
- **Validation**: Leave-one-subject-out cross-validation (LOOCV)
- **Performance**: Significantly outperforms baselines on multiple public datasets
- **Reference**: [PMC Few-Shot EEG](https://pmc.ncbi.nlm.nih.gov/articles/PMC11266297/)

#### **Zero-Shot EEG-to-Image Brain Decoding**
- **Approach**:
  - State-of-the-art EEG preprocessing + feature selection
  - Maps EEG → biologically inspired computer vision + linguistic models
- **Application**: Real-world image retrieval
- **Advantage**: More applicable than traditional classification
- **Reference**: [PMC Zero-Shot BCI](https://ncbi.nlm.nih.gov/pmc/articles/PMC6746355)

### 5.3 Domain Generalization

#### **Model-Agnostic Meta-Learning (MAML)**
- **Goal**: Learn domain-agnostic feature representations
- **Outcome**: Improved generalization to unseen test distributions
- **Reference**: [PMC Domain Adaptation Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/)

---

## 6. DIFFUSION MODELS FOR BRAIN MRI GENERATION

### 6.1 Med-DDPM (2024)

#### **Architecture**
- **3D semantic brain MRI synthesis**
- **Integration**: Semantic conditioning for data scarcity + privacy
- **Stability**: Superior to existing 3D brain imaging synthesis methods

#### **Performance**
- **Dice score**: 0.6207 (synthetic) vs 0.6531 (real images)
- **Quality**: Diverse, anatomically coherent, high visual fidelity
- **Reference**: [PubMed Med-DDPM](https://pubmed.ncbi.nlm.nih.gov/38578863/)

### 6.2 Conditional DPM (cDPM, 2024)

#### **Innovation**
- **Memory-efficient process**: Generates realistic brain MRIs from random noise
- **Conditional scheme**: Progressive slice generation based on previous slices
- **Advantage**: Limited computational resources + training data
- **Reference**: [PMC Conditional DPM](https://pmc.ncbi.nlm.nih.gov/articles/PMC10758344/)

### 6.3 Cancer and Alzheimer's Applications (2024)

#### **Glioma, Meningioma, Pituitary Tumor Synthesis**
- **Approach**: Fine-tuned models on clinically curated datasets
- **Method**: Text-to-image synthesis for brain cancer MRIs
- **Reference**: [PMC Cancer Diffusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC11387006/)

#### **Counterfactual MRI for Alzheimer's Detection**
- **Innovation**: Disease-conditioned diffusion models
- **Synthetic data utility**: Trained AD classifier with only 500 real scans
- **Performance boost**: +3% with synthetic augmentation
- **Reference**: [PubMed Counterfactual MRI](https://pubmed.ncbi.nlm.nih.gov/38370616/)

### 6.4 3D Multi-Modal Translation (2024)

#### **Capabilities**
- **Cross-modality synthesis**: Various source-target scenarios
- **Innovation**: One-to-many modality translations
- **Evaluation**: Surpasses other models on multi-modal brain MRI datasets (4 modalities)
- **Reference**: [WACV 2024 Adaptive Latent Diffusion](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Adaptive_Latent_Diffusion_Model_for_3D_Medical_Image_to_Image_WACV_2024_paper.html)

---

## 7. DOMAIN ADAPTATION FOR MEDICAL LLMs

### 7.1 Architectural Insights (2024-2025)

#### **BERT vs GPT for Medical Applications**
- **GPT-based models**: Better for communicative tasks (report generation, patient interaction)
  - Unidirectional language processing
- **BERT-based models**: Better for innovative applications (classification, knowledge discovery)
  - Bidirectional text understanding
  - More straightforward domain-specific extensions
- **Reference**: [JMIR LLM Architectures](https://www.jmir.org/2025/1/e70315)

### 7.2 Continual Pre-Training Strategies

#### **Me-LLaMA (Medical LLaMA Adaptation)**
- **Approach**: LLaMA2 continual pre-training on mixed medical corpus
  - Biomedical literature + clinical notes + general domain data
- **Performance (Me-LLaMA 70B)**:
  - **Gains**: 2.1% to 55% across datasets
  - **Best results**: Continual pre-training + instruction tuning together
- **Reference**: [Nature npj Digital Medicine](https://www.nature.com/articles/s41746-025-01533-1)

### 7.3 3DS: Decomposed Difficulty-based Data Selection (2024)

#### **Innovation**
- **Novel methodology**: Difficulty-based sample selection for medical domain adaptation
- **Dataset**: Carefully curated Chinese medical dataset
  - Medical dialogues + domain-specific instructions
- **Open-source**: Available for healthcare-oriented LLM research
- **Reference**: [arXiv 3DS](https://arxiv.org/html/2410.10901)

### 7.4 Research Trends (2024-2025)

#### **Publication Statistics**
- **2024**: 557 LLM articles
- **Early 2025**: 27 articles (marked growth trajectory)
- **Domain distribution**: 93.55% general-domain LLMs, 6.45% medical-domain LLMs
- **Multimodal trend**: Significant innovations in multi-modal medical LLM development
- **Reference**: [BMC Medical Informatics](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-025-02954-4)

---

## 8. PROMPT TUNING FOR VISION TRANSFORMERS

### 8.1 Dynamic Visual Prompt Tuning (DVPT, 2025)

#### **Architecture**
- **Learnable parameters**: <1% of model (prompts in input space)
- **Deep prompt fine-tuning**: Prompts at each transformer layer level
- **Performance**: SOTA parameter-efficient fine-tuning
- **Applications**: 2D and 3D medical image classification + segmentation
- **Reference**: [ScienceDirect DVPT](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000474)

### 8.2 Brain MRI Applications

#### **Fine-Tuned Vision Transformer (FT-ViT)**
- **Dataset**: 5,712 brain tumor images (CE-MRI)
- **Performance**: 98.13% accuracy
- **Tasks**: Glioma, meningioma, pituitary, no tumor classification
- **Reference**: [MDPI Brain Tumor Detection](https://www.mdpi.com/2075-4418/13/12/2094)

#### **Hierarchical Multi-Scale Attention (HMSA)**
- **Innovation**: Multi-resolution patch embedding
- **Performance**: 35% reduction in training duration
- **Feature extraction**: Different spatial scales
- **Reference**: [Nature Scientific Reports](https://www.nature.com/articles/s41598-025-23100-0)

#### **Alzheimer's Disease Classification**
- **Performance boost**: +5% (sex classification), +9-10% (AD classification)
- **AUC**: 0.987 (sex), 0.892 (AD)
- **Data efficiency**: Superior with only 100 MRI training images
- **Reference**: [PubMed ViT AD Detection](https://pubmed.ncbi.nlm.nih.gov/38083552/)

### 8.3 Multi-Center Medical Image Segmentation

#### **Prompt-Based Tuning for Head/Neck Cancer**
- **Application**: Multi-center segmentation with domain shift
- **Advantage**: Adapts to different acquisition protocols + centers
- **Reference**: [MDPI Prompt-Based Tuning](https://www.mdpi.com/2306-5354/10/7/879)

---

## 9. BENCHMARK DATASETS AND QUALITY METRICS

### 9.1 UK Biobank

#### **Scale**
- **Target**: 100,000 participants with brain, heart, body MRI
- **Current**: 10,000+ subjects with processed data released
- **Modalities**: 6 brain imaging types
  1. T1-weighted MRI
  2. T2-FLAIR
  3. Susceptibility-weighted MRI
  4. Resting fMRI
  5. Task fMRI
  6. Diffusion MRI

#### **Quality Control**
- **Challenge**: 100,000 subjects makes visual inspection unfeasible
- **Solution**: Automated processing + QC pipeline (FSL-based)
- **Key metrics**: SNR, CNR, CJV, FBER, IQR
- **Reference**: [PMC UK Biobank](https://pmc.ncbi.nlm.nih.gov/articles/PMC5770339/)

### 9.2 Human Connectome Project (HCP)

#### **Scale**: 1,200 subjects
- **QC approach**: Visual inspection (feasible at this scale)
- **HCP pipelines**: Minimal preprocessing optimized for project needs
- **Comparison to UK Biobank**: 10x smaller, but very high quality

#### **Population Analysis**
- **HCP**: Single mode of covariation (461 young healthy adults)
- **UK Biobank**: Multiple modes identified
  - 10x more subjects
  - Larger age range
  - More imaging modalities + non-imaging variables
- **Reference**: [ScienceDirect UK Biobank QC](https://www.sciencedirect.com/science/article/pii/S1053811917308613)

### 9.3 Quality Control Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **SNR** | Signal-to-Noise Ratio | Scanner hardware performance |
| **CNR** | Contrast-to-Noise Ratio | Tissue differentiation quality |
| **CJV** | Coefficient of Joint Variation | Image uniformity |
| **FBER** | Foreground-Background Energy Ratio | Artifact detection |
| **IQR** | Image Quality Rate | Overall quality assessment |

### 9.4 GenMIND Dataset (2024)

#### **Innovation**
- **18,000 synthetic neuroimaging samples**
- **Population**: Diverse global healthy adult representation
- **Purpose**: Address data scarcity through generative models
- **Reference**: [Nature Scientific Data](https://www.nature.com/articles/s41597-024-04157-4)

---

## 10. KOREAN BRAIN IMAGING CONTEXT

### 10.1 Transfer Learning Study

#### **ABCD → Korean Adolescent Transfer**
- **Source**: ABCD Study (US adolescents)
- **Target**: Seoul National University Hospital (SNUH)
  - 147 Korean adolescents (54 males, age 14.6 ± 1.5 years)
- **Model**: Deep neural network trained on fMRI
- **Task**: General psychopathology prediction
- **Funding**: National Research Foundation (NRF), Korean Government
- **Reference**: [ScienceDirect Korean Transfer Learning](https://www.sciencedirect.com/science/article/abs/pii/S2451902225001338)

### 10.2 Gap Analysis

#### **Current State**
- **Limited Korean-specific brain foundation models**: No dedicated large-scale model found
- **Transfer learning approach**: Primary strategy (US → Korean adaptation)
- **Dataset availability**: Small-scale (147 subjects at SNUH)

#### **Opportunities**
1. **Scale up Korean brain imaging datasets**: Follow UK Biobank model (10,000-100,000 subjects)
2. **Multi-center collaboration**: Aggregate data across Korean hospitals
3. **Domain adaptation**: Fine-tune existing models (BrainLM, NeuroSTORM) on Korean data
4. **Cultural/genetic specificity**: Develop Korean-optimized preprocessing + normalization

---

## 11. TECHNICAL IMPLEMENTATION ROADMAP

### 11.1 Recommended Architecture Stack

#### **Foundation Model**
- **Base**: NeuroSTORM or BrainLM (open-source, proven performance)
- **Adaptation**: Korean-specific fine-tuning with prompt tuning
- **Rationale**: Leverage 9,000+ hours pre-training, add Korean data via TPT

#### **Multimodal Integration**
- **Strategy**: Brain-MGF-style adaptive gating for fMRI-EEG fusion
- **Components**:
  - Graph Transformer for structural connectivity
  - Temporal Transformer for functional dynamics
  - Cross-modal attention with softmax gating

#### **Self-Supervised Pre-Training**
- **Approach**: BrainMAE framework
  - Masked autoencoding on Korean fMRI data
  - Region-aware graph attention for ROI relationships
- **Advantage**: Maximizes limited labeled data

#### **Domain Adaptation**
- **Strategy**: Me-LLaMA approach
  - Continual pre-training on Korean medical literature + clinical notes
  - Instruction tuning for Korean language tasks
- **Expected gain**: 2-55% performance improvement

#### **Few-Shot Learning**
- **For rare disorders**: Expert-guided few-shot learning
  - Integrate Korean radiologist annotations
  - Deep triplet networks for limited training examples

### 11.2 Data Collection Strategy

#### **Phase 1: Multi-Center Aggregation**
- **Target**: 5,000-10,000 Korean subjects
- **Modalities**: T1, T2-FLAIR, resting fMRI, diffusion MRI
- **Partners**: University hospitals (Seoul National, Yonsei, Samsung Medical Center)

#### **Phase 2: Quality Control**
- **Automated QC**: UK Biobank pipeline adaptation
- **Metrics**: SNR, CNR, CJV, FBER, IQR
- **Manual review**: Subset validation (5-10%)

#### **Phase 3: Annotation**
- **Clinical labels**: Disease diagnosis, severity scores
- **Expert segmentations**: Tumors, lesions (for BrainSegFounder-style model)
- **Demographic data**: Age, sex, genetics (if available)

### 11.3 Model Training Pipeline

#### **Stage 1: Self-Supervised Pre-Training**
```
1. Masked Autoencoding (BrainMAE) on unlabeled Korean fMRI
2. Graph Contrastive Learning (GCDA) on functional connectivity
3. Temporal Contrastive Learning on time-series
Duration: 4-6 weeks on multi-GPU cluster
```

#### **Stage 2: Foundation Model Adaptation**
```
1. Load NeuroSTORM or BrainLM weights
2. Task-specific Prompt Tuning on Korean data
3. Continual pre-training with Korean medical corpus
Duration: 2-4 weeks
```

#### **Stage 3: Multimodal Fusion Training**
```
1. Train modality-specific encoders (fMRI, structural MRI, EEG)
2. Cross-modal attention fusion (Brain-MGF architecture)
3. End-to-end fine-tuning on Korean clinical tasks
Duration: 2-3 weeks
```

#### **Stage 4: Evaluation and Validation**
```
1. Hold-out test set (20% of Korean data)
2. Cross-validation across centers
3. Clinical validation with Korean radiologists
4. Benchmark against GPT-4V, Med-Gemini on Korean cases
Duration: 2-4 weeks
```

### 11.4 Performance Targets

| Task | Baseline (General Models) | Target (Korean Model) |
|------|--------------------------|----------------------|
| **Brain tumor classification** | Med-Gemini 56% | 75-80% |
| **Alzheimer's disease detection** | MultiViT 0.833 AUC | 0.85-0.90 AUC |
| **Stroke lesion segmentation** | BrainSegFounder SOTA | Match or exceed |
| **Multi-disorder classification** | BrainGFM SOTA | +5-10% on Korean data |
| **Report generation (Korean)** | GPT-4V baseline | +20-30% accuracy |

---

## 12. LIMITATIONS AND CHALLENGES

### 12.1 Current Model Limitations

#### **General-Purpose LLMs (GPT-4V, Gemini)**
- **Hallucinations**: Frequent in medical imaging interpretation
- **Visual diagnosis**: Significantly worse than radiologists (3.8-7.5% vs 42%)
- **Multimodal integration**: Struggles to effectively combine image + text
- **Reference**: [MDPI GPT-4V Evaluation](https://www.mdpi.com/2075-4418/15/11/1320)

#### **Brain Foundation Models**
- **Data scarcity**: Even with 100,000 subjects (UK Biobank), still limited for rare disorders
- **Generalization**: Cross-population transfer (US → Korea) requires validation
- **Modality gaps**: LSTMs/Transformers optimized for EEG/MEG, less explored for volumetric fMRI

### 12.2 Technical Challenges

#### **Data Privacy**
- **Korean regulations**: Strict medical data sharing restrictions
- **Solution**: Federated learning, differential privacy, synthetic data (diffusion models)

#### **Computational Resources**
- **Pre-training cost**: BrainLM (650M params) requires multi-GPU cluster, weeks of training
- **Solution**: Start with smaller models (111M), leverage pre-trained weights

#### **Evaluation Metrics**
- **Lack of standardization**: Vision-language models in medical imaging lack consistent benchmarks
- **Solution**: Establish Korean-specific evaluation suite (KoreanBrainBench)

### 12.3 Clinical Translation

#### **Regulatory Approval**
- **Korean FDA (MFDS)**: Rigorous approval process for AI medical devices
- **Evidence requirements**: Large-scale clinical trials

#### **Clinical Workflow Integration**
- **Radiologist adoption**: Requires interpretability, trust
- **Solution**: Explainable AI (attention maps, counterfactual explanations)

---

## 13. RESEARCH GAPS AND OPPORTUNITIES

### 13.1 Identified Gaps

1. **Korean-specific brain atlases**: Limited parcellation schemes for Korean populations
2. **Genetic-imaging integration**: GWAS + neuroimaging for Korean genotypes
3. **Longitudinal Korean data**: Aging trajectories, disease progression
4. **Pediatric Korean brain development**: Critical gap vs Western datasets
5. **Multi-modal Korean benchmarks**: No comprehensive evaluation suite

### 13.2 Strategic Opportunities

#### **Leverage Samsung's Strengths**
1. **Semiconductor expertise**: Custom AI accelerators for brain imaging workloads
2. **Electronics integration**: Portable EEG/fMRI devices for large-scale screening
3. **Data infrastructure**: Secure multi-center federated learning platform

#### **International Collaboration**
1. **UK Biobank partnership**: Access to processing pipelines, QC methods
2. **HCP collaboration**: Methodology sharing, cross-validation
3. **BrainLM/NeuroSTORM teams**: Joint development on Korean adaptation

#### **Differentiation Strategy**
1. **Asian population focus**: First large-scale Asian brain foundation model
2. **Multimodal integration**: Combine fMRI, EEG, genetics (Samsung BioLogics connection)
3. **Clinical deployment**: Real-world hospital integration (Samsung Medical Center)

---

## 14. CONCRETE NEXT STEPS

### 14.1 Immediate Actions (1-3 months)

1. **Literature deep-dive**: Download and analyze papers for:
   - BrainLM, NeuroSTORM, BrainGFM (architectures, code)
   - Brain-MGF (multimodal fusion details)
   - Med-DDPM (synthetic data generation)
   - UK Biobank QC pipelines

2. **Code repository setup**:
   - Clone [GitHub - vandijklab/BrainLM](https://github.com/vandijklab/BrainLM)
   - Clone [GitHub - CUHK-AIM-Group/NeuroSTORM](https://github.com/CUHK-AIM-Group/NeuroSTORM)
   - Clone [GitHub - lab-smile/BrainSegFounder](https://github.com/lab-smile/BrainSegFounder)
   - Evaluate pre-trained models on sample data

3. **Korean data feasibility study**:
   - Contact Seoul National University Hospital, Samsung Medical Center
   - Assess available datasets (scale, modalities, annotations)
   - Establish IRB protocols, data sharing agreements

4. **Computational infrastructure**:
   - Provision multi-GPU cluster (8x A100 minimum)
   - Set up MLOps pipeline (DVC, MLflow, Weights & Biases)

### 14.2 Short-Term Milestones (3-6 months)

1. **Pilot study**: Transfer learning experiment
   - Fine-tune BrainLM on 500-1,000 Korean fMRI scans
   - Evaluate on Alzheimer's detection, tumor classification
   - Compare to GPT-4V, Med-Gemini baselines

2. **Synthetic data generation**:
   - Train Med-DDPM on Korean MRI subset
   - Generate 5,000 synthetic Korean brain MRIs
   - Validate anatomical coherence with radiologists

3. **Multimodal fusion prototype**:
   - Implement Brain-MGF architecture
   - Train on paired fMRI-EEG data (if available)
   - Benchmark against single-modality baselines

### 14.3 Long-Term Vision (1-2 years)

1. **KoreanBrain-1B**: 1 billion parameter foundation model
   - Pre-trained on 10,000+ Korean subjects
   - Multimodal (fMRI, structural MRI, EEG, genetics)
   - Korean language report generation

2. **Clinical validation**: Multi-center trials
   - 5+ Korean hospitals
   - 3+ disease categories (Alzheimer's, stroke, tumors)
   - Regulatory approval pathway (MFDS)

3. **Open-source release**:
   - Model weights on Hugging Face
   - Training code on GitHub
   - Korean brain imaging benchmark dataset (privacy-preserved)

---

## 15. CONCLUSION

### Key Takeaways

1. **State-of-the-art exists**: BrainLM, NeuroSTORM, BrainGFM, BrainSegFounder provide proven architectures
2. **Multimodal fusion is critical**: Brain-MGF shows 74% accuracy vs 59-69% single-modality
3. **Self-supervised learning overcomes data scarcity**: BrainMAE, GCDA enable learning from limited labels
4. **Domain adaptation strategies are mature**: Me-LLaMA, 3DS provide roadmap for medical LLM adaptation
5. **Korean-specific models are feasible**: Transfer learning study shows US→Korea generalization

### Competitive Positioning

**Current landscape**:
- US/UK dominate with massive datasets (UK Biobank 100K, HCP 1.2K)
- General LLMs (GPT-4V, Med-Gemini) struggle with neuroimaging (3.8-56% accuracy)
- Specialized brain models (BrainLM, NeuroSTORM) achieve SOTA but lack Asian data

**Samsung opportunity**:
- First large-scale Asian brain foundation model (10K+ Korean subjects)
- Multimodal integration (fMRI + EEG + genetics) leveraging Samsung ecosystem
- Clinical deployment at scale (Samsung Medical Center + partner hospitals)
- Semiconductor-optimized inference (custom AI chips)

**Differentiation**:
- Korean language medical report generation (vs English-only competitors)
- Genetic-imaging integration for Asian populations
- Real-world clinical validation and regulatory approval

### Risk Mitigation

1. **Data scarcity**: Augment with diffusion models (Med-DDPM), few-shot learning
2. **Computational cost**: Start small (BrainLM 111M), scale gradually
3. **Regulatory hurdles**: Parallel clinical validation during development
4. **Competition**: Focus on Korean/Asian differentiation, clinical deployment speed

---

## SOURCES

### Brain Foundation Models
- [BrainLM: A foundation model for brain activity recordings (bioRxiv)](https://www.biorxiv.org/content/10.1101/2023.09.12.557460v1.full)
- [BrainLM Hugging Face](https://huggingface.co/vandijklab/brainlm)
- [GitHub - vandijklab/BrainLM](https://github.com/vandijklab/BrainLM)
- [Towards a general-purpose foundation model for fMRI analysis (arXiv)](https://arxiv.org/html/2506.11167v1)
- [GitHub - CUHK-AIM-Group/NeuroSTORM](https://github.com/CUHK-AIM-Group/NeuroSTORM)
- [A Brain Graph Foundation Model (arXiv)](https://arxiv.org/html/2506.02044)
- [BrainSegFounder (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002263)
- [BrainSegFounder (PubMed)](https://pubmed.ncbi.nlm.nih.gov/39146701/)
- [GitHub - lab-smile/BrainSegFounder](https://github.com/lab-smile/BrainSegFounder)
- [MindFormer (arXiv)](https://arxiv.org/html/2405.17720v1)
- [Brain Foundation Models Survey (arXiv)](https://arxiv.org/html/2503.00580v1)
- [Foundation and Large-Scale AI Models in Neuroscience (arXiv)](https://www.arxiv.org/pdf/2510.16658)
- [General-Purpose Brain Foundation Models (OpenReview)](https://openreview.net/attachment?id=HwDQH0r37I&name=pdf)

### General-Purpose Medical LLMs
- [Do LLMs Have 'the Eye' for MRI? (MDPI Diagnostics)](https://www.mdpi.com/2075-4418/15/11/1320)
- [Advancing Multimodal Medical Capabilities of Gemini (arXiv)](https://arxiv.org/html/2405.03162v1)
- [Advancing medical AI with Med-Gemini (Google Research)](https://research.google/blog/advancing-medical-ai-with-med-gemini/)
- [Exploring Med-Gemini (Medium)](https://medium.com/@vinodkumargr/exploring-med-gemini-a-breakthrough-in-medical-imaging-ai-datasets-used-for-training-and-28e8bcaf8949)
- [Capabilities of Gemini Models in Medicine (arXiv)](https://arxiv.org/html/2404.18416v2)
- [Medical VLM-24B (John Snow Labs)](https://www.johnsnowlabs.com/introducing-medical-vlm-24b-our-first-medical-vision-language-model/)
- [Multimodal generative AI for 3D medical images (npj Digital Medicine)](https://www.nature.com/articles/s41746-025-01649-4)

### Vision-Language Models
- [Vision-language foundation models review (Springer)](https://link.springer.com/article/10.1007/s13534-025-00484-6)
- [Vision-language models for medical report generation (Frontiers)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1430984/full)
- [Vision-language foundation model for 3D medical imaging (Nature npj AI)](https://www.nature.com/articles/s44387-025-00015-9)
- [Multimodal Large Language Models in Medical Imaging (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12479233/)

### Cross-Modal Attention
- [Brain-MGF: Multimodal Graph Fusion (arXiv)](https://arxiv.org/html/2511.18325)
- [CMAF-Net (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11250309/)
- [Multi-modal cross-attention for Alzheimer's (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0010482523005152)
- [Multimodal contrastive learning for Alzheimer's (IEEE)](https://ieeexplore.ieee.org/iel7/10385250/10385251/10385864.pdf)
- [Multi-modal cross-domain self-supervised (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S089360802400995X)

### Self-Supervised Learning
- [BrainMAE (arXiv)](https://arxiv.org/abs/2406.17086)
- [BrainMAE framework (arXiv)](https://arxiv.org/html/2406.17086v1)
- [Self-supervised graph contrastive learning (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11875923/)
- [Self-supervised learning for medical imaging (Nature npj Digital Medicine)](https://www.nature.com/articles/s41746-023-00811-0)
- [SSL to unveil brain dysfunctional signatures (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12324563/)

### Few-Shot and Zero-Shot Learning
- [Few-Shot Learning with Deep Triplet Networks (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-33391-1_21)
- [Expert-Guided Few-Shot Learning (arXiv)](https://arxiv.org/html/2509.08007)
- [Few-shot learning for medical imaging (Wiley)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/nbm.5143)
- [Domain Adaptation Survey (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/)
- [Few-Shot Learning for EEG (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11266297/)
- [Zero-shot learning for BCI (PMC)](https://ncbi.nlm.nih.gov/pmc/articles/PMC6746355)

### Diffusion Models
- [Conditional Diffusion Models for Brain MRI (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38578863/)
- [GitHub - Diffusion Models in Medical Imaging](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging)
- [Generating Realistic Brain MRIs (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10758344/)
- [Advanced image generation for cancer (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11387006/)
- [Counterfactual MRI Generation (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38370616/)
- [Adaptive Latent Diffusion (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Adaptive_Latent_Diffusion_Model_for_3D_Medical_Image_to_Image_WACV_2024_paper.html)

### Domain Adaptation for LLMs
- [LLM Architectures in Healthcare (JMIR)](https://www.jmir.org/2025/1/e70315)
- [Multimodal Integration in Healthcare (JMIR)](https://www.jmir.org/2025/1/e76557)
- [3DS: Medical Domain Adaptation (arXiv)](https://arxiv.org/html/2410.10901)
- [LLMs in Healthcare Review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12189880/)
- [Medical foundation LLMs (Nature npj Digital Medicine)](https://www.nature.com/articles/s41746-025-01533-1)
- [LLM evaluations in clinical medicine (BMC)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-025-02954-4)

### Prompt Tuning
- [Fine-Tuned Vision Transformer (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10297056/)
- [Prompt-Based Tuning for Medical Segmentation (MDPI)](https://www.mdpi.com/2306-5354/10/7/879)
- [Hierarchical multi-scale ViT (Nature Scientific Reports)](https://www.nature.com/articles/s41598-025-23100-0)
- [DVPT: Dynamic Visual Prompt Tuning (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000474)
- [Efficiently Training ViT on MRI (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38083552/)

### Benchmark Datasets
- [UK Biobank Brain Imaging (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5770339/)
- [UK Biobank QC (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1053811917308613)
- [Multimodal population brain imaging (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5086094/)
- [GenMIND Dataset (Nature Scientific Data)](https://www.nature.com/articles/s41597-024-04157-4)

### Korean Context
- [Transfer Learning Korean Adolescents (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S2451902225001338)

---

**Document End**
