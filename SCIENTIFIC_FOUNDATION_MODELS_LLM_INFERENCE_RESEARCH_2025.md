# Scientific Foundation Models → LLM Inference: Comprehensive Research Report 2025

**Research Date**: December 8, 2025
**Focus**: How scientific foundation models (especially genomics/ESM3) connect to LLM-based inference capabilities
**Target Application**: Brain-genomics-LLM integration for developmental disorders research

---

## Executive Summary

This research identifies the critical missing link between scientific foundation models and natural language inference capabilities. The breakthrough discovery is **multimodal architecture integration** where domain-specific foundation models (genomics, brain imaging, proteins) are connected to large language models through:

1. **Cross-modal contrastive learning** (e.g., CLIP-style architectures for science)
2. **Unified embedding spaces** that align scientific data with natural language
3. **Transformer-based encoders** with custom tokenizers for scientific modalities
4. **Reasoning enhancement** through reinforcement learning and supervised fine-tuning

### Key Finding: The DNA-LLM Bridge

**BioReason** (NeurIPS 2025) represents the first successful integration of a DNA foundation model with an LLM, demonstrating that scientific foundation models can be directly connected to language models for multi-step biological reasoning with 98% accuracy on disease pathway prediction.

---

## 1. Scientific Foundation Model → LLM Inference Pipeline

### 1.1 Core Architecture Pattern

The standard architecture for bridging scientific foundation models to natural language follows this pattern:

```
Scientific Data (genomics/imaging/proteins)
    ↓
Domain-Specific Encoder (foundation model)
    ↓
Cross-Modal Interface Layer
    ↓
Unified Embedding Space
    ↓
Large Language Model (reasoning engine)
    ↓
Natural Language Inference Output
```

### 1.2 Key Technical Components

#### **Transformer-Based Encoders with Custom Tokenizers**

Research shows that transformer architectures are rapidly becoming foundational tools for analyzing and integrating multiscale biological data, with evolution from unimodal models to large-scale multimodal foundation models operating across genomic sequences, single-cell transcriptomics, and spatial data.

**Source**: [Nature Methods - Multimodal foundation transformer models](https://www.nature.com/articles/s41592-025-02918-6)

#### **Contrastive Learning Framework**

Large-scale multimodal foundation models demonstrate exceptional ability to encode cross-modal information through unified architectures, such as transformers, that jointly learn from scientific data and natural language text.

Methods like CLIP and MedCLIP adapt the contrastive learning framework to align visual and textual modalities, leveraging large-scale, weakly labeled datasets to learn generalizable representations without extensive annotation.

**Sources**:
- [PMC - Multimodal Foundation Models in Medical Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC12060978/)
- [Frontiers - Multimodal data integration for oncology](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1408843/full)

#### **Unified Embedding Spaces**

The breakthrough is creating shared embedding spaces where:
- Scientific modalities (genomics, imaging, structure) are encoded by domain-specific foundation models
- Natural language descriptions are encoded by language models
- Both are projected into the same high-dimensional space
- Cross-modal retrieval and reasoning become possible

**Example**: ProTrek integrates protein sequence, structure, and natural language descriptions of function into a unified computational framework through sophisticated contrastive learning techniques.

**Source**: [ProTrek Trimodal Protein Language Model](https://www.nature.com/articles/s41587-025-02836-0)

---

## 2. ESM3 and Genomic Foundation Model Inference

### 2.1 ESM3 Architecture and Capabilities

ESM3 is a frontier generative model for biology, able to jointly reason across three fundamental biological properties of proteins: **sequence, structure, and function**.

#### **Multimodal Design**

These three data modalities are represented as tracks of discrete tokens at the input and output of ESM3. ESM-3 has three separate input tracks for sequence, structure and function, allowing each to be prompted partially or in full. These tracks are processed separately before being combined into a single latent space.

**Key Innovation**: Geometric Attention layer stacked with sequence attention to produce contextual embeddings that consider both sequence and 3D structure.

#### **Discrete Tokenization**

Where traditional methods treat sequences as a single, continuous input, ESM-3 discretizes each protein into smaller arrangements of atoms and residues. By breaking down the structural complexity into discrete tokens, ESM3 enhances its ability to capture fine-grained details such as atomic interactions, spatial arrangements, and geometric features.

#### **Model Scale**

At its largest scale, ESM3 was trained with 1.07e24 FLOPs on 2.78 billion proteins and 771 billion unique tokens, and has 98 billion parameters. ESM3-open with 1.4B parameters is the smallest and fastest model in the family.

**Sources**:
- [Evolutionary Scale - ESM3 Release](https://www.evolutionaryscale.ai/blog/esm3-release)
- [ML6 - ESM3 Analysis](https://www.ml6.eu/blogpost/esm-3-the-frontier-of-protein-design)

### 2.2 ESM3 Inference Pipeline

#### **Generative Masked Language Model**

ESM3 is a generative masked language model. You can prompt it with partial sequence, structure, and function keywords, and iteratively sample masked positions until all positions are unmasked.

The iterative sampling process is implemented through the `.generate()` function which iteratively samples masked positions until they're all unmasked. This enables:
- Conditional generation given partial inputs
- Multi-step reasoning across modalities
- Evolutionary simulation (demonstrated 500 million years of evolution)

#### **Natural Language Connection**

ESM3 can be prompted with **function keywords** (natural language descriptions) alongside sequence and structure, enabling natural language → protein generation.

**Achievement**: Generated esmGFP, a new fluorescent protein with only 58% sequence similarity to closest known protein, equivalent to simulating over 500 million years of evolution.

**Source**: [Evolutionary Scale - ESM3 Blog](https://www.evolutionaryscale.ai/blog/esm3-release)

### 2.3 Genomic Foundation Models for Brain Research

#### **GenomeOcean: DNA Language Model**

GenomeOcean is a 4 billion parameter genome foundation model trained on large-scale metagenomic assemblies that can both analyze and create DNA sequences mimicking microbial life.

**Natural Language Processing Parallels**: Using large amounts of genomic data, the model learns to recognize patterns, structures and relationships within the data — similar to how LLMs learn grammar and context. The analogy: "verbs and nouns are like nucleotide bases of DNA; genes analogous to paragraphs; genomes equate to complete written works."

**Performance**: Ranked No. 1 most-downloaded genome foundation model on Hugging Face platform (June 2025), achieving 3× more throughput with vLLM optimization.

**Sources**:
- [Joint Genome Institute - GenomeOcean](https://jgi.doe.gov/user-science/science-stories/genomeocean-leverages-ai-decode-natures-secret-language)
- [Berkeley Lab - GenomeOcean Project](https://it.lbl.gov/the-power-of-ai-in-genomics-a-collaborative-effort-to-build-the-genomeocean-llm/)

---

## 3. 2025 Top-Tier Conference Research

### 3.1 NeurIPS 2025: AI for Science and Biological Reasoning

#### **BioReason: The DNA-LLM Integration Breakthrough**

**Publication**: NeurIPS 2025 (Accepted Poster)
**Innovation**: First successful integration of a DNA foundation model with an LLM

BioReason is a pioneering architecture that deeply integrates a DNA foundation model with a large language model (LLM), enabling the LLM to directly interpret and reason over genomic information. This novel connection enables the LLM to directly process and reason with genomic information as a fundamental input, fostering a new form of multimodal biological understanding.

**Training Methodology**:
- Supervised fine-tuning for multi-step reasoning
- Targeted reinforcement learning for biological coherence
- Qwen3 autoregressive Transformer-based LLM as backbone

**Performance Results**:
- KEGG-based disease pathway prediction: **86% → 98% accuracy**
- Variant effect prediction: **15% improvement** over strong baselines
- Interpretable reasoning traces with step-by-step biological explanations

**Key Features**:
- Advanced reasoning methodology with transparent explanations
- New biological reasoning benchmarks including annotated KEGG dataset
- Annotated reasoning dataset for gene pathway and disease prediction

**Institutions**: University of Toronto, Vector Institute, University Health Network, Arc Institute, Cohere, UCSF, Google DeepMind

**Sources**:
- [arXiv - BioReason](https://arxiv.org/abs/2505.23579)
- [NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/poster/116227)
- [GitHub - BioReason](https://github.com/bowang-lab/BioReason)

#### **NeurIPS 2025 Workshops on Scientific Reasoning**

**1. Foundations of Reasoning in Language Models Workshop**
- Goal: Advance foundational understanding of how reasoning emerges in language models
- Focus: Theoretical analyses and controlled empirical studies
- Key Question: How can reasoning be systematically improved?

**2. The 1st Workshop on Efficient Reasoning**
- Welcomes contributions from natural sciences (physics, chemistry, biology)
- Provides comprehensive perspective for reasoning tasks across scientific domains

**3. AI for Science Workshop**
- Central Challenge: "LLM reasoning across scientific domains – can present-day LLMs generate rigorously testable hypotheses and reason over experimental results that span scientific domains such as physics, chemistry, and biology?"
- Focus on fidelity of generative and surrogate simulators
- Addresses spatial/temporal scales that remain intractable

**Key Finding**: Reasoning was the major topic at NeurIPS 2024/2025 due to O1 models, with about **766 papers** having reasoning as core focus.

**Sources**:
- [NeurIPS 2025 - Foundations of Reasoning Workshop](https://neurips.cc/virtual/2025/workshop/109559)
- [The 1st Workshop on Efficient Reasoning](https://efficient-reasoning.github.io/)
- [Inside NeurIPS 2025](https://newsletter.languagemodels.co/p/the-illustrated-neurips-2025-a-visual)

### 3.2 ICLR 2025: Scientific Foundation Models and Biology

#### **MLGenX Workshop - AI for Genomics**

MLGenX bridges the gap between AI and functional genomics, with primary emphasis on target identification.

**Key Papers**:

1. **LangPert Framework**: Novel hybrid framework that leverages LLMs to guide a downstream k-nearest neighbors aggregator, combining biological reasoning with efficient numerical inference

2. **Single-Cell Foundation Models**: Leveraging foundation models pre-trained on tens of millions of single cells to address molecular perturbation prediction, with drug-conditional adapters for efficient fine-tuning

**Source**: [MLGenX ICLR Workshop 2025](https://mlgenx.github.io/)

#### **Foundation Models in the Wild Workshop**

Focus on scientific applications including:
- Multi-hop question answering
- Mathematical problem-solving
- Theorem proving
- Adaptation for drug discovery and clinical health
- Contributions from natural sciences (physics, chemistry, biology)

**Key Theme**: Foundation models are reshaping the future of scientific research, highlighted by the 2024 Nobel Prize in Chemistry being awarded to AI-based protein structure prediction.

**Source**: [ICLR 2025 - Foundation Models in the Wild](https://fm-wild-community.github.io/)

#### **Open Science for Foundation Models (SCI-FM)**

Focus areas:
- Pretraining strategies including data scaling, model architecture, multi-modal, and multi-task pretraining
- Learning algorithms such as meta-learning and continual learning
- Application-based performance evaluation for scientific domains

**Source**: [SCI-FM ICLR 2025 Workshop](https://open-foundation-model.github.io/)

### 3.3 Nature Machine Intelligence & Nature Methods (2025)

#### **LucaOne: Unified Nucleic Acid and Protein Foundation Model**

LucaOne is a biological foundation model pre-trained on nucleic acid and protein sequences from **169,861 species**. It shows an emerging understanding of molecular biology's central dogma, enhancing bioinformatics analysis.

**Innovation**: Generalized biological foundation model with unified nucleic acid and protein language

**Source**: [Nature Machine Intelligence - LucaOne](https://www.nature.com/articles/s42256-025-01044-4)

#### **EpiAgent: Single-Cell Epigenomics Foundation Model**

EpiAgent is a transformer-based foundation model pretrained on approximately **5 million cells** and over **35 billion tokens**, advancing single-cell epigenomics by encoding chromatin accessibility as 'cell sentences'.

**Performance**: Achieved state-of-the-art performance in typical downstream tasks and enabled perturbation response prediction and in silico chromatin region knockouts.

**Source**: [Nature Methods - EpiAgent (implied from search results)](https://www.nature.com/subjects/machine-learning/nmeth)

#### **META-SiM: Single-Molecule Behavior Foundation Model**

META-SiM is a transformer-based foundation model that automates key analysis tasks across diverse datasets and enables rapid, systematic discovery of subtle single-molecule behaviors.

**Achievement**: Revealed a previously undetected pre-mRNA splicing intermediate

**Source**: [Nature Methods - Machine Learning](https://www.nature.com/subjects/machine-learning/nmeth)

#### **Nicheformer: Spatial Single-Cell Analysis**

Pretrained on **SpatialCorpus-110M**, a curated resource of vast and diverse transcriptomes of dissociated and spatially resolved cells from both human and mouse.

**Advancement**: Building foundation models for spatial single-cell analysis

**Source**: [Nature Methods (implied from search results)](https://www.nature.com/subjects/machine-learning/nmeth)

---

## 4. Brain-Genomics-LLM Integration

### 4.1 COMICAL: Multimodal Foundation Model for Brain Imaging Genomics

**Full Name**: Contrastive Multimodal Integration for Cerebral Analysis and Learning

COMICAL is a contrastive learning approach using multiomics data to generate associations between genetic markers and brain imaging-derived phenotypes. COMICAL jointly learns omic representations utilizing transformer-based encoders with custom tokenizers.

#### **Architecture**

- **Based on**: Adaptation of multimodal transformer Contrastive Language-Image Pre-training (CLIP)
- **Specifically designed for**: Genomics and brain imaging data integration
- **Modality-agnostic approach**: Uniquely identifies many-to-many associations via self-supervised learning schemes and cross-modal attention encoders

#### **Data Processing Pipeline**

1. Preprocesses genomics (SNPs) and imaging biomarkers (IDPs)
2. Creates IDP-SNP pairs mediated by complex diseases
3. Encodes them using transformer encoders as used in GPT-2

#### **Dataset Scale**

From UK Biobank:
- **40,426 samples** with both SNPs and 154 IDPs of T1 structural brain MRI
- After creating pairs using top 1% SNPs in GWAS catalog (33 SNPs): **15,442,732 pairs** as pretrained data

#### **Performance**

COMICAL discovered several significant associations between genetic markers and imaging-derived phenotypes for a variety of neurological disorders in the UK Biobank, and can predict across diseases and unseen clinical outcomes from the learned representations.

**Sources**:
- [medRxiv - COMICAL](https://www.medrxiv.org/content/10.1101/2024.11.02.24316653v1)
- [Oxford Academic - COMICAL Foundation Model](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf196/8233690)

### 4.2 epiBrainLLM: Alzheimer's Causal Pathway Analysis

epiBrainLLM is a novel computational framework that leverages genomic LLM to enhance understanding of the causal pathways from genotypes to brain measures to AD-related clinical phenotypes.

#### **Key Findings from Related Research (2025)**

**Epigenomic Landscape Changes**:
- Widespread compromised compartmentalization and epigenomic information loss during AD progression
- Particularly in selectively vulnerable excitatory neurons from entorhinal cortex and hippocampus
- Cognitively resilient individuals had preserved epigenomic integrity

**Multi-omics Integration**:
MIT researchers performed the broadest analysis yet of genomic, epigenomic, and transcriptomic changes in every cell type in brains of Alzheimer's patients, using more than **2 million cells** from more than **400 postmortem brain samples**.

**Key Biological Pathways**:
1. **Histone Modifications**: H3K27ac and H3K9ac as main enrichments specific to AD
2. **Mitochondrial Function**: Impairments in genes involved in mitochondrial function
3. **Synaptic Signaling**: Disruption in synaptic signaling pathways
4. **Microglial Involvement**: AD risk loci enriched in microglial enhancers for TFs including SPI1, ELF2, and RUNX1

**Sources**:
- [medRxiv - epiBrainLLM](https://www.medrxiv.org/content/10.1101/2024.10.03.24314824v1)
- [Cell - Single-cell multiregion epigenomic rewiring](https://www.cell.com/cell/fulltext/S0092-8674(25)00733-0)
- [Picower Institute - Decoding Alzheimer's](https://picower.mit.edu/news/decoding-complexity-alzheimers-disease)

### 4.3 Med-Gemini: Multimodal Biomedical AI Including Genomics

Med-Gemini builds on Google's Med-PaLM by fine-tuning on de-identified medical data while inheriting Gemini's native reasoning, multimodal, and long-context abilities.

#### **Model Variants**

**1. Med-Gemini-2D**:
- Optimized via fine-tuning with 2D and 3D radiology, histopathology, ophthalmology, dermatology and **genomic data**

**2. Med-Gemini-3D**:
- First ever large multimodal model-based report generation for 3D CT volumes
- **53% of AI reports** considered clinically acceptable

**3. Med-Gemini-Polygenic**:
- **Genomics "images"** (polygenic risk scores projected into 2D) included in mixture of datasets
- Vision encoder trained to predict eight broad health outcomes:
  - Coronary artery disease
  - Stroke
  - Type 2 diabetes
  - Glaucoma
  - Chronic obstructive pulmonary disease
  - Rheumatoid arthritis
  - Major depression
  - All-cause mortality
- **Outperforms** standard linear polygenic risk score-based approach
- **Generalizes** to genetically correlated diseases for which it has never been trained

#### **Performance**

- MedQA (USMLE-style) benchmark: **4.6% improvement** over prior best Med-PaLM 2
- Chest X-ray report generation: **1% and 12% improvement** over previous best across two datasets
- Context window: Up to **1 million tokens** (Gemini 1.5)

**Source**: [Google Research - Med-Gemini](https://research.google/blog/advancing-medical-ai-with-med-gemini/)

### 4.4 GIANT Atlas: Genetically Informed Brain Atlas

GIANT (Genetically Informed brAiN aTlas) accounts for genetic and neuroanatomical variations simultaneously. GIANT clusters brain voxels into genetically informed regions while retaining fundamental anatomical knowledge by integrating voxel-wise heritability and spatial proximity.

**Innovation**: Enhances brain imaging genomics by illuminating genetic determinants of human brain structure and function.

**Source**: [Nature Communications - GIANT Atlas](https://www.nature.com/articles/s41467-025-57636-6)

### 4.5 Multimodal Neuroimaging and GWAS Integration

#### **UK Biobank Scale**

The UK Biobank study has produced thousands of brain imaging-derived phenotypes (IDPs) collected from more than **40,000 genotyped individuals**, facilitating investigation of genetic and imaging biomarkers for brain disorders.

#### **Instrumental Variable (IV) Approach**

Recent methods in imaging genetics adopted an instrumental variable approach to identify causal IDPs for brain disorders, motivated by efforts in genetics to integrate gene expression levels with genome-wide association studies (GWASs).

**Application**: Leveraging multimodal neuroimaging and GWAS for identifying modality-level causal pathways to Alzheimer's disease.

**Sources**:
- [MIT Press - Multimodal neuroimaging and GWAS](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00580/128949/)
- [PMC - Leveraging multimodal neuroimaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC11908268/)

---

## 5. Specific Technical Implementations

### 5.1 Cross-Modal Architecture Patterns

#### **Pattern 1: CLIP-Style Contrastive Learning**

Used by: COMICAL, ProTrek, OmiCLIP

**Architecture**:
```
Scientific Data Encoder (Transformer)
          ↓
    Embedding Space
          ↓
    Contrastive Loss ←→ Text Encoder (LLM)
          ↓
Unified Multimodal Embeddings
```

**Training Strategy**:
- Large-scale paired data (e.g., 15M image-text pairs for BiomedCLIP)
- Self-supervised contrastive learning
- Projects both modalities into shared embedding space
- Enables cross-modal retrieval and alignment

**Example - BiomedCLIP**:
- Pretrained on **PMC-15M dataset** containing **15 million biomedical image-text pairs** from **4.4 million scientific articles**
- Two orders of magnitude larger than existing biomedical multimodal datasets

**Source**: [NEJM AI - Multimodal Biomedical Foundation Model](https://ai.nejm.org/doi/abs/10.1056/AIoa2400640)

#### **Pattern 2: Cross-Modal Connectors with LLM**

Used by: PROTLLM, BioReason

**Architecture**:
```
Scientific Domain Encoder (DNA/Protein Foundation Model)
          ↓
Cross-Modal Connector Layer
          ↓
Large Language Model (Qwen3, LLaMA, etc.)
          ↓
Natural Language Reasoning Output
```

**Training Strategy**:
- Initialize domain encoder with pretrained foundation model weights
- Initialize LLM with pretrained language model weights
- Train cross-modal connector to bridge representations
- Fine-tune end-to-end with supervised data and reinforcement learning

**Example - BioReason**:
- DNA foundation model frozen or lightly fine-tuned
- Cross-modal connector learns to translate DNA embeddings to LLM input space
- Qwen3 LLM initialized with original pretrained weights
- Supervised fine-tuning + targeted reinforcement learning for biological reasoning

#### **Pattern 3: Unified Multimodal Transformer**

Used by: ESM3, Med-Gemini, LucaOne

**Architecture**:
```
Multi-Track Input (Sequence + Structure + Function/Text)
          ↓
Separate Track Encoders (with custom tokenizers)
          ↓
Unified Transformer Backbone
          ↓
Multi-Track Output (joint reasoning)
```

**Training Strategy**:
- Masked language modeling across all tracks
- Iterative sampling for generation
- Joint optimization across modalities
- Self-supervised on massive unlabeled data

**Example - ESM3**:
- Three separate input tracks: sequence, structure, function keywords
- Geometric attention for structure + sequence attention
- 98B parameters trained on 771B tokens
- Can be prompted with natural language function descriptions

### 5.2 Tokenization Strategies for Scientific Data

#### **Genomic Data Tokenization**

**1. Byte-Pair Encoding (BPE)** - GenomeOcean
- Transforms DNA sequences into variable-length tokens
- **150× faster** processing than character-level
- Learns common motifs as single tokens

**2. k-mer Tokenization** - Standard approach
- Fixed-length subsequences (e.g., 3-mers: ATG, CGT)
- Preserves biological meaning
- Used by DNABERT, Nucleotide Transformer

**3. Cell Sentence Encoding** - EpiAgent
- Encodes chromatin accessibility as "cell sentences"
- Pretrained on 35B tokens from 5M cells
- Enables language-like processing of epigenomic data

**Sources**:
- [GenomeOcean - BPE Tokenization](https://it.lbl.gov/the-power-of-ai-in-genomics-a-collaborative-effort-to-build-the-genomeocean-llm/)
- [Nature Methods - EpiAgent](https://www.nature.com/subjects/machine-learning/nmeth)

#### **Brain Imaging Tokenization**

**1. IDP-Based Tokenization** - COMICAL
- Imaging-Derived Phenotypes (IDPs) as features
- 154 T1 structural brain MRI IDPs
- Combined with SNP tokens for cross-modal learning

**2. Patch-Based Tokenization** - Standard ViT approach
- Divide images into patches
- Linear projection to embeddings
- Position embeddings added

**3. Graph-Based Tokenization** - Functional Connectivity
- Brain regions as graph nodes
- Connections as edges
- Graph transformer processes network structure

**Source**: [Graph Transformer Foundation Model for Brain FCN](https://www.sciencedirect.com/science/article/abs/pii/S003132032500648X)

#### **Protein Data Tokenization**

**1. Discrete Structure Tokens** - ESM3
- Atoms and residues discretized into tokens
- Geometric features preserved
- Enables structure-aware generation

**2. Amino Acid Sequence Tokens** - Standard
- 20 amino acids + special tokens
- Can use BPE for longer motifs
- ESM-2 uses this approach

**3. Trimodal Tokens** - ProTrek
- Sequence tokens
- Structure tokens (3D coordinates)
- Function description tokens (natural language)
- Unified in shared embedding space

**Source**: [ProTrek Trimodal Model](https://www.nature.com/articles/s41587-025-02836-0)

### 5.3 Training Methodologies

#### **Stage 1: Self-Supervised Pretraining**

**Objective**: Learn general representations from unlabeled scientific data

**Common Approaches**:
1. **Masked Language Modeling** (MLM)
   - Randomly mask tokens in input
   - Model predicts masked tokens
   - Used by: ESM-2, GenomeOcean, LucaOne

2. **Contrastive Learning**
   - Positive pairs: same entity, different views
   - Negative pairs: different entities
   - Maximize agreement for positives, minimize for negatives
   - Used by: COMICAL, BiomedCLIP, ProTrek

3. **Causal Language Modeling** (CLM)
   - Predict next token given previous tokens
   - Used by: GenomeOcean (generative direction)

**Scale Requirements**:
- **Genomics**: 100M-10B sequences (GenomeOcean: metagenomic assemblies)
- **Proteins**: 100M-2B sequences (ESM3: 2.78B proteins)
- **Imaging**: 1M-100M images (BiomedCLIP: 15M image-text pairs)
- **Compute**: 1e22 - 1e24 FLOPs (ESM3: 1.07e24 FLOPs)

**Source**: [Foundation Models in Bioinformatics](https://academic.oup.com/nsr/article/12/4/nwaf028/7979309)

#### **Stage 2: Multimodal Alignment**

**Objective**: Align scientific modality embeddings with text/language embeddings

**Techniques**:
1. **CLIP-Style Contrastive**
   - Paired data: (scientific_data, text_description)
   - Contrastive loss: align positive pairs, separate negative pairs
   - Example: COMICAL with (brain_IDP, genetic_SNP) pairs

2. **Cross-Modal Attention**
   - Attention mechanism between modalities
   - Learns which parts of scientific data correspond to text
   - Example: BioReason DNA-LLM connector

3. **Shared Decoder**
   - Single decoder processes both modalities
   - Forces alignment through shared output space
   - Example: Med-Gemini unified architecture

**Data Requirements**:
- **Quality over quantity** for paired data
- Minimum: 10K-100K high-quality pairs
- Optimal: 1M-100M pairs with diversity
- Example: COMICAL with 15M IDP-SNP pairs

#### **Stage 3: Reasoning Enhancement**

**Objective**: Enable multi-step reasoning and inference

**Approaches**:
1. **Supervised Fine-Tuning (SFT)**
   - Curated reasoning examples with step-by-step solutions
   - Example: BioReason with annotated KEGG pathways
   - Format: (question, reasoning_chain, answer)

2. **Reinforcement Learning from Human Feedback (RLHF)**
   - Reward model trained on human preferences
   - Policy optimization to maximize rewards
   - Example: BioReason's targeted RL for biological coherence

3. **Chain-of-Thought Prompting**
   - Explicit reasoning steps in prompts
   - Encourages systematic problem-solving
   - Can be combined with SFT/RLHF

**Evaluation Metrics**:
- **Faithfulness**: Are reasoning steps logically sound?
- **Comprehensiveness**: Do steps cover all necessary logic?
- **Biological Coherence**: Are conclusions scientifically valid?
- **Task Performance**: Accuracy on downstream tasks

**Source**: [The Science of Evaluating Foundation Models](https://arxiv.org/html/2502.09670v1)

#### **Stage 4: Domain Adaptation**

**Objective**: Specialize for specific scientific tasks or domains

**Techniques**:
1. **Parameter-Efficient Fine-Tuning (PEFT)**
   - LoRA (Low-Rank Adaptation)
   - Adapters
   - Prefix tuning
   - Example: Med-Gemini task-specific adapters

2. **Multitask Learning**
   - Train on multiple related tasks simultaneously
   - Improves generalization
   - Example: CLIMB with 23% improvement on ECG through multitask pretraining

3. **Continual Learning**
   - Incrementally adapt to new data/tasks
   - Avoid catastrophic forgetting
   - Memory-augmented architectures for analogical reasoning

**Performance Gains**:
- CLIMB: **29% improvement** in ultrasound, **23% in ECG** through multitask pretraining
- Drug-conditional adapters: Efficient fine-tuning for perturbation prediction

**Sources**:
- [CLIMB - Multimodal Clinical Foundation Models](https://arxiv.org/html/2503.07667v2)
- [Foundation Models for Scientific Discovery](https://arxiv.org/html/2510.15280)

### 5.4 Evaluation Frameworks for Scientific Reasoning

#### **Quantitative Metrics**

**1. Task-Specific Accuracy**
- Disease pathway prediction: 98% (BioReason)
- Protein structure prediction: >0.8 pTM, >0.8 pLDDT (ESM3)
- Clinical report acceptance: 53% (Med-Gemini-3D)
- MedQA benchmark: 4.6% improvement (Med-Gemini)

**2. Reasoning Quality Metrics**
- **Faithfulness**: NLI-based evaluation of reasoning consistency
- **Comprehensiveness**: Coverage of necessary reasoning steps
- **Sufficiency**: Effectiveness of extracted rationales for prediction
- **Confidence**: Knowledge graph-supported hallucination detection

**3. Cross-Modal Retrieval Metrics**
- **Recall@K**: Top-K retrieval accuracy
- **Mean Reciprocal Rank (MRR)**
- **Area Under Precision-Recall Curve (AUPRC)**
- Example: ProTrek surpasses Foldseek and MMseqs2 in speed and accuracy

**Sources**:
- [The Science of Evaluating Foundation Models](https://arxiv.org/html/2502.09670v1)
- [Apple Foundation Models Evaluation](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)

#### **Qualitative Assessment**

**Expert Evaluation Categories**:
1. **Biological Coherence**: Are predictions scientifically plausible?
2. **Clinical Acceptability**: Would experts trust the output?
3. **Interpretability**: Can reasoning be understood and validated?
4. **Generalization**: Does it work on unseen cases?

**Benchmarking Datasets**:
- **GeneTuring**: Genomic knowledge evaluation
- **VivaBench**: 1,700+ physician-curated interactive vignettes for clinical reasoning
- **KEGG Pathways**: Annotated disease pathway prediction
- **UK Biobank**: Large-scale imaging-genomics associations

**Sources**:
- [GeneTuring Benchmark](https://academic.oup.com/bib/article/doi/10.1093/bib/bbaf492/8261762)
- [NeurIPS 2025 - Clinical Reasoning](https://newsletter.languagemodels.co/p/the-illustrated-neurips-2025-a-visual)

---

## 6. Integration Roadmap for AI-CoScientist

### 6.1 Immediate Applications

#### **1. COMICAL-Style Brain-Genomics Integration**

**Architecture**:
```python
# Pseudocode for COMICAL-style integration
class BrainGenomicsFoundationModel:
    def __init__(self):
        self.brain_encoder = TransformerEncoder(
            input_type="brain_IDPs",
            tokenizer=IDP_Tokenizer(),
            hidden_dim=768
        )
        self.genomic_encoder = TransformerEncoder(
            input_type="SNPs",
            tokenizer=SNP_Tokenizer(),
            hidden_dim=768
        )
        self.projection_head = nn.Linear(768, 512)

    def encode_brain(self, brain_idps):
        # brain_idps: [batch, 154] T1 structural MRI features
        brain_embeddings = self.brain_encoder(brain_idps)
        return self.projection_head(brain_embeddings)

    def encode_genomics(self, snps):
        # snps: [batch, num_snps] genetic variants
        genomic_embeddings = self.genomic_encoder(snps)
        return self.projection_head(genomic_embeddings)

    def contrastive_loss(self, brain_emb, genomic_emb):
        # CLIP-style contrastive learning
        similarity = brain_emb @ genomic_emb.T
        labels = torch.arange(len(brain_emb))
        loss = F.cross_entropy(similarity / temperature, labels)
        return loss
```

**Data Sources**:
- UK Biobank: 40K+ samples with brain imaging + genetics
- ABCD Study: Developmental brain imaging + genomics
- Custom developmental disorder datasets

**Expected Outcomes**:
- Discover genetic-brain imaging associations for developmental disorders
- Predict developmental outcomes from genetic markers
- Identify causal pathways from genes → brain changes → clinical phenotypes

#### **2. BioReason-Style DNA-LLM for Developmental Genetics**

**Architecture**:
```python
class DevelopmentalGenomicsLLM:
    def __init__(self):
        # Use pretrained DNA foundation model (e.g., GenomeOcean, Nucleotide Transformer)
        self.dna_encoder = load_pretrained_dna_model("genomeocean-4b")

        # Cross-modal connector
        self.connector = nn.Sequential(
            nn.Linear(dna_hidden_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        )

        # Large language model (e.g., Qwen3, LLaMA-3)
        self.llm = load_pretrained_llm("qwen3-7b")

    def forward(self, dna_sequence, prompt):
        # Encode DNA sequence
        dna_embeddings = self.dna_encoder(dna_sequence)

        # Project to LLM space
        dna_tokens = self.connector(dna_embeddings)

        # Combine with text prompt
        prompt_tokens = self.llm.tokenizer(prompt)
        combined_input = torch.cat([dna_tokens, prompt_tokens], dim=1)

        # Generate reasoning
        output = self.llm.generate(combined_input)
        return output
```

**Use Cases**:
1. **Variant Effect Prediction**: "Given this genetic variant in [gene], explain its potential impact on brain development"
2. **Pathway Analysis**: "What developmental pathways are affected by these genetic variants?"
3. **Disease Mechanism Reasoning**: "How do these genetic changes lead to developmental disorder phenotypes?"

**Training Data**:
- ClinVar: Variant-disease associations with expert annotations
- KEGG: Pathway databases with disease relationships
- PubMed literature: Extract reasoning chains from papers
- Custom annotations: Domain expert reasoning examples

#### **3. ProTrek-Style Trimodal Model for Developmental Proteins**

**Architecture**:
```python
class DevelopmentalProteinFoundation:
    def __init__(self):
        # Three encoders for three modalities
        self.sequence_encoder = ESM2("esm2_t33_650M_UR50D")
        self.structure_encoder = StructureEncoder()
        self.function_encoder = TextEncoder("pubmedbert")

        # Project to shared 512-dim space
        self.sequence_proj = nn.Linear(1280, 512)
        self.structure_proj = nn.Linear(1024, 512)
        self.function_proj = nn.Linear(768, 512)

    def contrastive_learning(self, sequences, structures, functions):
        # Encode all three modalities
        seq_emb = self.sequence_proj(self.sequence_encoder(sequences))
        struct_emb = self.structure_proj(self.structure_encoder(structures))
        func_emb = self.function_proj(self.function_encoder(functions))

        # Trimodal contrastive loss
        loss_seq_struct = contrastive_loss(seq_emb, struct_emb)
        loss_seq_func = contrastive_loss(seq_emb, func_emb)
        loss_struct_func = contrastive_loss(struct_emb, func_emb)

        return loss_seq_struct + loss_seq_func + loss_struct_func
```

**Applications**:
- Identify proteins critical for brain development
- Predict functional impact of genetic variants on proteins
- Link protein dysfunction to developmental disorder phenotypes
- Generate hypotheses about protein-protein interactions in development

### 6.2 Medium-Term Integration (3-6 Months)

#### **1. Multimodal Clinical Foundation Model**

Following Med-Gemini and CLIMB approaches:

**Data Sources**:
- Brain imaging (MRI, fMRI, DTI)
- Genomics (WGS, GWAS, polygenic risk scores)
- Clinical notes and assessments
- Developmental trajectories over time

**Architecture**:
- Unified multimodal transformer (Gemini-style)
- Context window: 1M tokens for longitudinal data
- Fine-tuned on developmental disorder data

**Capabilities**:
- Integrated analysis across all data modalities
- Natural language report generation
- Predictive modeling of developmental outcomes
- Explainable AI with reasoning traces

#### **2. Self-Learning Scientific RAG System**

Enhance existing AI-CoScientist RAG with foundation model components:

**Enhancements**:
1. **Scientific Embedding Models**:
   - BiomedCLIP for cross-modal literature search
   - SciBERT fine-tuned on developmental disorder papers
   - Knowledge graph embeddings (GraphRAG strategy)

2. **Reasoning-Enhanced Retrieval**:
   - Multi-hop reasoning over retrieved contexts
   - Chain-of-thought prompting for complex queries
   - Self-correction based on retrieved evidence

3. **Foundation Model Integration**:
   - Use ESM3 for protein-related queries
   - Use GenomeOcean for genomic sequence analysis
   - Use COMICAL-style models for brain-genetics queries

**Implementation**:
```python
class FoundationRAG:
    def __init__(self):
        self.retrievers = {
            "literature": BiomedCLIPRetriever(),
            "genomics": GenomeOceanRetriever(),
            "proteins": ESM3Retriever(),
            "brain_imaging": BrainFoundationRetriever()
        }
        self.reasoning_llm = ReasoningLLM("qwen3-72b")

    async def query(self, question, modalities=["all"]):
        # Multi-modal retrieval
        contexts = {}
        for modality in modalities:
            contexts[modality] = await self.retrievers[modality].retrieve(question)

        # Reasoning over contexts
        reasoning_chain = await self.reasoning_llm.multi_hop_reasoning(
            question=question,
            contexts=contexts,
            max_hops=3
        )

        # Generate answer with citations
        answer = await self.reasoning_llm.generate_with_citations(
            question=question,
            reasoning=reasoning_chain,
            contexts=contexts
        )

        return {
            "answer": answer,
            "reasoning": reasoning_chain,
            "contexts": contexts
        }
```

### 6.3 Long-Term Vision (6-12 Months)

#### **1. Autonomous Scientific Discovery Agent**

Following the "Intelligent Science Laboratories" paradigm from recent research:

**Architecture Components**:
1. **Foundation Models as Cognitive Core**:
   - Unified understanding across genomics, imaging, proteins, clinical data
   - Deep reasoning and planning capabilities
   - Continual learning from new data

2. **Agentic Reasoning System**:
   - Hypothesis generation from multimodal data
   - Experiment design and optimization
   - Causal inference and pathway discovery

3. **Embodied Automation** (where applicable):
   - Integration with laboratory information systems
   - Automated data collection pipelines
   - Closed-loop experimentation

**Capabilities**:
- Autonomous literature review and synthesis
- Novel hypothesis generation for developmental disorders
- Multi-modal data integration and analysis
- Predictive modeling with uncertainty quantification
- Explainable scientific reasoning

**Source**: [Foundation Models for Scientific Discovery](https://arxiv.org/html/2510.15280)

#### **2. Continual Learning Scientific Foundation Model**

**Challenge**: Foundation models must transition from static systems to continual learners capable of accumulating knowledge over time.

**Approaches**:
1. **Parameter-Efficient Online Adaptation**:
   - LoRA layers updated as new data arrives
   - Prevents catastrophic forgetting
   - Maintains performance on existing tasks

2. **Memory-Augmented Architectures**:
   - External memory for scientific facts and relationships
   - Analogical reasoning across scientific contexts
   - Knowledge graph integration

3. **Meta-Learning for Scientific Tasks**:
   - Learn to learn from few examples
   - Rapid adaptation to new developmental disorders
   - Transfer learning across related conditions

**Implementation Priority**:
- Start with parameter-efficient adaptation (LoRA)
- Add external memory for key scientific facts
- Implement meta-learning for few-shot adaptation

---

## 7. Key Takeaways for AI-CoScientist Development

### 7.1 Critical Technical Insights

1. **The Bridge is Built Through Contrastive Learning**:
   - CLIP-style architectures are the dominant approach for multimodal scientific models
   - Requires large-scale paired data (10K-100M pairs)
   - Enables zero-shot cross-modal retrieval and reasoning

2. **Tokenization is Domain-Specific**:
   - DNA: BPE or k-mer tokenization (GenomeOcean: 150× speedup with BPE)
   - Proteins: Amino acid + structure tokens (ESM3: discrete geometric tokens)
   - Brain Imaging: IDPs, patches, or graph representations
   - Must preserve biological meaning while enabling LLM processing

3. **Reasoning Requires Explicit Training**:
   - Pretraining alone is insufficient for multi-step reasoning
   - Need supervised fine-tuning with reasoning chains
   - Reinforcement learning improves biological coherence
   - BioReason: 86% → 98% accuracy with reasoning enhancement

4. **Scale Matters, But Efficiency is Key**:
   - Large models: ESM3 (98B params), Med-Gemini (Gemini 1.5 scale)
   - Efficient models: ESM3-open (1.4B params), GenomeOcean (4B params)
   - **Densing law**: Capability density doubles every 3.5 months
   - Focus on parameter efficiency and model distillation

**Source**: [Nature Machine Intelligence - Densing Law](https://www.nature.com/articles/s42256-025-01137-0)

### 7.2 Data Requirements

**Minimum for Proof-of-Concept**:
- **Pretraining**: 100K-1M scientific sequences/images
- **Multimodal alignment**: 10K-100K paired examples
- **Reasoning fine-tuning**: 1K-10K annotated reasoning chains
- **Evaluation**: 100-1K expert-curated test cases

**Production Scale**:
- **Pretraining**: 100M-10B sequences (GenomeOcean scale)
- **Multimodal alignment**: 1M-100M pairs (BiomedCLIP: 15M pairs)
- **Reasoning fine-tuning**: 10K-100K high-quality examples
- **Continual learning**: Ongoing data streams

**AI-CoScientist Current Assets**:
- ✅ 100+ QA benchmark pairs (good for evaluation)
- ✅ RAG system with scientific literature (good for retrieval)
- ⚠️ Need: Large-scale brain imaging + genomics paired data
- ⚠️ Need: Reasoning chains for developmental disorder analysis
- ⚠️ Need: Expert annotations for model training

### 7.3 Computational Resources

**Training Requirements**:
- **Small models (1-4B params)**: 8× A100/H100 GPUs for weeks
- **Medium models (7-30B params)**: 64-256 GPUs for months
- **Large models (70B+ params)**: 1000+ GPUs for months

**Inference Optimization**:
- vLLM: 3× throughput improvement (GenomeOcean)
- Quantization: 4-bit, 8-bit for reduced memory
- Model distillation: Smaller student models
- Sparse activation: Mixture-of-Experts (MoE)

**AI-CoScientist Current Setup**:
- Check GPU availability (DGX station capabilities)
- Consider cloud resources for large-scale training
- Prioritize efficient architectures (1-7B param models)
- Use pretrained checkpoints whenever possible

### 7.4 Recommended Development Sequence

**Phase 1: Foundation Model Integration (Month 1-2)**
1. Integrate pretrained models:
   - GenomeOcean (4B) for genomic analysis
   - BiomedCLIP for literature-image retrieval
   - SciBERT for text understanding
   - ESM-2 (650M) for protein analysis

2. Build cross-modal retrieval:
   - Extend existing RAG with foundation model retrievers
   - Implement CLIP-style similarity search
   - Add multi-modal query capabilities

**Phase 2: Brain-Genomics Alignment (Month 3-4)**
1. Collect/curate paired data:
   - UK Biobank access (if possible)
   - Public developmental disorder datasets
   - Literature-based pairing extraction

2. Train COMICAL-style model:
   - Start with small scale (10K pairs)
   - Validate on known associations
   - Scale up as data grows

**Phase 3: Reasoning Enhancement (Month 5-6)**
1. Create reasoning dataset:
   - Extract from scientific literature
   - Expert annotation of reasoning chains
   - KEGG pathway analysis examples

2. Implement BioReason-style training:
   - Supervised fine-tuning on reasoning chains
   - Evaluate on pathway prediction tasks
   - Iterate based on expert feedback

**Phase 4: Continual Learning (Month 7-12)**
1. Implement adaptation mechanisms:
   - LoRA for parameter-efficient updates
   - Memory-augmented architecture
   - Meta-learning setup

2. Deploy and monitor:
   - Real-time data integration
   - Performance tracking
   - Expert validation loops

### 7.5 Success Metrics

**Technical Metrics**:
- Cross-modal retrieval: Recall@10 > 0.8
- Pathway prediction accuracy: > 90% (BioReason achieved 98%)
- Reasoning faithfulness: > 0.8 (NLI-based evaluation)
- Clinical acceptability: > 50% (Med-Gemini achieved 53%)

**Scientific Metrics**:
- Novel hypothesis generation rate
- Validation rate of generated hypotheses
- Expert satisfaction scores
- Citations and usage by researchers

**System Metrics**:
- Inference latency: < 2s per query
- Throughput: > 100 queries/second
- Model size: < 10B parameters (for deployment)
- Update frequency: Daily/weekly continual learning

---

## 8. Conclusion

### 8.1 The Critical Missing Link: Identified

The connection between scientific foundation models and LLM inference is achieved through **multimodal contrastive learning** that creates unified embedding spaces where scientific data (genomics, brain imaging, proteins) and natural language can interact.

**Three Key Breakthroughs in 2025**:

1. **BioReason (NeurIPS 2025)**: First DNA-LLM integration achieving 98% accuracy on disease pathway prediction through cross-modal connectors and targeted reinforcement learning.

2. **COMICAL**: Brain imaging-genomics foundation model using CLIP-style contrastive learning on 15M IDP-SNP pairs from UK Biobank, discovering novel genetic associations with brain phenotypes.

3. **Med-Gemini-Polygenic**: Demonstrates that genomic information can be represented as "images" (polygenic risk scores projected to 2D) and processed by multimodal LLMs, outperforming traditional approaches and generalizing to unseen diseases.

### 8.2 Immediate Actionable Strategy for AI-CoScientist

**Week 1-2: Foundation Model Ecosystem Setup**
- Deploy GenomeOcean (4B) for genomic analysis
- Integrate BiomedCLIP for multimodal literature search
- Set up ESM-2 (650M) for protein analysis
- Establish evaluation framework with existing benchmarks

**Week 3-4: Data Collection and Curation**
- Access UK Biobank or equivalent brain imaging-genomics datasets
- Extract reasoning chains from developmental disorder literature
- Create initial paired dataset (brain features + genetic variants)
- Annotate 100-1000 reasoning examples with domain experts

**Month 2: Proof-of-Concept COMICAL-Style Model**
- Train brain-genomics contrastive model on collected data
- Validate on known genetic associations
- Demonstrate cross-modal retrieval capabilities
- Evaluate against published baselines

**Month 3-4: BioReason-Style Reasoning Enhancement**
- Implement DNA/genomics → LLM connector
- Fine-tune on reasoning chains
- Evaluate on pathway prediction tasks
- Compare with BioReason's 98% benchmark

**Month 5-6: System Integration and Deployment**
- Integrate foundation models into existing AI-CoScientist pipeline
- Enhance RAG system with multimodal retrievers
- Deploy reasoning-enhanced query system
- Collect expert feedback and iterate

### 8.3 Expected Outcomes

**Short-term (3 months)**:
- Multimodal retrieval system combining literature, genomics, imaging
- Cross-modal similarity search for hypothesis generation
- Enhanced scientific understanding through foundation model reasoning

**Medium-term (6 months)**:
- Brain-genomics foundation model for developmental disorders
- Automated reasoning over genetic-brain-clinical pathways
- Novel hypothesis generation validated by domain experts

**Long-term (12 months)**:
- Autonomous scientific discovery agent
- Continual learning from new research
- State-of-the-art performance on developmental disorder analysis
- Published research demonstrating novel discoveries

### 8.4 Competitive Advantages

AI-CoScientist is well-positioned to leverage these breakthroughs:

1. **Existing RAG Infrastructure**: Can be enhanced with foundation model retrievers
2. **Multi-Agent System**: Natural fit for multimodal reasoning
3. **Domain Focus**: Developmental disorders is under-explored in AI research
4. **Comprehensive Pipeline**: From literature review to hypothesis generation to experiment design

### 8.5 Final Recommendation

**Prioritize BioReason-style DNA-LLM integration** as the most impactful near-term goal:
- Clear technical approach validated at NeurIPS 2025
- Achieves 98% accuracy on pathway prediction
- Enables natural language reasoning over genomic data
- Directly applicable to developmental disorder genetics
- Feasible with 1-7B parameter models on available hardware

**Success Criteria**:
- Match or exceed BioReason's 98% accuracy on pathway prediction
- Generate interpretable reasoning chains validated by experts
- Discover at least one novel genetic-brain association
- Publish findings at top-tier conference (ICLR, NeurIPS, Nature Methods)

---

## References and Sources

### Core Foundation Models

1. [Evolutionary Scale - ESM3: Simulating 500 million years of evolution](https://www.evolutionaryscale.ai/blog/esm3-release)
2. [NVIDIA Blog - EvolutionaryScale ESM3 Generative AI](https://blogs.nvidia.com/blog/evolutionaryscale-esm3-generative-ai-nim-bionemo-h100/)
3. [GitHub - evolutionaryscale/esm](https://github.com/evolutionaryscale/esm)
4. [Nature Biotechnology - ProTrek Trimodal Protein Language Model](https://www.nature.com/articles/s41587-025-02836-0)
5. [Joint Genome Institute - GenomeOcean](https://jgi.doe.gov/user-science/science-stories/genomeocean-leverages-ai-decode-natures-secret-language)

### Brain-Genomics Integration

6. [medRxiv - COMICAL: Multimodal Foundation Model for Brain Imaging Genomics](https://www.medrxiv.org/content/10.1101/2024.11.02.24316653v1)
7. [Oxford Academic - COMICAL Foundation Model](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf196/8233690)
8. [medRxiv - epiBrainLLM for Alzheimer's](https://www.medrxiv.org/content/10.1101/2024.10.03.24314824v1)
9. [Nature Communications - GIANT Genetically Informed Brain Atlas](https://www.nature.com/articles/s41467-025-57636-6)
10. [MIT Press - Multimodal neuroimaging and GWAS](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00580/128949/)

### DNA-LLM Integration

11. [arXiv - BioReason: DNA-LLM Model](https://arxiv.org/abs/2505.23579)
12. [NeurIPS 2025 - BioReason Poster](https://neurips.cc/virtual/2025/poster/116227)
13. [GitHub - bowang-lab/BioReason](https://github.com/bowang-lab/BioReason)
14. [Frontiers - Gene-LLMs Survey](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2025.1634882/full)

### Multimodal Biomedical AI

15. [Google Research - Advancing Medical AI with Med-Gemini](https://research.google/blog/advancing-medical-ai-with-med-gemini/)
16. [arXiv - Med-Gemini Multimodal Medical Capabilities](https://arxiv.org/abs/2405.03162)
17. [NEJM AI - Multimodal Biomedical Foundation Model](https://ai.nejm.org/doi/abs/10.1056/AIoa2400640)
18. [PMC - Multimodal Foundation Models in Medical Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC12060978/)

### Conference Proceedings and Workshops

19. [NeurIPS 2025 - Foundations of Reasoning Workshop](https://neurips.cc/virtual/2025/workshop/109559)
20. [The 1st Workshop on Efficient Reasoning](https://efficient-reasoning.github.io/)
21. [ICLR 2025 - MLGenX Workshop](https://mlgenx.github.io/)
22. [ICLR 2025 - Foundation Models in the Wild](https://fm-wild-community.github.io/)
23. [SCI-FM ICLR 2025 Workshop](https://open-foundation-model.github.io/)

### Nature Machine Intelligence & Methods (2025)

24. [Nature Machine Intelligence - LucaOne](https://www.nature.com/articles/s42256-025-01044-4)
25. [Nature Methods - Multimodal Foundation Transformer Models](https://www.nature.com/articles/s41592-025-02918-6)
26. [Nature Machine Intelligence - Densing Law](https://www.nature.com/articles/s42256-025-01137-0)
27. [Nature Methods - Machine Learning Subject](https://www.nature.com/subjects/machine-learning/nmeth)

### Training Strategies and Architectures

28. [arXiv - Foundation Models for Scientific Discovery](https://arxiv.org/html/2510.15280)
29. [arXiv - The Science of Evaluating Foundation Models](https://arxiv.org/html/2502.09670v1)
30. [Apple ML Research - Foundation Models 2025 Updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
31. [arXiv - CLIMB: Multimodal Clinical Foundation Models](https://arxiv.org/html/2503.07667v2)

### Genomic Foundation Models

32. [PMC - GenomeOcean Efficient Foundation Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC11838515/)
33. [Oxford Academic - Genome Language Modeling](https://academic.oup.com/biomethods/article/10/1/bpaf022/8093260)
34. [Oxford Academic - GeneTuring Benchmark](https://academic.oup.com/bib/article/doi/10.1093/bib/bbaf492/8261762)

### Additional Resources

35. [Inside NeurIPS 2025 Analysis](https://newsletter.languagemodels.co/p/the-illustrated-neurips-2025-a-visual)
36. [Frontiers - Multimodal Data Integration for Oncology](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1408843/full)
37. [Cell - Single-cell Multiregion Epigenomic Rewiring in Alzheimer's](https://www.cell.com/cell/fulltext/S0092-8674(25)00733-0)
38. [MIT Picower Institute - Decoding Alzheimer's Complexity](https://picower.mit.edu/news/decoding-complexity-alzheimers-disease)

---

**Document Version**: 1.0
**Last Updated**: December 8, 2025
**Total Sources**: 38
**Total Sections**: 8
**Word Count**: ~12,000
