# NeurIPS 2025 Papers - RAPTOR Processing Report

**Date**: December 7-8, 2025
**Processing Time**: ~14 minutes for 13 papers
**System**: AI-CoScientist DD-RAPTOR Pipeline with Multi-Provider LLM

---

## Executive Summary

Successfully processed **13 out of 14** NeurIPS 2025 papers through the DD-RAPTOR pipeline, creating a comprehensive hierarchical knowledge base with:

- **1,161 Level 0 chunks** (512-token sections with overlap)
- **53 Level 1 summaries** (section-level abstractions)
- **13 Level 2 summaries** (paper-level abstractions)
- **Total: 1,227 searchable knowledge nodes**

All processed papers are stored in:
- **JSON format**: `/home/juke/git/AI-CoScientist/data/reference_papers/neurips_2025_processed/`
- **ChromaDB**: `/home/juke/git/AI-CoScientist/chromadb_data_neurips2025/` (30 MB)

---

## Papers Processed (13/14)

### Priority 1 Papers (5/5 - 100% Success)

1. **Brain Foundation Models: A Survey on**
   - File: `2503.00580_brain_foundation_models_survey.pdf`
   - Chunks: 42 L0, 2 L1
   - Total Words: 10,308
   - Status: ✅ Success

2. **Foundation Model in Biomedicine**
   - File: `2503.02104_biomedical_foundation_model_survey.pdf`
   - Chunks: 69 L0, 5 L1
   - Total Words: 5,641
   - Status: ✅ Success

3. **MMaDA: Multimodal Large Diffusion Language Models**
   - File: `2505.15809_mmada_multimodal_diffusion.pdf`
   - Chunks: 62 L0, 2 L1
   - Total Words: 15,918
   - Status: ✅ Success

4. **Brain Imaging Foundation Models, Are We There Yet?**
   - File: `2506.13306_brain_imaging_foundation_models.pdf`
   - Chunks: 74 L0, 3 L1
   - Total Words: 18,913
   - Status: ✅ Success

5. **Foundation and Large-Scale AI Models in Neuroscience**
   - File: `2510.16658_foundation_ai_neuroscience.pdf`
   - Chunks: 106 L0, 3 L1
   - Total Words: 24,853
   - Status: ✅ Success

### Priority 2 Papers (8/9 - 89% Success)

6. **Eagle 2.5: Boosting Long-Context Post-Training**
   - File: `2504.15271_Eagle2.5.pdf`
   - Chunks: 66 L0, 4 L1
   - Total Words: 16,190
   - Status: ✅ Success

7. **ModuLM: Enabling Modular and Multimodal**
   - File: `2506.00880_ModuLM.pdf`
   - Chunks: 47 L0, 4 L1
   - Total Words: 10,699
   - Status: ✅ Success

8. **NeurIPS 2025 E2LM Competition: Early Training**
   - File: `2506.07731_E2LM.pdf`
   - Chunks: 49 L0, 2 L1
   - Total Words: 10,483
   - Status: ✅ Success

9. **Training a Scientific Reasoning Model for Chemistry**
   - File: `2506.17238_ScientificReasoningChemistry.pdf`
   - Chunks: 70 L0, 6 L1
   - Total Words: 18,226
   - Status: ✅ Success

10. **The Evolving Role of Large Language Models in Scientific Innovation**
    - File: `2507.11810_LLMsScientificInnovation.pdf`
    - Chunks: 186 L0, 5 L1
    - Total Words: 36,410
    - Status: ✅ Success

11. **Cross-Domain EEG**
    - File: `2508.15716_CrossDomainEEG.pdf`
    - Chunks: 44 L0, 4 L1
    - Total Words: 11,028
    - Status: ✅ Success

12. **A Survey of Scientific Large Language Models**
    - File: `2508.21148_ScientificLLMsSurvey.pdf` (Largest: 34 MB PDF)
    - Chunks: 267 L0, 10 L1
    - Total Words: 96,629
    - Status: ✅ Success

13. **NeurIPT: Foundation Model for Neural Interfaces**
    - File: `2510.16548_NeurIPT.pdf`
    - Chunks: 79 L0, 3 L1
    - Total Words: 17,681
    - Status: ✅ Success

### Failed Papers (1/14)

14. **PRIMT**
    - File: `2509.15607_PRIMT.pdf` (17 MB)
    - Status: ❌ Failed
    - Reason: All LLM providers failed during section parsing
    - Notes: Gemini blocked content (Finish reason: 2), Anthropic organization disabled

---

## Processing Statistics

### Chunk Distribution

| Level | Description | Count | Average per Paper |
|-------|-------------|-------|-------------------|
| L0 | Text Chunks (512 tokens) | 1,161 | 89.3 |
| L1 | Section Summaries | 53 | 4.1 |
| L2 | Paper Summaries | 13 | 1.0 |
| **Total** | **All Nodes** | **1,227** | **94.4** |

### Word Count Analysis

- **Total Words**: 292,969
- **Average per Paper**: 22,536 words
- **Largest Paper**: Scientific LLMs Survey (96,629 words)
- **Smallest Paper**: Biomedical Foundation Model (5,641 words)

### File Size Distribution

- **JSON Files**: 34 MB total
- **ChromaDB**: 30 MB
- **Average JSON per Paper**: ~2.6 MB

---

## Technical Implementation

### RAPTOR Hierarchy

The DD-RAPTOR pipeline creates a 3-level hierarchical knowledge structure:

```
Level 2 (Paper Summary)
    │
    ├─ Level 1 (Section Summaries)
    │   ├─ Abstract Summary
    │   ├─ Introduction Summary
    │   ├─ Methods Summary
    │   └─ ...
    │
    └─ Level 0 (Text Chunks)
        ├─ Chunk 1 (512 tokens)
        ├─ Chunk 2 (512 tokens, 50-token overlap)
        └─ ...
```

### LLM Provider Usage

Due to API issues, the system used OpenAI GPT-4o as the primary fallback:

- **Gemini 2.5 Pro**: ❌ Blocked content (empty response, finish reason: 2)
- **Anthropic Claude Sonnet**: ❌ Organization disabled
- **OpenAI GPT-4o**: ✅ Primary provider (all summaries)
- **DeepSeek**: ⚠️ Not needed (OpenAI succeeded)

### Embedding Model

- **Model**: SciBERT (`allenai/scibert_scivocab_uncased`)
- **Embeddings Generated**: 1,227 total
- **Dimensionality**: 768 (SciBERT standard)

---

## ChromaDB Collections

### Structure

The ChromaDB instance contains 3 collections:

| Collection | Name | Description | Documents |
|------------|------|-------------|-----------|
| L0 | `neurips_2025_L0` | Original text chunks | 1,161 |
| L1 | `neurips_2025_L1` | Section summaries | 53 |
| L2 | `neurips_2025_L2` | Paper summaries | 13 |

### Storage Location

```
/home/juke/git/AI-CoScientist/chromadb_data_neurips2025/
├── 92b93fd7-ea54-4ca3-9086-5a18ef4ef486/  (neurips_2025_L0)
├── 9527efca-ea60-499a-8cc5-f1abb3c41325/  (neurips_2025_L1)
├── bf0f76d4-5c94-4475-b941-77ad2cbcf453/  (neurips_2025_L2)
└── chroma.sqlite3  (26 MB metadata database)
```

### Metadata Fields

Each chunk includes rich metadata:
- `paper_id`: Unique identifier
- `paper_title`: Full paper title
- `section`: Section name
- `section_order`: Order in paper
- `journal`: "NeurIPS 2025"
- `year`: 2025
- `chunk_index`: Position in section
- `total_chunks`: Total chunks in section

---

## Search Capabilities

The ChromaDB knowledge base now enables:

1. **Semantic Search**: Find relevant papers by meaning, not keywords
2. **Multi-Level Retrieval**: Search at chunk, section, or paper level
3. **Contextual Queries**: Locate specific concepts across all papers
4. **Hierarchical Navigation**: Drill down from summaries to detailed chunks

### Example Queries

```python
from chromadb import PersistentClient

# Connect to ChromaDB
client = PersistentClient(path="chromadb_data_neurips2025")

# Search Level 2 (paper summaries)
collection_l2 = client.get_collection("neurips_2025_L2")
results = collection_l2.query(
    query_texts=["brain foundation models for neuroimaging"],
    n_results=5
)

# Search Level 0 (detailed chunks)
collection_l0 = client.get_collection("neurips_2025_L0")
results = collection_l0.query(
    query_texts=["transformer architecture for EEG signals"],
    n_results=10
)
```

---

## Success Metrics

### Overall Performance

- **Success Rate**: 92.9% (13/14 papers)
- **Processing Speed**: ~1.1 minutes per paper
- **Chunks per Paper**: 89.3 average
- **Quality**: All 13 papers have complete 3-level hierarchies

### Data Quality Indicators

- ✅ All embeddings successfully generated (SciBERT)
- ✅ All L0 chunks have proper metadata
- ✅ All L1 summaries linked to parent chunks
- ✅ All L2 summaries linked to section summaries
- ✅ ChromaDB integrity verified

---

## Next Steps

### Immediate Actions

1. **Retry PRIMT Paper**: Re-run with different LLM settings or manual processing
2. **Test Search Functionality**: Validate retrieval quality with sample queries
3. **Integration**: Connect ChromaDB to main AI-CoScientist RAG system

### Potential Enhancements

1. **Cross-Reference Analysis**: Link related concepts across papers
2. **Citation Extraction**: Parse and index paper citations
3. **Figure/Table Processing**: Extract and index visual elements
4. **Quality Metrics**: Implement RAGAS evaluation on retrieval

---

## Files Created

### Scripts

1. `/home/juke/git/AI-CoScientist/scripts/process_neurips_2025_papers.py`
   - Main processing pipeline
   - Multi-provider LLM integration
   - RAPTOR hierarchy construction

2. `/home/juke/git/AI-CoScientist/scripts/load_neurips_2025_chromadb.py`
   - ChromaDB collection creation
   - JSON to vector store loading

### Data Outputs

1. **JSON Files** (13 files, 34 MB total)
   - Location: `/home/juke/git/AI-CoScientist/data/reference_papers/neurips_2025_processed/`
   - Format: Complete paper data with embeddings

2. **Processing Results**
   - Location: `/home/juke/git/AI-CoScientist/data/reference_papers/neurips_2025_processed/processing_results.json`
   - Contains: Success/failure tracking, statistics

3. **ChromaDB Instance** (30 MB)
   - Location: `/home/juke/git/AI-CoScientist/chromadb_data_neurips2025/`
   - Collections: 3 (L0, L1, L2)
   - Total Documents: 1,227

---

## Conclusion

The NeurIPS 2025 paper processing pipeline successfully created a comprehensive, searchable knowledge base covering state-of-the-art research in:

- Brain foundation models and neuroimaging
- Biomedical AI and foundation models
- Scientific LLMs and reasoning
- Multimodal diffusion models
- Neural interfaces and EEG processing
- Scientific innovation with LLMs

This knowledge base is now ready for integration with the AI-CoScientist research automation system, enabling advanced literature review, hypothesis generation, and cross-domain knowledge synthesis.

**Total Processing Time**: ~14 minutes
**Total Knowledge Nodes**: 1,227
**Total Coverage**: 292,969 words across 13 cutting-edge papers

---

**Report Generated**: December 8, 2025, 02:43 UTC
