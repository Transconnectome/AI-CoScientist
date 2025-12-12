# 🔒 ChromaDB Data Protection Guide

## ⚠️ CRITICAL: Do NOT Delete `chromadb_data/`

The `chromadb_data/` directory contains **irreplaceable RAG embeddings**:
- **355 documents** from papers_collection
- **~10,000+ text chunks** with OpenAI embeddings
- **Estimated value**: ~$50-100 in API costs to regenerate
- **Time to recreate**: 2-3 hours

## 📂 Protected Directories

```
chromadb_data/          # CRITICAL: Active ChromaDB storage
├── chroma.sqlite3      # Database with all embeddings
└── [collection_id]/    # Collection data and indexes
```

## 🛡️ Protection Measures

### 1. Git Ignore
✅ `chromadb_data/` is in `.gitignore` - won't be committed accidentally

### 2. Automated Backups

**Quick Backup**:
```bash
# Create immediate backup
./scripts/backup_chromadb.sh
```

**Scheduled Backups** (recommended):
```bash
# Add to crontab for daily backups at 2 AM
0 2 * * * cd /Users/jiookcha/Documents/git/AI-CoScientist && ./scripts/backup_chromadb.sh
```

**Backups are stored in**: `chromadb_backups/`

### 3. Manual Backup

```bash
# Create timestamped backup
tar -czf chromadb_backup_$(date +%Y%m%d_%H%M%S).tar.gz chromadb_data/

# Restore from backup
tar -xzf chromadb_backup_YYYYMMDD_HHMMSS.tar.gz
```

### 4. Before Risky Operations

**ALWAYS backup before**:
- Running cleanup scripts
- Testing new ingestion code
- Updating ChromaDB library
- Running database migrations
- Experimenting with embeddings

```bash
# Backup before risky operation
./scripts/backup_chromadb.sh
echo "Backup created at: $(ls -t chromadb_backups/ | head -1)"
```

## 🚨 Recovery Instructions

### If Accidentally Deleted

1. **Stop immediately** - Don't run any new operations
2. **Check backups**:
   ```bash
   ls -lh chromadb_backups/
   ```
3. **Restore latest backup**:
   ```bash
   tar -xzf chromadb_backups/chromadb_backup_YYYYMMDD_HHMMSS.tar.gz
   ```
4. **Verify restoration**:
   ```bash
   python scripts/investigate_rag_history.py
   ```

### If No Backup Available

Re-ingest from filtered documents (2-3 hours):
```bash
python scripts/ingest_all_documents.py
```

## 📊 Verification

Check ChromaDB health:
```bash
# Verify data exists
ls -lh chromadb_data/

# Count documents
python scripts/investigate_rag_history.py

# Check size
du -sh chromadb_data/
```

## 🔧 Maintenance

### Regular Checks

**Weekly**:
- ✅ Verify chromadb_data/ exists and has data
- ✅ Check backup count (should have 5 recent backups)
- ✅ Verify ChromaDB accessible via Python

**Monthly**:
- ✅ Test restore from backup
- ✅ Archive old backups to external storage
- ✅ Review ingestion logs for errors

### Backup Rotation

Script automatically keeps **last 5 backups**, deleting older ones.

To keep more backups:
1. Edit `scripts/backup_chromadb.sh`
2. Change: `tail -n +6` to `tail -n +11` (for 10 backups)

## 📝 Best Practices

1. **Never delete chromadb_data/ directly**
2. **Always backup before experiments**
3. **Keep at least 3 recent backups**
4. **Archive important backups externally**
5. **Document any ChromaDB schema changes**
6. **Test backups periodically**

## 🆘 Emergency Contacts

- **ChromaDB Docs**: https://docs.trychroma.com/
- **Backup Location**: `./chromadb_backups/`
- **Ingestion Script**: `scripts/ingest_all_documents.py`
- **Recovery Guide**: This file

---

**Last Updated**: 2025-10-16
**Current Status**: 355 documents, ~10K chunks, 2 collections (improvement_patterns, research_documents)
