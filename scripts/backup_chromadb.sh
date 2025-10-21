#!/bin/bash
# ChromaDB Backup Script
# Protects critical RAG embeddings from accidental deletion

set -e

CHROMADB_DIR="./chromadb_data"
BACKUP_DIR="./chromadb_backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="chromadb_backup_${TIMESTAMP}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔒 ChromaDB Backup System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if ChromaDB directory exists
if [ ! -d "$CHROMADB_DIR" ]; then
    echo "❌ Error: ChromaDB directory not found at $CHROMADB_DIR"
    exit 1
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Get ChromaDB size
CHROMADB_SIZE=$(du -sh "$CHROMADB_DIR" | cut -f1)
echo "📊 ChromaDB size: $CHROMADB_SIZE"

# Count documents in ChromaDB
if command -v python3 &> /dev/null; then
    DOC_COUNT=$(python3 -c "
import chromadb
from chromadb.config import Settings
try:
    client = chromadb.PersistentClient(
        path='$CHROMADB_DIR',
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    collections = client.list_collections()
    total = sum(c.count() for c in collections)
    print(total)
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
    echo "📄 Documents in ChromaDB: $DOC_COUNT"
fi

# Create backup
echo ""
echo "💾 Creating backup: $BACKUP_NAME"
echo "   Source: $CHROMADB_DIR"
echo "   Destination: $BACKUP_DIR/$BACKUP_NAME.tar.gz"

tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" "$CHROMADB_DIR"

BACKUP_SIZE=$(du -sh "$BACKUP_DIR/$BACKUP_NAME.tar.gz" | cut -f1)
echo "✅ Backup created successfully: $BACKUP_SIZE"

# List existing backups
echo ""
echo "📚 Existing backups:"
ls -lh "$BACKUP_DIR" | tail -n +2 | awk '{print "   " $9 " (" $5 ")"}'

# Cleanup old backups (keep last 5)
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR" | wc -l | tr -d ' ')
if [ "$BACKUP_COUNT" -gt 5 ]; then
    echo ""
    echo "🧹 Cleaning old backups (keeping last 5)..."
    cd "$BACKUP_DIR"
    ls -t | tail -n +6 | xargs rm -f
    cd - > /dev/null
    echo "✅ Cleanup complete"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Backup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
