#!/bin/bash
#
# AI-CoScientist - System Backup Script
#
# This script creates complete system backups:
# 1. PostgreSQL database
# 2. Papers collection (PDFs)
# 3. Configuration files
# 4. Application logs
#
# Usage:
#   ./scripts/backup_system.sh [full|db|papers]
#
# Examples:
#   ./scripts/backup_system.sh           # Full backup
#   ./scripts/backup_system.sh db        # Database only
#   ./scripts/backup_system.sh papers    # Papers only
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
BACKUP_DIR="backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_TYPE="${1:-full}"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create backup directory
create_backup_dir() {
    mkdir -p "$BACKUP_DIR"
    log_info "Backup directory: $BACKUP_DIR"
}

# Backup PostgreSQL database
backup_database() {
    log_info "Backing up PostgreSQL database..."

    local db_backup_file="$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

    if docker-compose exec -T postgres pg_dump -U postgres ai_coscientist | gzip > "$db_backup_file"; then
        local size=$(du -h "$db_backup_file" | cut -f1)
        log_info "✓ Database backed up: $db_backup_file ($size)"
    else
        log_error "Database backup failed"
        return 1
    fi
}

# Backup papers collection
backup_papers() {
    log_info "Backing up papers collection..."

    local papers_backup_file="$BACKUP_DIR/papers_$TIMESTAMP.tar.gz"

    if [ ! -d "papers_collection" ]; then
        log_warn "papers_collection directory not found, skipping"
        return 0
    fi

    # Count papers
    local paper_count=$(find papers_collection -name "*.pdf" | wc -l | tr -d ' ')
    log_info "Found $paper_count PDFs to backup..."

    if tar -czf "$papers_backup_file" papers_collection/ 2>/dev/null; then
        local size=$(du -h "$papers_backup_file" | cut -f1)
        log_info "✓ Papers backed up: $papers_backup_file ($size)"
    else
        log_error "Papers backup failed"
        return 1
    fi
}

# Backup configuration
backup_config() {
    log_info "Backing up configuration files..."

    local config_backup_file="$BACKUP_DIR/config_$TIMESTAMP.tar.gz"

    tar -czf "$config_backup_file" \
        .env.production \
        docker-compose.yml \
        Dockerfile \
        2>/dev/null || true

    local size=$(du -h "$config_backup_file" | cut -f1)
    log_info "✓ Configuration backed up: $config_backup_file ($size)"
}

# Backup logs
backup_logs() {
    log_info "Backing up recent logs (last 7 days)..."

    local logs_backup_file="$BACKUP_DIR/logs_$TIMESTAMP.tar.gz"

    if [ ! -d "logs" ]; then
        log_warn "logs directory not found, skipping"
        return 0
    fi

    # Find logs from last 7 days
    find logs/ -name "*.log" -mtime -7 -print0 | tar -czf "$logs_backup_file" --null -T - 2>/dev/null || true

    if [ -f "$logs_backup_file" ]; then
        local size=$(du -h "$logs_backup_file" | cut -f1)
        log_info "✓ Logs backed up: $logs_backup_file ($size)"
    else
        log_warn "No recent logs found"
    fi
}

# Clean old backups (keep last 7 days)
cleanup_old_backups() {
    log_info "Cleaning up old backups (keeping last 7 days)..."

    local deleted_count=0

    # Delete backups older than 7 days
    while IFS= read -r -d '' file; do
        rm -f "$file"
        ((deleted_count++))
    done < <(find "$BACKUP_DIR" -name "*.gz" -mtime +7 -print0)

    if [ $deleted_count -gt 0 ]; then
        log_info "✓ Deleted $deleted_count old backup(s)"
    else
        log_info "✓ No old backups to clean"
    fi
}

# Display backup summary
display_summary() {
    echo
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              Backup Completed Successfully                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo

    log_info "Backup Summary:"
    echo "  • Backup Type: $BACKUP_TYPE"
    echo "  • Timestamp: $TIMESTAMP"
    echo "  • Backup Directory: $BACKUP_DIR"
    echo

    log_info "Backup Files:"
    ls -lh "$BACKUP_DIR"/*"$TIMESTAMP"* 2>/dev/null || echo "  (no files created)"
    echo

    log_info "Total Backup Size:"
    du -sh "$BACKUP_DIR"
    echo

    log_info "Disk Usage:"
    df -h .
    echo

    log_info "Restore Instructions:"
    echo "  1. Stop services: docker-compose stop api celery-worker celery-beat"
    echo "  2. Restore database: gunzip < $BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz | docker-compose exec -T postgres psql -U postgres ai_coscientist"
    echo "  3. Restore papers: tar -xzf $BACKUP_DIR/papers_$TIMESTAMP.tar.gz"
    echo "  4. Restart services: docker-compose start api celery-worker celery-beat"
    echo
}

# Main backup flow
main() {
    echo
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           AI-CoScientist Backup Script                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo

    create_backup_dir

    case "$BACKUP_TYPE" in
        "full")
            log_info "Performing FULL backup..."
            backup_database
            backup_papers
            backup_config
            backup_logs
            ;;
        "db")
            log_info "Performing DATABASE backup..."
            backup_database
            ;;
        "papers")
            log_info "Performing PAPERS backup..."
            backup_papers
            ;;
        *)
            log_error "Invalid backup type: $BACKUP_TYPE"
            echo "Usage: $0 [full|db|papers]"
            exit 1
            ;;
    esac

    cleanup_old_backups
    display_summary

    log_info "🎉 Backup completed successfully!"
}

# Run main function
main
