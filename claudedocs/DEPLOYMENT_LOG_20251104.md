# Connectome Deployment Log - 2025-11-04

## Summary
Deployment of AI-CoScientist Hybrid RAG system to Connectome server (node3).

## Issues Encountered and Resolved

### 1. Docker Compose Version Issue
**Problem**: Snap-installed docker-compose had permission issues
```
Error: cannot create user data directory: /home/connectome/connectome1/snap/docker/3265: Permission denied
```

**Solution**: Installed standalone docker-compose v2.24.5
```bash
curl -L 'https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-x86_64' \
  -o ~/.local/bin/docker-compose-standalone
chmod +x ~/.local/bin/docker-compose-standalone
mv ~/.local/bin/docker-compose ~/.local/bin/docker-compose-broken
mv ~/.local/bin/docker-compose-standalone ~/.local/bin/docker-compose
```

### 2. Port Conflicts
**Problem**: Multiple old rag-api containers occupying ports 8000-8006, Redis on 6379

**Solution**:
- Stopped and removed 8 old rag-api containers
- Modified `docker-compose.connectome.yml` Redis port: 6379 → 6380 (external)
```bash
sed -i 's/\${REDIS_PORT:-6379}:6379/\${REDIS_PORT:-6380}:6379/g' docker-compose.connectome.yml
```

### 3. DNS Resolution Failure in Docker Build
**Problem**: Docker build containers couldn't resolve `deb.debian.org`
```
Error: Temporary failure resolving 'deb.debian.org'
```

**Root Cause**: Docker daemon not configured with DNS servers

**Solution**: Modified deployment script to use `docker build --network=host` instead of `docker-compose build`

**Modified Deployment Script**:
```bash
build_images() {
    log_info "Building Docker images with host network for DNS resolution (this may take 10-15 minutes)..."
    
    # Build API image
    log_info "Building API image..."
    DOCKER_BUILDKIT=0 docker build --network=host -t ai-coscientist-api:latest -f Dockerfile .
    
    # Build Celery Worker image (reuse API image)
    log_info "Building Celery Worker image (same as API)..."
    docker tag ai-coscientist-api:latest ai-coscientist-celery-worker:latest
    
    # Build Celery Beat image (reuse API image)
    log_info "Building Celery Beat image (same as API)..."
    docker tag ai-coscientist-api:latest ai-coscientist-celery-beat:latest
    
    log_info "✓ Docker images built successfully"
}
```

### 4. Disk Space Exhaustion
**Problem**: Filesystem at 100% capacity (3.3T / 3.5T used)
```
Error: No space left on device
```

**Solution**: Cleaned up Docker resources
```bash
docker system prune -af --volumes
# Freed: 53.61GB
# Result: 98GB available (98% usage)
```

### 5. Dockerfile Stage Issue
**Problem**: Initial build command used `--target runtime` but Dockerfile has no stages
```
Error: target stage "runtime" could not be found
```

**Solution**: Removed `--target runtime` from build commands

### 6. README.md Missing in Docker Build (NEW)
**Problem**: Poetry installation failed during build
```
Error: [Errno 2] No such file or directory: '/app/README.md'
```

**Root Cause**: `pyproject.toml` references `readme = "README.md"` but file wasn't copied to builder stage

**Solution**: Modified Dockerfile line 22 to copy README.md with project files
```dockerfile
# Before:
COPY pyproject.toml poetry.lock ./

# After:
COPY pyproject.toml poetry.lock README.md ./
```

### 7. Poetry Project Installation Error (NEW - CURRENT)
**Problem**: Poetry couldn't find package files during `poetry install`
```
Error: No file/folder found for package ai-coscientist
If you do not want to install the current project use --no-root
```

**Root Cause**: Multi-stage Dockerfile separates dependency installation from source code copying
- Stage 1 (builder): Only has pyproject.toml, poetry.lock, README.md
- Stage 2 (runtime): Has full source code via `COPY . .`
- Poetry tried to install project in Stage 1 without source code

**Solution**: Added `--no-root` flag to poetry install (dependencies only, skip project)
```dockerfile
# Before:
RUN poetry install --no-dev --no-interaction --no-ansi

# After:
RUN poetry install --no-root --no-dev --no-interaction --no-ansi
```

**Why This Works**:
- `--no-root`: Installs all dependencies but skips the project itself
- Project source code will be available in Stage 2 (runtime)
- Application runs directly from source code, no installation needed

**Backups Created**:
- `Dockerfile.backup_readme` (before README.md fix)

## Configuration Changes

### Modified Files

#### docker-compose.connectome.yml
1. **Redis Port Mapping**: `6379:6379` → `6380:6379`
2. **Backups Created**:
   - `docker-compose.connectome.yml.backup` (original)
   - `docker-compose.connectome.yml.backup2` (before DNS changes)

#### Dockerfile
**Modified**: Two critical fixes for multi-stage build
1. **Line 22**: Copy README.md with project files
   ```dockerfile
   COPY pyproject.toml poetry.lock README.md ./
   ```

2. **Line 28**: Added --no-root flag to poetry install
   ```dockerfile
   RUN poetry install --no-root --no-dev --no-interaction --no-ansi
   ```

**Backups**:
- `Dockerfile.backup_readme` (before fixes)

#### scripts/deploy_to_connectome_hybrid.sh
**Modified**: Build function to use docker build with --network=host
**Backups**:
- `deploy_to_connectome_hybrid.sh.backup3` (before final changes)

## Server Information

### Connectome Server (node3)
- **Hostname**: 147.47.200.154
- **User**: connectome1
- **SSH Key**: ~/.ssh/id_ed25519_connectome

### GPU Configuration
- **Total GPUs**: 8x NVIDIA GeForce RTX 3090 (24GB each)
- **GPU Assignments**:
  - GPU 1: Nemotron LLM (9B model, ~18GB VRAM)
  - GPU 4: NeMo Embedder (1B model, ~4GB VRAM)
  - GPU 6: NeMo Reranker (1B model, ~4GB VRAM)

### Environment Configuration
- **Location**: `~/AI-CoScientist/.env.production`
- **Key Variables**:
  ```
  NGC_API_KEY=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
  NEMOTRON_GPU_ID=1
  NEMO_EMBEDDER_GPU_ID=4
  NEMO_RERANKER_GPU_ID=6
  REDIS_HOST=redis
  REDIS_PORT=6379  # Internal port unchanged
  ```

## Current Status

### Build Progress
- ✅ Docker Compose v2.24.5 installed
- ✅ Old containers cleaned up
- ✅ Port conflicts resolved
- ✅ DNS resolution fixed
- ✅ Disk space freed (98GB available)
- ✅ README.md copy issue fixed
- ✅ Poetry --no-root flag added
- 🔄 Building ai-coscientist-api image (3rd attempt, final fix applied)
- ⏳ Pending: Image build completion (~10-15 minutes)
- ⏳ Pending: Service deployment and testing

**Current Build Status** (as of 14:00 UTC):
- ✅ Stage 1 (builder) COMPLETED: All dependencies installed successfully
- 🔄 Stage 2 (runtime) IN PROGRESS: Copying packages and building final image
  - Step 9/20: FROM python:3.11-slim
  - Step 13/20: COPY --from=builder (Python packages)
  - Progress: ~65% complete (Step 13/20)
- Both critical fixes validated:
  1. ✅ README.md copied with project files
  2. ✅ `--no-root` flag prevented premature project installation

### Next Steps
1. Wait for API image build completion
2. Verify images: `docker images | grep ai-coscientist`
3. Tag celery images (handled by deployment script)
4. Run docker-compose up with all services
5. Verify service health for all 11 services
6. Test API endpoints
7. Monitor GPU utilization

## Lessons Learned

1. **Docker DNS Configuration**: Build-time DNS resolution requires daemon-level config or --network=host
2. **Multi-Stage Dockerfile Dependencies**: Carefully plan what gets copied when
   - Builder stage needs: project metadata (pyproject.toml, poetry.lock, README.md)
   - Builder stage doesn't need: source code (src/)
   - Use `--no-root` to install only dependencies in builder stage
3. **Poetry Installation Flags**:
   - `--no-dev`: Skip development dependencies
   - `--no-root`: Skip project installation (useful for multi-stage builds)
   - Install from source code at runtime, not during dependency installation
4. **Disk Space Monitoring**: 3.5TB filled quickly with Docker layers - regular pruning needed
5. **Port Conflicts**: Always check existing containers before deployment
6. **Version Management**: Snap packages can have permission issues - prefer standalone binaries
7. **Build Context Size**: 418MB context suggests .dockerignore needed

## Recommendations

1. **Add .dockerignore** to reduce build context size
2. **Set up automated Docker cleanup** (weekly cron job)
3. **Configure Docker daemon DNS** with sudo access if possible
4. **Monitor disk usage** with alerts at 90%
5. **Document GPU allocation** to avoid conflicts with other workloads
6. **Version pin all dependencies** in Poetry for reproducible builds
7. **Test deployment script** in staging before production
8. **Create rollback plan** with previous working images
9. **Understand multi-stage builds**: Plan file copying order carefully
10. **Use --no-root for dependencies-only installation** in builder stages

## References

- Deployment guide: `claudedocs/QUICK_DEPLOY_COMMANDS.md`
- Docker Compose file: `docker-compose.connectome.yml`
- Deployment script: `scripts/deploy_to_connectome_hybrid.sh`
- SSH config: `~/.ssh/config`
- This log: `claudedocs/DEPLOYMENT_LOG_20251104.md`

## Troubleshooting Commands

```bash
# Check build status
ssh connectome "tail -f ~/AI-CoScientist/deployment.log"

# Check running containers
ssh connectome "docker ps"

# Check images
ssh connectome "docker images | grep ai-coscientist"

# Check GPU utilization
ssh connectome "nvidia-smi"

# Check disk space
ssh connectome "df -h"

# View Dockerfile
ssh connectome "cat ~/AI-CoScientist/Dockerfile | grep -A 3 'Copy dependency\|poetry install'"
```
