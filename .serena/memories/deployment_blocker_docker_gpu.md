# Deployment Blocker: Docker GPU Runtime Issue on Node3

## Problem
Docker on node3 cannot access NVIDIA GPUs due to missing nvidia-container-runtime configuration.

## Error
```
nvidia-container-cli: initialization error: load library failed: 
libnvidia-ml.so.1: cannot open shared object file: no such file or directory
```

## Environment Status
- **Server**: Connectome node3
- **GPUs**: 8x RTX 3090 (nvidia-smi works fine)
- **Docker**: Installed (version 24.0.7)
- **NVIDIA Drivers**: Installed (GPUs visible via nvidia-smi)
- **Issue**: nvidia-docker2 / nvidia-container-toolkit not configured

## Resolution Options

### Option 1: Install nvidia-container-toolkit on node3
Requires sudo/admin access:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Option 2: Use node1 or node2
- Try deployment on different node with working Docker GPU runtime
- Requires NFS sync or direct file placement

### Option 3: Non-Docker deployment  
- Deploy services directly without Docker (not recommended)
- Would require significant reconfiguration

## Current Deployment Configuration
- Repository: ~/AI-CoScientist (on node3)
- Environment: .env.production created
- GPU Assignments: GPU 1 (Nemotron), GPU 4 (Embedder), GPU 6 (Reranker)
- SLURM script: deploy_slurm.sh ready

## Next Steps
Need user decision on:
1. Can they install nvidia-container-toolkit? (admin access needed)
2. Should we try node1 (requires fixing NFS sync)?
3. Alternative deployment approach?
