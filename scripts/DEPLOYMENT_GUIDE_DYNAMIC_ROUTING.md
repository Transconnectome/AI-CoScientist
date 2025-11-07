# Dynamic Model Routing Deployment Guide

Complete step-by-step deployment instructions for the DeepSeek-R1 + Nemotron hybrid routing system.

## 📋 Prerequisites

- ✅ dgx-spark server access via SSH
- ✅ Ollama 0.12.9+ installed
- ✅ DeepSeek-R1 32B model already running
- ✅ Python 3.8+ with pip
- ✅ 8x NVIDIA RTX 3090 GPUs (24GB VRAM each)
- ✅ Sufficient disk space (20GB+ for Nemotron model)

## 🚀 Quick Start (4-Week Deployment)

### Week 1: Nemotron Model Setup (Days 1-7)

#### Step 1: SSH to dgx-spark

```bash
ssh dgx-spark
```

#### Step 2: Download and register Nemotron model

```bash
# Execute the Week 1 setup script
cd /home/juke/git/AI-CoScientist
bash scripts/setup_nemotron_dgx.sh
```

**What this script does:**
- Downloads Nemotron-Nano-9B-v2 GGUF (14GB, ~10-30 minutes)
- Creates Ollama Modelfile with optimized parameters
- Registers model with Ollama
- Runs basic functionality tests
- Benchmarks performance

**Expected output:**
```
✅ Model registered: nemotron-nano-9b-v2
⚡ Performance: ~17.5 tokens/sec
```

#### Step 3: Verify both models are available

```bash
ollama list
```

Expected output:
```
NAME                    ID              SIZE
deepseek-r1:32b         abc123          18.5 GB
nemotron-nano-9b-v2     def456          14.0 GB
```

---

### Week 2: Model Router Proxy Setup (Days 8-14)

#### Step 1: Install Python dependencies

```bash
cd /home/juke/git/AI-CoScientist
pip install fastapi uvicorn httpx
```

#### Step 2: Test the model router proxy locally

```bash
# Run proxy in terminal (test mode)
python3 scripts/model_router_proxy.py
```

You should see:
```
🚀 Model Router Proxy Starting
📍 Proxy: http://localhost:11435
🎯 Ollama: http://localhost:11434
```

Test with curl in another terminal:
```bash
curl http://localhost:11435/health
# Expected: {"status":"healthy","service":"model-router-proxy"}
```

#### Step 3: Install as systemd service

```bash
# Copy service file
sudo cp scripts/model-router.service /etc/systemd/system/

# Create log directory
mkdir -p /home/juke/git/AI-CoScientist/logs

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable model-router
sudo systemctl start model-router

# Check status
sudo systemctl status model-router
```

**Expected output:**
```
● model-router.service - Model Router Proxy for Cline
   Active: active (running) since [timestamp]
```

#### Step 4: Update Cline configuration

**On your local machine (MacBook):**

```bash
# Backup current settings
cp ~/.vscode-server/data/Machine/settings.json ~/.vscode-server/data/Machine/settings.json.backup

# Update settings with routing configuration
# Use the settings_enhanced.json template from scripts/
```

**Key changes in settings.json:**

```json
{
  "cline.ollamaBaseUrl": "http://localhost:11435",  // Changed from 11434 to 11435
  "cline.modelRouting": {
    "enabled": true,
    "policy": "balanced",
    // ... (see scripts/settings_enhanced.json for full config)
  }
}
```

**Via VS Code Remote SSH:**
1. Connect to dgx-spark via Remote SSH
2. Open Settings (⌘,)
3. Search for "cline.ollamaBaseUrl"
4. Change from `http://localhost:11434` to `http://localhost:11435`
5. Reload window

---

### Week 3: Validation and Testing (Days 15-21)

#### Step 1: Install test dependencies

```bash
cd /home/juke/git/AI-CoScientist
pip install httpx
```

#### Step 2: Run validation test suite

```bash
python3 scripts/validate_routing.py
```

**Expected output:**
```
🚀 Starting Dynamic Model Routing Validation

HEALTH CHECK TESTS
✅ Proxy health check: OK
✅ Ollama health check: OK (2 models)

DYNAMIC MODEL ROUTING VALIDATION TEST SUITE
✅ PASSED: Simple Code Completion → nemotron-nano-9b-v2
✅ PASSED: Debugging → deepseek-r1:32b
✅ PASSED: Architecture → deepseek-r1:32b

TEST REPORT
Total Tests: 8
✅ Passed: 8
❌ Failed: 0

🎉 ALL TESTS PASSED!
```

#### Step 3: Manual testing with Cline

**Test simple task (should use Nemotron):**
```
Prompt in Cline: "def add(a, b):"
Expected: Fast response (~2-3 sec), uses Nemotron
```

**Test complex task (should use DeepSeek-R1):**
```
Prompt in Cline: "Debug this authentication flow that has session expiration issues"
Expected: Slower but thorough response (~10-15 sec), uses DeepSeek-R1
```

#### Step 4: Check routing logs

```bash
# View routing decisions
tail -f /home/juke/git/AI-CoScientist/logs/model_router.log

# You should see lines like:
# 2025-11-08 10:30:15 - INFO - Routing decision: {"selected_model": "nemotron-nano-9b-v2", "task_type": "code_completion", ...}
```

---

### Week 4: Performance Monitoring (Days 22-30)

#### Step 1: Run performance monitoring dashboard

```bash
python3 scripts/monitor_routing_performance.py
```

**Expected output:**
```
MODEL ROUTING PERFORMANCE DASHBOARD
================================================================================
📊 Total Routing Decisions: 127
⏰ Time Range: Last 24 hours

🤖 MODEL USAGE DISTRIBUTION
nemotron-nano-9b-v2          85 (66.9%) ████████████████████████████████████
deepseek-r1:32b              42 (33.1%) ████████████████

⚡ PERFORMANCE METRICS
Baseline (DeepSeek-R1 only):  4.5 tokens/sec
Current (Hybrid routing):     13.2 tokens/sec
Performance Improvement:      +193.3%
```

#### Step 2: Monitor GPU utilization

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Expected: Both models should be loaded, GPU memory ~32GB used
```

#### Step 3: Set up continuous monitoring (optional)

```bash
# Add to crontab for hourly monitoring
crontab -e

# Add this line:
0 * * * * /usr/bin/python3 /home/juke/git/AI-CoScientist/scripts/monitor_routing_performance.py >> /home/juke/git/AI-CoScientist/logs/monitoring_cron.log 2>&1
```

---

## 🔧 Configuration Options

### Routing Policies

You can switch policies by editing `~/.vscode-server/data/Machine/settings.json`:

**1. Conservative (prefer DeepSeek-R1):**
```json
"cline.modelRouting": {
  "policy": "conservative",
  // Only uses Nemotron for trivial tasks
}
```

**2. Balanced (recommended):**
```json
"cline.modelRouting": {
  "policy": "balanced",
  // 70% Nemotron, 30% DeepSeek-R1
}
```

**3. Aggressive (maximize speed):**
```json
"cline.modelRouting": {
  "policy": "aggressive",
  // 85% Nemotron, 15% DeepSeek-R1
}
```

**4. Single model mode:**
```json
"cline.modelRouting": {
  "policy": "deepseek_only",  // or "nemotron_only"
}
```

### Manual Model Override

Force a specific model for all requests:

```json
"cline.modelRouting": {
  "enabled": true,
  "manualOverride": "deepseek-r1:32b"  // or "nemotron-nano-9b-v2"
}
```

---

## 📊 Expected Performance Improvements

### Before (DeepSeek-R1 only)

| Metric | Value |
|--------|-------|
| Avg Response Speed | 4.5 tokens/sec |
| Simple Task Time | 30 seconds |
| Complex Task Time | 90 seconds |
| GPU Utilization | 12.5% (1/8 GPUs) |

### After (Hybrid Routing)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Avg Response Speed | 13-15 tokens/sec | +330% |
| Simple Task Time | 5-8 seconds | +375% |
| Complex Task Time | 90 seconds | same |
| GPU Utilization | 25% (2/8 GPUs) | +100% |

---

## 🐛 Troubleshooting

### Issue 1: Proxy won't start

```bash
# Check if port 11435 is in use
sudo lsof -i :11435

# If occupied, kill the process
sudo kill -9 <PID>

# Restart proxy
sudo systemctl restart model-router
```

### Issue 2: Models not loading

```bash
# Check Ollama is running
sudo systemctl status ollama

# Verify models are registered
ollama list

# Re-register model if needed
bash scripts/setup_nemotron_dgx.sh
```

### Issue 3: Cline not using routing

```bash
# Verify Cline is pointing to proxy (port 11435, not 11434)
cat ~/.vscode-server/data/Machine/settings.json | grep ollamaBaseUrl

# Should show: "http://localhost:11435"

# Check proxy logs
tail -50 /home/juke/git/AI-CoScientist/logs/model_router.log
```

### Issue 4: Slow performance

```bash
# Check GPU utilization
nvidia-smi

# Check which model is being used
tail -f /home/juke/git/AI-CoScientist/logs/model_router.log

# If always using DeepSeek-R1, switch to "aggressive" policy
```

### Issue 5: Out of GPU memory

```bash
# Check VRAM usage
nvidia-smi

# If >90% on any GPU, restart Ollama
sudo systemctl restart ollama

# Wait 30 seconds for models to reload
```

---

## 🔄 Rollback Instructions

### To disable routing (use DeepSeek-R1 only)

```bash
# Stop proxy
sudo systemctl stop model-router

# Update Cline config
# Change: "cline.ollamaBaseUrl": "http://localhost:11434"  # Back to direct Ollama
```

### To completely remove Nemotron

```bash
# Remove model
ollama rm nemotron-nano-9b-v2

# Remove GGUF file
rm /home/juke/models/nemotron-nano-9b-v2-q8_0.gguf

# Disable proxy service
sudo systemctl disable model-router
sudo systemctl stop model-router
```

---

## 📝 Files Created

| File | Purpose | Location |
|------|---------|----------|
| `setup_nemotron_dgx.sh` | Week 1 setup script | scripts/ |
| `model_router_proxy.py` | Week 2 FastAPI proxy | scripts/ |
| `model-router.service` | Week 2 systemd config | scripts/ |
| `settings_enhanced.json` | Week 2 Cline config template | scripts/ |
| `validate_routing.py` | Week 3 test suite | scripts/ |
| `monitor_routing_performance.py` | Week 4 monitoring | scripts/ |

---

## 🎯 Success Criteria

- ✅ Both models registered in Ollama
- ✅ Proxy service running and healthy
- ✅ All validation tests passing
- ✅ Cline successfully routing requests
- ✅ Performance improvement >150%
- ✅ No GPU memory issues
- ✅ Logs showing correct model selection

---

## 📞 Support

If you encounter issues:

1. Check logs: `/home/juke/git/AI-CoScientist/logs/model_router.log`
2. Verify service status: `sudo systemctl status model-router`
3. Run health checks: `python3 scripts/validate_routing.py`
4. Review this guide's troubleshooting section

---

**Deployment Version:** 1.0
**Last Updated:** 2025-11-08
**Tested On:** dgx-spark (Ubuntu 20.04, Ollama 0.12.9, 8x RTX 3090)
