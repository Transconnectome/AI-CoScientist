# Week 1 Deployment Instructions

Execute these commands step-by-step to deploy Nemotron on dgx-spark.

## 🚀 Deployment Steps

### Step 1: SSH to dgx-spark

```bash
ssh dgx-spark
```

### Step 2: Create the setup script

Copy-paste this entire block into your dgx-spark terminal:

```bash
cd /home/juke/git/AI-CoScientist
mkdir -p scripts

cat > scripts/setup_nemotron_dgx.sh << 'SCRIPT_EOF'
#!/bin/bash
# Week 1 Setup: Nemotron Model Download and Ollama Configuration
# Execute this script on dgx-spark server

set -e  # Exit on error

MODELS_DIR="/home/juke/models"
MODEL_NAME="nemotron-nano-9b-v2"
MODEL_FILE="nemotron-nano-9b-v2-q8_0.gguf"
DOWNLOAD_URL="https://huggingface.co/DevQuasar/nvidia.Nemotron-Nano-9B-v2-GGUF/resolve/main/${MODEL_FILE}"

echo "🚀 Starting Nemotron Nano 9B Setup on dgx-spark"
echo "================================================"

# Day 1: Download Model
echo ""
echo "📥 Day 1: Downloading Nemotron Model (14GB, ~10-30 min)..."
mkdir -p ${MODELS_DIR}
cd ${MODELS_DIR}

if [ -f "${MODEL_FILE}" ]; then
    echo "✅ Model file already exists: ${MODEL_FILE}"
else
    echo "⏳ Downloading from HuggingFace..."
    wget --progress=bar:force:noscroll ${DOWNLOAD_URL}
    echo "✅ Download complete: ${MODEL_FILE}"
fi

# Verify file size (should be ~14GB)
FILE_SIZE=$(du -h ${MODEL_FILE} | cut -f1)
echo "📊 Model file size: ${FILE_SIZE}"

# Day 2: Create Ollama Modelfile
echo ""
echo "📝 Day 2: Creating Ollama Modelfile..."
cat > /tmp/nemotron-nano-9b.modelfile <<EOF
FROM ${MODELS_DIR}/${MODEL_FILE}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER thinking_budget 0.5

SYSTEM You are Nemotron, an efficient AI coding assistant with configurable reasoning depth. You excel at code completion, refactoring, documentation, and fast general-purpose coding tasks.
EOF

echo "✅ Modelfile created at /tmp/nemotron-nano-9b.modelfile"

# Day 3: Register with Ollama
echo ""
echo "🔧 Day 3: Registering model with Ollama..."
ollama create ${MODEL_NAME} -f /tmp/nemotron-nano-9b.modelfile
echo "✅ Model registered: ${MODEL_NAME}"

# Verify registration
echo ""
echo "🔍 Verifying Ollama model registration..."
ollama list | grep nemotron

# Day 4: Test model
echo ""
echo "🧪 Day 4: Testing model with simple query..."
echo "def hello_world():" | ollama run ${MODEL_NAME} "Complete this Python function"

# Benchmark performance
echo ""
echo "⚡ Benchmarking performance..."
echo "Testing response speed (tokens/sec)..."
time ollama run ${MODEL_NAME} "Write a Python function to calculate factorial" >/dev/null

echo ""
echo "✅ Setup Complete!"
echo "================================================"
echo "📊 Summary:"
echo "  - Model: ${MODEL_NAME}"
echo "  - Location: ${MODELS_DIR}/${MODEL_FILE}"
echo "  - Size: ${FILE_SIZE}"
echo "  - Status: Registered and tested"
echo ""
echo "🎯 Next Steps:"
echo "  1. Run model router proxy setup (Week 2)"
echo "  2. Update Cline configuration"
echo "  3. Test dynamic routing"
SCRIPT_EOF

chmod +x scripts/setup_nemotron_dgx.sh
echo "✅ Script created and made executable"
```

### Step 3: Verify script was created

```bash
ls -lh scripts/setup_nemotron_dgx.sh
```

Expected output:
```
-rwxr-xr-x 1 juke juke 2.1K Nov  8 10:30 scripts/setup_nemotron_dgx.sh
```

### Step 4: Run the setup script

```bash
bash scripts/setup_nemotron_dgx.sh
```

---

## 📊 What to Expect

### During Download (10-30 minutes)
```
🚀 Starting Nemotron Nano 9B Setup on dgx-spark
================================================

📥 Day 1: Downloading Nemotron Model (14GB, ~10-30 min)...
⏳ Downloading from HuggingFace...

nemotron-nano-9b-v2-q8_0.gguf   12%[=====>                ] 1.7G  60.2MB/s   eta 3m 12s
```

**💡 Tip:** The download speed depends on:
- Your internet connection to dgx-spark
- HuggingFace server load
- Typical range: 20-100 MB/s

### During Registration (1-2 minutes)
```
📝 Day 2: Creating Ollama Modelfile...
✅ Modelfile created at /tmp/nemotron-nano-9b.modelfile

🔧 Day 3: Registering model with Ollama...
transferring model data
creating model layer
writing layer sha256:...
writing manifest
✅ Model registered: nemotron-nano-9b-v2
```

### During Testing (30 seconds)
```
🧪 Day 4: Testing model with simple query...
    return "Hello, World!"

⚡ Benchmarking performance...

real    0m5.234s
user    0m0.032s
sys     0m0.012s
```

### Final Summary
```
✅ Setup Complete!
================================================
📊 Summary:
  - Model: nemotron-nano-9b-v2
  - Location: /home/juke/models/nemotron-nano-9b-v2-q8_0.gguf
  - Size: 14G
  - Status: Registered and tested

🎯 Next Steps:
  1. Run model router proxy setup (Week 2)
  2. Update Cline configuration
  3. Test dynamic routing
```

---

## ✅ Verification Steps

After the script completes, verify everything worked:

### 1. Check both models are registered
```bash
ollama list
```

Expected output:
```
NAME                    ID              SIZE      MODIFIED
deepseek-r1:32b         abc123def       18.5 GB   2 weeks ago
nemotron-nano-9b-v2     def456abc       14.0 GB   1 minute ago
```

### 2. Test Nemotron directly
```bash
ollama run nemotron-nano-9b-v2 "Write a hello world function in Python"
```

Expected: Fast response (2-5 seconds) with Python code

### 3. Check GPU memory usage
```bash
nvidia-smi
```

Expected: Two models loaded, ~32GB total VRAM used

### 4. Check disk space
```bash
df -h /home/juke/models
```

Expected: ~32GB used (18.5GB DeepSeek + 14GB Nemotron)

---

## 🐛 Troubleshooting

### Issue: Download fails with "Connection timed out"
```bash
# Resume the download
cd /home/juke/models
wget -c https://huggingface.co/DevQuasar/nvidia.Nemotron-Nano-9B-v2-GGUF/resolve/main/nemotron-nano-9b-v2-q8_0.gguf

# Then re-run the script
bash /home/juke/git/AI-CoScientist/scripts/setup_nemotron_dgx.sh
```

### Issue: "ollama: command not found"
```bash
# Check if Ollama is in PATH
which ollama

# If not found, use full path
/usr/local/bin/ollama list

# Or restart your shell
exec bash
```

### Issue: Out of disk space
```bash
# Check available space
df -h /home/juke/models

# If needed, clean up old files
# (Be careful - don't delete important data!)
```

### Issue: GPU out of memory
```bash
# Check current GPU usage
nvidia-smi

# If >90% VRAM used, restart Ollama
sudo systemctl restart ollama

# Wait 30 seconds, then retry
sleep 30
bash scripts/setup_nemotron_dgx.sh
```

---

## 📞 Next Steps After Week 1

Once you see "✅ Setup Complete!", you're ready for Week 2:
1. Install Python dependencies (FastAPI, uvicorn, httpx)
2. Deploy the model router proxy
3. Configure it as a systemd service
4. Update Cline settings

Let me know when Week 1 completes and I'll guide you through Week 2!

---

**Estimated Total Time:** 15-45 minutes
- Download: 10-30 min (depends on connection speed)
- Registration: 1-2 min
- Testing: 1-2 min
- Verification: 1-5 min
