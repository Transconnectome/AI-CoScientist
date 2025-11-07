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
