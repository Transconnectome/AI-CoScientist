# Dynamic Model Routing - Quick Implementation Guide

**For**: AI-CoScientist DGX Cline Setup
**Based on**: `DYNAMIC_MODEL_ROUTING_DESIGN_20251108.md`
**Status**: Ready to Implement
**Timeline**: 4 weeks

---

## 🚀 Quick Start

### Prerequisites Check
```bash
# Verify current setup
ssh dgx-spark
ollama list  # Should show: deepseek-r1:32b (18.49 GB)
nvidia-smi   # Verify 8x RTX 3090, GPU #1 in use
cd /home/juke/git/AI-CoScientist
cat .vscode/settings.json | jq .cline
```

### Expected Output
```json
{
  "apiProvider": "ollama",
  "ollamaModelId": "deepseek-r1:32b",
  "ollamaBaseUrl": "http://localhost:11434"
}
```

---

## 📅 Week 1: Foundation Setup

### Day 1: Download Nemotron Model

```bash
# 1. SSH to dgx-spark
ssh dgx-spark

# 2. Create models directory if not exists
mkdir -p /home/juke/models
cd /home/juke/models

# 3. Download Nemotron Nano 9B GGUF (Q8 quantization, 14GB)
wget https://huggingface.co/DevQuasar/nvidia.Nemotron-Nano-9B-v2-GGUF/resolve/main/nemotron-nano-9b-v2-q8_0.gguf

# Expected download time:
# - 100 Mbps: ~18 minutes
# - 1 Gbps: ~2 minutes
# - 10 Gbps: ~12 seconds

# 4. Verify download
ls -lh nemotron-nano-9b-v2-q8_0.gguf
# Expected: ~14 GB file
```

### Day 2: Register Model with Ollama

```bash
# 1. Create Modelfile
cat > /tmp/nemotron-nano-9b.modelfile <<'EOF'
FROM /home/juke/models/nemotron-nano-9b-v2-q8_0.gguf

# Ollama parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192

# Nemotron-specific: Thinking Budget
# 0.0 = minimal reasoning (fastest)
# 0.5 = balanced
# 1.0 = maximum reasoning (slowest)
PARAMETER thinking_budget 0.5

# System prompt
SYSTEM You are Nemotron, an efficient AI coding assistant with configurable reasoning depth. You provide fast, accurate responses for general coding tasks.
EOF

# 2. Register with Ollama
ollama create nemotron-nano-9b-v2 -f /tmp/nemotron-nano-9b.modelfile

# Expected output:
# transferring model data
# using existing layer sha256:...
# creating new layer sha256:...
# writing manifest
# success

# 3. Verify registration
ollama list

# Expected output:
# NAME                    ID              SIZE      MODIFIED
# deepseek-r1:32b         edba8017331d    18 GB     X days ago
# nemotron-nano-9b-v2     <new-id>        14 GB     1 minute ago
```

### Day 3: Test Nemotron Performance

```bash
# Test 1: Simple code completion
time ollama run nemotron-nano-9b-v2 "Write a Python function to calculate factorial"

# Expected:
# - Response time: 5-8 seconds
# - Token speed: 15-20 tokens/s
# - Quality: Correct implementation

# Test 2: Complex reasoning (compare with DeepSeek)
time ollama run nemotron-nano-9b-v2 "Debug this Python code: [paste buggy code]"
time ollama run deepseek-r1:32b "Debug this Python code: [paste buggy code]"

# Expected:
# - Nemotron: 10-15s, good analysis
# - DeepSeek: 30-60s, deeper analysis with chain-of-thought

# Test 3: GPU assignment verification
nvidia-smi

# Expected:
# GPU 0: DeepSeek-R1 (~18 GB VRAM used)
# GPU 1: Nemotron (~14 GB VRAM used) <- should be on different GPU
```

### Day 4-5: Update Configuration

```bash
# 1. Backup current config
cp .vscode/settings.json .vscode/settings.json.backup

# 2. Create enhanced configuration
cat > .vscode/settings.json <<'EOF'
{
  "cline.apiProvider": "ollama",
  "cline.ollamaBaseUrl": "http://localhost:11434",

  "cline.modelRouting": {
    "enabled": true,
    "policy": "balanced",
    "manualOverride": true,

    "models": {
      "deepseek-r1:32b": {
        "displayName": "DeepSeek-R1 32B (Reasoning)",
        "description": "Complex reasoning, architecture, debugging",
        "capabilities": [
          "reasoning",
          "architecture",
          "debugging",
          "design",
          "security_audit",
          "performance_optimization"
        ],
        "performance": {
          "tokensPerSecond": 4.5,
          "loadTimeSeconds": 5,
          "vramGB": 18.5
        },
        "priority": 2,
        "enabled": true,
        "healthCheck": {
          "enabled": true,
          "intervalSeconds": 300
        }
      },

      "nemotron-nano-9b-v2": {
        "displayName": "Nemotron Nano 9B (Fast)",
        "description": "Fast general tasks, code completion, refactoring",
        "capabilities": [
          "code_completion",
          "simple_edit",
          "refactoring",
          "documentation",
          "test_writing",
          "navigation"
        ],
        "performance": {
          "tokensPerSecond": 17.5,
          "loadTimeSeconds": 2,
          "vramGB": 14
        },
        "priority": 1,
        "enabled": true,
        "thinkingBudget": 0.5,
        "healthCheck": {
          "enabled": true,
          "intervalSeconds": 300
        }
      }
    },

    "routingPolicies": {
      "balanced": {
        "description": "Intelligent routing - fast for simple, deep for complex",
        "complexityThreshold": 0.7,
        "defaultModel": "nemotron-nano-9b-v2",
        "fallbackModel": "deepseek-r1:32b",
        "taskTypeMapping": {
          "code_completion": "nemotron-nano-9b-v2",
          "simple_edit": "nemotron-nano-9b-v2",
          "formatting": "nemotron-nano-9b-v2",
          "documentation": "nemotron-nano-9b-v2",
          "refactoring": "nemotron-nano-9b-v2",
          "test_writing": "nemotron-nano-9b-v2",
          "implementation": "nemotron-nano-9b-v2",
          "debugging": "deepseek-r1:32b",
          "architecture": "deepseek-r1:32b",
          "design": "deepseek-r1:32b",
          "security_audit": "deepseek-r1:32b",
          "performance_optimization": "deepseek-r1:32b"
        }
      },

      "conservative": {
        "description": "Quality first - mostly DeepSeek",
        "complexityThreshold": 0.3,
        "defaultModel": "deepseek-r1:32b",
        "fallbackModel": "nemotron-nano-9b-v2"
      },

      "aggressive": {
        "description": "Speed first - mostly Nemotron",
        "complexityThreshold": 0.9,
        "defaultModel": "nemotron-nano-9b-v2",
        "fallbackModel": "deepseek-r1:32b"
      }
    },

    "userPreferences": {
      "rememberChoices": true,
      "learningEnabled": true,
      "explicitOverridesOnly": false
    },

    "analytics": {
      "enabled": true,
      "logDecisions": true,
      "logPath": ".vscode/cline-routing.log"
    }
  },

  "cline.mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/juke/git/AI-CoScientist"
      ]
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git",
        "--repository",
        "/home/juke/git/AI-CoScientist"
      ]
    },
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "/home/juke/git/AI-CoScientist/chromadb_data/chroma.sqlite3"
      ]
    }
  }
}
EOF

# 3. Validate JSON
cat .vscode/settings.json | jq . > /dev/null
echo "✅ Configuration valid"

# 4. Commit changes
git add .vscode/settings.json
git commit -m "feat(dgx): Add dynamic model routing configuration

- Configure DeepSeek-R1 32B for complex reasoning
- Configure Nemotron-Nano-9B for fast general tasks
- Set balanced routing policy
- Enable health monitoring and analytics"
```

### Week 1 Validation

```bash
# Run validation script
cat > /tmp/validate_week1.sh <<'EOF'
#!/bin/bash
echo "=== Week 1 Validation ==="

# Check 1: Both models registered
echo "1. Checking models..."
ollama list | grep -E "deepseek-r1:32b|nemotron-nano-9b-v2"

# Check 2: Configuration valid
echo "2. Checking configuration..."
jq . /home/juke/git/AI-CoScientist/.vscode/settings.json > /dev/null && echo "✅ Config valid"

# Check 3: Both models respond
echo "3. Testing DeepSeek-R1..."
time ollama run deepseek-r1:32b "Hello" --max-tokens 10

echo "4. Testing Nemotron..."
time ollama run nemotron-nano-9b-v2 "Hello" --max-tokens 10

# Check 4: GPU assignments
echo "5. GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=== Validation Complete ==="
EOF

chmod +x /tmp/validate_week1.sh
bash /tmp/validate_week1.sh
```

**Week 1 Success Criteria**:
- ✅ Nemotron model downloaded (14GB)
- ✅ Both models registered in Ollama
- ✅ Performance verified (Nemotron: 15-20 tok/s, DeepSeek: 4-5 tok/s)
- ✅ Configuration file updated and valid
- ✅ Both models responsive via `ollama run`

---

## 📅 Week 2: Router Implementation

**Note**: This requires Cline extension modification. Two approaches:

### Approach A: Extension Modification (Recommended)
Modify Cline source code to add routing layer

### Approach B: Proxy Layer (Quick Prototype)
Create HTTP proxy between Cline and Ollama

### Approach B Implementation (Recommended for Prototyping)

#### Day 1-2: Create Routing Proxy

```python
# File: scripts/model_router_proxy.py
import asyncio
import json
from typing import Dict, Optional
from fastapi import FastAPI, Request, Response
import httpx
from dataclasses import dataclass
from enum import Enum
import re

app = FastAPI()

# Configuration (from .vscode/settings.json)
OLLAMA_BASE = "http://localhost:11434"
CONFIG_PATH = "/home/juke/git/AI-CoScientist/.vscode/settings.json"

class TaskType(Enum):
    CODE_COMPLETION = "code_completion"
    SIMPLE_EDIT = "simple_edit"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    REFACTORING = "refactoring"

def load_config() -> dict:
    """Load routing configuration"""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return config.get("cline", {}).get("modelRouting", {})

def calculate_complexity(prompt: str) -> float:
    """Calculate task complexity score (0.0-1.0)"""
    score = 0.0
    prompt_lower = prompt.lower()

    # High complexity keywords
    high_keywords = ["debug", "architecture", "design", "optimize", "security"]
    if any(kw in prompt_lower for kw in high_keywords):
        score += 0.6

    # Medium complexity keywords
    medium_keywords = ["refactor", "test", "implement"]
    if any(kw in prompt_lower for kw in medium_keywords):
        score += 0.3

    # Long prompts = more complex
    if len(prompt.split()) > 50:
        score += 0.1

    return min(score, 1.0)

def classify_task(prompt: str) -> TaskType:
    """Classify task type"""
    prompt_lower = prompt.lower()

    if any(kw in prompt_lower for kw in ["complete", "autocomplete"]):
        return TaskType.CODE_COMPLETION
    elif any(kw in prompt_lower for kw in ["debug", "fix bug", "error"]):
        return TaskType.DEBUGGING
    elif any(kw in prompt_lower for kw in ["architecture", "design"]):
        return TaskType.ARCHITECTURE
    elif any(kw in prompt_lower for kw in ["refactor", "clean"]):
        return TaskType.REFACTORING
    else:
        return TaskType.SIMPLE_EDIT

def select_model(prompt: str, config: dict) -> str:
    """Select optimal model based on prompt"""
    policy = config.get("routingPolicies", {}).get("balanced", {})
    task_mapping = policy.get("taskTypeMapping", {})

    # Classify task
    task_type = classify_task(prompt)

    # Check task type mapping
    if task_type.value in task_mapping:
        return task_mapping[task_type.value]

    # Fallback to complexity-based
    complexity = calculate_complexity(prompt)
    threshold = policy.get("complexityThreshold", 0.7)

    if complexity >= threshold:
        return "deepseek-r1:32b"
    else:
        return policy.get("defaultModel", "nemotron-nano-9b-v2")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    """Proxy all requests with model routing"""
    config = load_config()

    # Parse request body
    body = await request.body()

    # If this is a generate request, apply routing
    if path == "api/generate" and request.method == "POST":
        data = json.loads(body)
        prompt = data.get("prompt", "")

        # Select model
        selected_model = select_model(prompt, config)
        data["model"] = selected_model

        # Log decision
        print(f"🤖 Routing: {prompt[:50]}... → {selected_model}")

        # Update body
        body = json.dumps(data).encode()

    # Forward to Ollama
    async with httpx.AsyncClient() as client:
        url = f"{OLLAMA_BASE}/{path}"
        response = await client.request(
            method=request.method,
            url=url,
            content=body,
            headers=dict(request.headers),
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers)
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Model Router Proxy on http://localhost:11435")
    print("📝 Routing requests from Cline → Ollama with intelligent model selection")
    uvicorn.run(app, host="0.0.0.0", port=11435)
```

#### Day 3: Deploy Proxy

```bash
# 1. Install dependencies
pip install fastapi uvicorn httpx

# 2. Test proxy locally
python scripts/model_router_proxy.py

# Expected output:
# 🚀 Starting Model Router Proxy on http://localhost:11435
# 📝 Routing requests from Cline → Ollama

# 3. Test in another terminal
curl -X POST http://localhost:11435/api/generate \
  -d '{"model": "any", "prompt": "Debug this code"}'

# Expected:
# 🤖 Routing: Debug this code → deepseek-r1:32b

# 4. Update Cline config to use proxy
# Change in .vscode/settings.json:
"cline.ollamaBaseUrl": "http://localhost:11435"  # Changed from 11434
```

#### Day 4-5: Create systemd Service

```bash
# 1. Create service file
sudo tee /etc/systemd/system/model-router.service <<'EOF'
[Unit]
Description=Model Router Proxy for Cline
After=network.target

[Service]
Type=simple
User=juke
WorkingDirectory=/home/juke/git/AI-CoScientist
ExecStart=/usr/bin/python3 /home/juke/git/AI-CoScientist/scripts/model_router_proxy.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 2. Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable model-router
sudo systemctl start model-router

# 3. Check status
sudo systemctl status model-router

# Expected:
# ● model-router.service - Model Router Proxy for Cline
#    Active: active (running)
```

### Week 2 Validation

```bash
# Test routing with different prompts
cat > /tmp/test_routing.sh <<'EOF'
#!/bin/bash

test_routing() {
    prompt=$1
    echo "Testing: $prompt"
    curl -s -X POST http://localhost:11435/api/generate \
      -d "{\"model\": \"any\", \"prompt\": \"$prompt\"}" \
      | jq -r .model
}

echo "=== Routing Tests ==="
test_routing "Complete this function"          # Expected: nemotron
test_routing "Debug this memory leak"          # Expected: deepseek
test_routing "Design the system architecture"  # Expected: deepseek
test_routing "Refactor this code"              # Expected: nemotron
test_routing "Write unit tests"                # Expected: nemotron
EOF

bash /tmp/test_routing.sh
```

**Week 2 Success Criteria**:
- ✅ Proxy server running and stable
- ✅ Routing correctly classifies >90% of test cases
- ✅ Logs show routing decisions
- ✅ Cline can connect through proxy
- ✅ Both models accessible

---

## 📅 Week 3: User Interface (Optional)

### Simple CLI Interface

```bash
# Create model selector command
cat > ~/bin/cline-model <<'EOF'
#!/bin/bash
# Quick model selector for Cline

case "$1" in
  status)
    echo "Current routing policy: $(jq -r '.cline.modelRouting.policy' ~/.vscode/settings.json)"
    ;;
  set-policy)
    # set-policy balanced|conservative|aggressive
    jq ".cline.modelRouting.policy = \"$2\"" ~/.vscode/settings.json > /tmp/settings.json
    mv /tmp/settings.json ~/.vscode/settings.json
    echo "Policy set to: $2"
    ;;
  force-model)
    # force-model deepseek|nemotron
    echo "Temporarily forcing model: $2"
    # Implementation TBD
    ;;
  stats)
    echo "=== Routing Statistics ==="
    tail -100 .vscode/cline-routing.log | jq -s 'group_by(.selected_model) | map({model: .[0].selected_model, count: length})'
    ;;
  *)
    echo "Usage: cline-model {status|set-policy|force-model|stats}"
    ;;
esac
EOF

chmod +x ~/bin/cline-model

# Test
cline-model status
cline-model set-policy aggressive
```

### Week 3 Success Criteria (Optional)
- ✅ CLI tool for policy switching
- ✅ Basic statistics available
- ✅ Easy manual model override

---

## 📅 Week 4: Testing & Optimization

### Performance Testing

```bash
# Create benchmark script
cat > /tmp/benchmark_routing.sh <<'EOF'
#!/bin/bash

# Test 1: Simple code completion (should use Nemotron)
echo "Test 1: Code Completion"
time curl -s -X POST http://localhost:11435/api/generate \
  -d '{"model": "any", "prompt": "Complete this Python function to calculate sum"}'

# Test 2: Complex debugging (should use DeepSeek)
echo "Test 2: Debugging"
time curl -s -X POST http://localhost:11435/api/generate \
  -d '{"model": "any", "prompt": "Debug this complex memory leak in C++ with RAII"}'

# Repeat 10 times each, calculate averages
EOF

bash /tmp/benchmark_routing.sh
```

### Integration Testing

```bash
# Test with actual Cline
# 1. Open VS Code/Cursor
# 2. Connect to dgx-spark via Remote SSH
# 3. Open AI-CoScientist project
# 4. Launch Cline
# 5. Test prompts:

# Simple (should be fast, Nemotron):
"Add a comment to this function"
"Format this code"
"Rename variable foo to bar"

# Complex (should be slower, DeepSeek):
"Debug why this authentication fails"
"Design a scalable microservices architecture"
"Optimize this algorithm's performance"
```

### Week 4 Success Criteria
- ✅ Average routing overhead <10ms
- ✅ Routing accuracy >90% (matches expected model)
- ✅ No crashes or errors in 100 consecutive requests
- ✅ User can easily switch policies
- ✅ Performance targets met (see design doc)

---

## 🎯 Post-Implementation

### Monitoring

```bash
# Create monitoring script
cat > scripts/monitor_routing.sh <<'EOF'
#!/bin/bash
# Monitor routing decisions in real-time

echo "=== Live Routing Monitor ==="
tail -f .vscode/cline-routing.log | jq -r '"\(.timestamp) | \(.task_type) | \(.selected_model) | \(.complexity_score)"'
EOF

chmod +x scripts/monitor_routing.sh
./scripts/monitor_routing.sh
```

### Analytics

```bash
# Generate daily report
cat > scripts/routing_report.sh <<'EOF'
#!/bin/bash
# Generate routing analytics report

echo "=== Routing Report (Last 24h) ==="
cat .vscode/cline-routing.log | jq -s '
{
  total_requests: length,
  model_usage: group_by(.selected_model) | map({model: .[0].selected_model, count: length}),
  avg_complexity: (map(.complexity_score) | add / length),
  task_distribution: group_by(.task_type) | map({type: .[0].task_type, count: length})
}
'
EOF

chmod +x scripts/routing_report.sh
./scripts/routing_report.sh
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Proxy not routing**
```bash
# Check proxy is running
sudo systemctl status model-router

# Check Cline is pointing to proxy
jq .cline.ollamaBaseUrl .vscode/settings.json
# Should be: http://localhost:11435

# Check proxy logs
sudo journalctl -u model-router -f
```

**Issue 2: Both models use same GPU**
```bash
# Check GPU assignments
nvidia-smi

# Ollama doesn't support GPU pinning directly
# Workaround: Run two separate Ollama instances on different ports
# with CUDA_VISIBLE_DEVICES set

# Terminal 1 (GPU 0 - DeepSeek):
CUDA_VISIBLE_DEVICES=0 ollama serve --port 11434

# Terminal 2 (GPU 1 - Nemotron):
CUDA_VISIBLE_DEVICES=1 ollama serve --port 11436

# Update proxy to route to different ports
```

**Issue 3: Wrong model selected**
```bash
# Check routing logic
cat .vscode/cline-routing.log | jq 'select(.selected_model != .expected_model)'

# Adjust complexity threshold or task mapping in settings.json
```

**Issue 4: Performance not improved**
```bash
# Verify Nemotron is actually being used
tail -100 .vscode/cline-routing.log | jq -r '.selected_model' | sort | uniq -c

# Expected: 60-70% Nemotron, 30-40% DeepSeek

# If not, adjust policy threshold
```

---

## ✅ Final Checklist

**Infrastructure**:
- [ ] Nemotron model downloaded (14GB)
- [ ] Both models registered in Ollama
- [ ] Models verified working
- [ ] GPU assignments optimized

**Configuration**:
- [ ] `.vscode/settings.json` updated
- [ ] Routing policies defined
- [ ] Analytics enabled

**Implementation**:
- [ ] Router proxy deployed and running
- [ ] systemd service configured
- [ ] Logs working correctly

**Testing**:
- [ ] Routing accuracy >90%
- [ ] Performance targets met
- [ ] No errors in 100+ requests
- [ ] User can switch policies

**Documentation**:
- [ ] Design doc complete
- [ ] Implementation guide complete
- [ ] Troubleshooting guide available
- [ ] Monitoring setup documented

---

## 📞 Next Steps

1. **Week 1**: Execute foundation setup
2. **Week 2**: Implement and test proxy
3. **Week 3**: Add optional UI enhancements
4. **Week 4**: Optimize and validate

**Questions or issues?**
- Check design doc: `DYNAMIC_MODEL_ROUTING_DESIGN_20251108.md`
- Review Nemotron research: `NEMOTRON_INTEGRATION_RESEARCH_20251108.md`
- Consult DGX setup guide: `DGX_CLINE_SETUP_GUIDE.md`

**Ready to start?** → Begin with Week 1, Day 1

