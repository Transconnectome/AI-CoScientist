# DGX-Spark Local Coding Agent Implementation Workflow

**Objective**: Deploy Cline + DeepSeek-R1 32B + MCP servers on dgx-spark as a local coding agent with full file system access

**Target Environment**: SNU dgx-spark server (8x RTX 3090, Ollama + deepseek-r1:32b already available)

**Estimated Total Time**: 2-3 hours

---

## 🎯 Architecture Overview

```yaml
Components:
  Server_Side:
    - Ollama: DeepSeek-R1 32B (256K context)
    - MCP Servers: filesystem, git, sqlite
    - Node.js: Runtime for MCP servers

  Client_Side:
    - VS Code: Remote SSH to dgx-spark
    - Cline Extension: AI coding agent
    - Configuration: .vscode/settings.json

  Integration:
    - Cline ↔ Ollama (HTTP API)
    - Cline ↔ MCP Servers (stdio)
    - VS Code ↔ dgx-spark (SSH)
```

---

## Phase 1: Environment Verification (15 min)

### 1.1 Check dgx-spark Prerequisites

**Local Machine**:
```bash
# Test SSH connectivity
ssh dgx-spark "echo 'Connection OK'"

# Check SSH config
cat ~/.ssh/config | grep -A 5 dgx-spark
```

**On dgx-spark**:
```bash
# Login to dgx-spark
ssh dgx-spark

# Verify Ollama installation
ollama --version
# Expected: ollama version 0.x.x or higher

# Verify DeepSeek-R1 model
ollama list | grep deepseek-r1
# Expected: deepseek-r1:32b

# Check Node.js (may not exist yet)
node --version
# If not installed: proceed to Phase 2
# If installed: verify >= 18.0.0

# Check available disk space
df -h ~
# Recommended: >10GB free for MCP servers and logs

# Check GPU availability
nvidia-smi
# Verify GPU 1, 5, 6 have availability (based on your .env config)
```

**Success Criteria**:
- ✅ SSH connection to dgx-spark works
- ✅ Ollama is installed and running
- ✅ deepseek-r1:32b model is available (18.49 GB)
- ✅ Sufficient disk space (>10GB)
- ✅ GPUs accessible

---

## Phase 2: Install Node.js and MCP Servers (30 min)

### 2.1 Install Node.js on dgx-spark

**Option A: Using nvm (Recommended)**
```bash
# On dgx-spark
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Reload shell
source ~/.bashrc

# Install Node.js LTS
nvm install --lts
nvm use --lts

# Verify installation
node --version  # Expected: v20.x.x or v18.x.x
npm --version   # Expected: v10.x.x or v9.x.x
```

**Option B: Using apt (if you have sudo)**
```bash
# On dgx-spark (if sudo available)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify
node --version
npm --version
```

### 2.2 Install MCP Server Packages

```bash
# On dgx-spark
# Install official MCP servers globally
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-git
npm install -g @modelcontextprotocol/server-sqlite

# Verify installations
which mcp-server-filesystem  # Should show path in ~/.nvm/... or /usr/local/...
which mcp-server-git
which mcp-server-sqlite

# Test filesystem server (should print usage)
npx @modelcontextprotocol/server-filesystem --help
```

**Expected Output**:
```
added 47 packages in 12s
@modelcontextprotocol/server-filesystem@x.x.x
@modelcontextprotocol/server-git@x.x.x
@modelcontextprotocol/server-sqlite@x.x.x
```

**Troubleshooting**:
- If `npm install -g` fails with permission error: Use nvm (no sudo required)
- If network issues: Set npm registry mirror (optional for Korean networks)

---

## Phase 3: Configure Ollama and Verify Model (15 min)

### 3.1 Ensure Ollama is Running

```bash
# On dgx-spark
# Check if Ollama service is running
systemctl status ollama
# OR check process
ps aux | grep ollama

# If not running, start it
ollama serve &

# Verify API is accessible
curl http://localhost:11434/api/tags | jq '.'
```

### 3.2 Test DeepSeek-R1 Model

```bash
# On dgx-spark
# Quick test inference
ollama run deepseek-r1:32b "Write a Python function to calculate factorial"

# Expected: Model responds with code
# If error: Model may still be downloading, check with monitor script

# Test API endpoint (what Cline will use)
curl -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "deepseek-r1:32b",
    "prompt": "Hello",
    "stream": false
  }' | jq '.'
```

**Success Criteria**:
- ✅ Ollama service running on port 11434
- ✅ DeepSeek-R1 responds to inference requests
- ✅ API endpoint accessible via curl

---

## Phase 4: Install Cline Extension (10 min)

### 4.1 Install VS Code Extension

**Local Machine**:
```bash
# Install Cline extension
code --install-extension saoudrizwan.claude-dev

# Verify installation
code --list-extensions | grep saoudrizwan.claude-dev
# Expected: saoudrizwan.claude-dev@x.x.x
```

**Alternative: Manual Installation**
1. Open VS Code
2. Go to Extensions (Cmd+Shift+X or Ctrl+Shift+X)
3. Search for "Cline" or "Claude Dev"
4. Click Install on "Cline" by Saoud Rizwan

### 4.2 Verify Extension Capabilities

1. After installation, Cline icon should appear in VS Code sidebar
2. Click Cline icon → should open agent interface
3. Note: Don't configure yet, we'll do that in Phase 6

---

## Phase 5: Setup VS Code Remote SSH (20 min)

### 5.1 Configure Remote SSH Extension

**Local Machine**:
```bash
# Install Remote SSH extension if not already installed
code --install-extension ms-vscode-remote.remote-ssh

# Verify SSH config
cat ~/.ssh/config
```

**Expected SSH Config Entry**:
```
Host dgx-spark
    HostName [actual-hostname-or-ip]
    User [your-username]
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes
```

### 5.2 Connect to dgx-spark via Remote SSH

1. Open VS Code
2. Click Remote SSH icon (bottom-left green icon)
3. Select "Connect to Host..." → Choose "dgx-spark"
4. Wait for VS Code Server to install on dgx-spark (first time only, ~2 min)
5. Once connected, verify: Bottom-left should show "SSH: dgx-spark"

**Verify Connection**:
```bash
# In VS Code Remote Terminal (Ctrl+` or Cmd+`)
pwd
# Expected: /home/[your-username]

hostname
# Expected: dgx-spark or similar

echo $SSH_CONNECTION
# Should show connection info
```

### 5.3 Open AI-CoScientist Project

**In Remote VS Code**:
```bash
# Open AI-CoScientist project
# File → Open Folder → Navigate to:
/home/[your-username]/Documents/git/AI-CoScientist
# Or wherever your project is located

# Verify project structure
ls -la
# Expected: src/, scripts/, tests/, .env files, etc.
```

---

## Phase 6: Create AI-CoScientist Configuration (30 min)

### 6.1 Create Cline Configuration File

**In Remote VS Code on dgx-spark**:

Create `.vscode/settings.json` in AI-CoScientist project root:

```bash
# In VS Code terminal on dgx-spark
cd /home/[your-username]/Documents/git/AI-CoScientist
mkdir -p .vscode
```

**Create/Edit `.vscode/settings.json`**:

```json
{
  "cline.apiProvider": "ollama",
  "cline.ollamaModelId": "deepseek-r1:32b",
  "cline.ollamaBaseUrl": "http://localhost:11434",

  "cline.mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/[YOUR-USERNAME]/Documents/git/AI-CoScientist"
      ]
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git",
        "--repository",
        "/home/[YOUR-USERNAME]/Documents/git/AI-CoScientist"
      ]
    },
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "/home/[YOUR-USERNAME]/Documents/git/AI-CoScientist/chromadb_data/chroma.sqlite3"
      ]
    }
  },

  "cline.alwaysAllowReadOnly": true,
  "cline.maxFileLineThreshold": 10000,
  "cline.soundEnabled": false,

  "cline.customInstructions": "You are a Python and AI expert assistant working on the AI-CoScientist project. This project uses:\n- Python 3.9+ with FastAPI backend\n- Hybrid AI: GPT-4 + Claude + Nemotron (local LLM)\n- ChromaDB for vector storage\n- Docker for deployment\n- GitHub for version control\n\nKey directories:\n- src/: Core application code\n- scripts/: Utility scripts\n- tests/: Test suite\n- deployment/: Docker and deployment configs\n\nAlways:\n1. Follow existing code patterns and styles\n2. Write comprehensive docstrings\n3. Include error handling\n4. Update tests when modifying code\n5. Use type hints for all functions\n6. Follow PEP 8 style guidelines"
}
```

**Important**: Replace `[YOUR-USERNAME]` with your actual username on dgx-spark!

### 6.2 Verify Configuration Syntax

```bash
# In VS Code terminal on dgx-spark
# Validate JSON syntax
cat .vscode/settings.json | jq '.'
# Should output formatted JSON without errors

# Verify MCP server paths are correct
ls -la $(which npx)
# Should show npx binary

# Test MCP filesystem server manually
npx @modelcontextprotocol/server-filesystem --help
# Should show usage information
```

### 6.3 Additional Configuration (Optional)

**Create `.vscode/launch.json` for debugging**:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  ]
}
```

---

## Phase 7: Integration Testing and Validation (45 min)

### 7.1 Test Cline → Ollama Connection

**In VS Code on dgx-spark**:

1. Open Cline panel (click Cline icon in sidebar)
2. In Cline chat interface, type:
   ```
   Hello! Can you confirm you're connected to DeepSeek-R1 on dgx-spark?
   ```
3. **Expected Response**: Cline should respond using DeepSeek-R1 model
4. **Verify Model**: Response should be coherent and mention local execution

**If Connection Fails**:
```bash
# On dgx-spark, check Ollama logs
journalctl -u ollama -f
# OR
tail -f ~/.ollama/logs/server.log

# Verify Ollama is listening
netstat -tuln | grep 11434
# Expected: tcp 0.0.0.0:11434 LISTEN

# Test direct curl
curl http://localhost:11434/api/tags
```

### 7.2 Test MCP Filesystem Server

**In Cline chat**:
```
Can you list all Python files in the src/ directory?
```

**Expected Behavior**:
- Cline uses MCP filesystem server
- Returns list of .py files in src/
- Shows file paths relative to project root

**Manual MCP Test**:
```bash
# On dgx-spark terminal
cd /home/[your-username]/Documents/git/AI-CoScientist

# Test filesystem server directly
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | \
  npx @modelcontextprotocol/server-filesystem $(pwd)

# Should return JSON with available tools
```

### 7.3 Test MCP Git Integration

**In Cline chat**:
```
What's the current git branch? Show me recent commits.
```

**Expected Behavior**:
- Cline uses MCP git server
- Returns current branch (feature/nemotron-hybrid-integration based on git status)
- Shows recent commit history

**Manual Git Test**:
```bash
# On dgx-spark
cd /home/[your-username]/Documents/git/AI-CoScientist
git status
git log --oneline -5
```

### 7.4 Test Code Generation Capability

**In Cline chat**:
```
Create a simple Python function in scripts/test_cline.py that:
1. Takes a string parameter
2. Returns the string reversed
3. Includes a docstring and type hints
4. Follows PEP 8 style
```

**Expected Behavior**:
- Cline generates the function
- Creates file via MCP filesystem server
- Code follows Python best practices
- Asks for confirmation before writing

**Verify Output**:
```bash
# On dgx-spark
cat scripts/test_cline.py
# Should contain the generated function

# Test the function
python3 scripts/test_cline.py
```

### 7.5 Test SQLite Integration (Optional)

**If you have SQLite database in project**:
```
Query the chromadb database and show me the schema.
```

**Expected Behavior**:
- Cline uses MCP sqlite server
- Connects to database
- Returns schema information

### 7.6 Performance Baseline

**Test Response Times**:

| Test | Expected Time | Metric |
|------|---------------|--------|
| Simple query | 1-3 seconds | First token latency |
| Code generation | 5-15 seconds | Full response |
| File operations | <1 second | MCP overhead |
| Git operations | 1-2 seconds | Git command execution |

**Benchmark Test**:
```
In Cline chat: "Generate a fibonacci function with memoization"
# Time the response
```

**Compare with Claude Code**:
- **Local Advantage**: No network latency (<100ms vs ~500ms)
- **Local Disadvantage**: Slower inference (DeepSeek-R1 32B vs Claude Sonnet)
- **Overall**: Should feel responsive for typical coding tasks

---

## Phase 8: Documentation and Usage Guide (20 min)

### 8.1 Create Quick Reference

**Create `claudedocs/CLINE_USAGE_GUIDE.md`**:

```markdown
# Cline Local Coding Agent - Quick Reference

## Starting Cline

1. Connect to dgx-spark via Remote SSH in VS Code
2. Open AI-CoScientist project folder
3. Click Cline icon in sidebar
4. Start chatting with your local AI assistant!

## Capabilities

### ✅ What Cline Can Do
- Read/write files in project directory
- Execute git commands (status, log, diff)
- Query SQLite databases
- Generate code with context awareness
- Refactor and improve existing code
- Debug issues with file access
- Run Python scripts and tests

### ❌ What Cline Cannot Do
- Access files outside project directory (security)
- Execute arbitrary shell commands (use with caution)
- Modify system configurations
- Access network resources directly

## Common Workflows

### Code Generation
\`\`\`
"Create a new FastAPI endpoint in src/api/routes.py for user authentication"
\`\`\`

### Code Review
\`\`\`
"Review src/schemas/hybrid_rag.py for potential improvements"
\`\`\`

### Debugging
\`\`\`
"The test in tests/test_embeddings.py is failing. Can you help debug it?"
\`\`\`

### Git Operations
\`\`\`
"Show me what files have changed and create a commit with message 'feat: add user auth'"
\`\`\`

### Database Queries
\`\`\`
"Query the ChromaDB database and show me all collections"
\`\`\`

## Best Practices

1. **Be Specific**: Provide context and clear requirements
2. **Review Changes**: Always review code before accepting
3. **Incremental Steps**: Break complex tasks into smaller steps
4. **Use Git**: Commit frequently to track changes
5. **Test Locally**: Run tests before deploying

## Performance Tips

- **First Request**: May be slow (~30s) while model loads into GPU
- **Subsequent Requests**: Much faster (~5s) as model stays in memory
- **Context Length**: DeepSeek-R1 supports 256K tokens (very large context)
- **GPU Sharing**: Model uses GPU 1 alongside Nemotron deployment

## Troubleshooting

### Cline Not Responding
\`\`\`bash
# Check Ollama service
ssh dgx-spark
systemctl status ollama
# OR
ps aux | grep ollama
\`\`\`

### MCP Server Errors
\`\`\`bash
# Test MCP servers manually
npx @modelcontextprotocol/server-filesystem --help
npx @modelcontextprotocol/server-git --help
\`\`\`

### Slow Responses
- Check GPU memory: `nvidia-smi`
- Verify no other heavy processes on GPU 1
- Consider using smaller model if needed

## Configuration Files

- `.vscode/settings.json`: Main Cline configuration
- `~/.ssh/config`: SSH connection settings
- `~/.bashrc` or `~/.zshrc`: Node.js and npm paths

## Getting Help

1. Check Cline documentation: https://github.com/cline/cline
2. Check MCP documentation: https://modelcontextprotocol.io
3. Check DeepSeek-R1 docs: https://github.com/deepseek-ai/DeepSeek-R1

## Comparison: Cline vs Claude Code

| Feature | Cline + DeepSeek-R1 | Claude Code |
|---------|---------------------|-------------|
| Cost | $0/month | $20/month |
| Privacy | 100% local | Cloud-based |
| Latency | <100ms (local) | ~500ms (network) |
| Context | 256K tokens | 200K tokens |
| Offline | ✅ Yes | ❌ No |
| Reasoning | Strong (R1 model) | Excellent (Sonnet 4.5) |
| Speed | Moderate | Fast |

## Next Steps

- Explore more MCP servers (web-search, puppeteer, etc.)
- Customize prompts in `.vscode/settings.json`
- Create project-specific Claude skills
- Integrate with CI/CD pipeline
\`\`\`

### 8.2 Store Configuration in Serena Memory

```bash
# In local machine terminal (not dgx-spark)
# This will be done after testing
```

### 8.3 Create Backup of Configuration

```bash
# On dgx-spark
cd /home/[your-username]/Documents/git/AI-CoScientist
cp .vscode/settings.json .vscode/settings.json.backup

# Add to git (optional, but recommended)
git add .vscode/settings.json
git add claudedocs/CLINE_USAGE_GUIDE.md
git commit -m "feat: Add Cline local coding agent configuration"
```

---

## 🎯 Success Criteria Checklist

### Environment Setup
- [ ] Node.js >= 18.0.0 installed on dgx-spark
- [ ] MCP server packages installed globally
- [ ] Ollama service running and accessible
- [ ] DeepSeek-R1 32B model loaded and responsive

### VS Code Integration
- [ ] Cline extension installed in VS Code
- [ ] Remote SSH connection to dgx-spark working
- [ ] AI-CoScientist project opened remotely
- [ ] `.vscode/settings.json` configured correctly

### MCP Server Integration
- [ ] Filesystem server can list and read project files
- [ ] Git server can execute git commands
- [ ] SQLite server can query databases (if applicable)
- [ ] No permission or path errors in MCP servers

### Functional Testing
- [ ] Cline responds to simple queries (< 5s)
- [ ] Code generation produces valid Python code
- [ ] File operations work via MCP filesystem
- [ ] Git operations return expected results
- [ ] Custom instructions are followed

### Documentation
- [ ] Usage guide created and accessible
- [ ] Configuration backed up
- [ ] Troubleshooting procedures documented
- [ ] Quick reference available for team

---

## 🚨 Common Issues and Solutions

### Issue 1: "Connection to Ollama failed"
**Symptoms**: Cline shows connection error, no response from model

**Solutions**:
```bash
# On dgx-spark
# Check Ollama service
systemctl status ollama
# If not running
ollama serve &

# Verify port
netstat -tuln | grep 11434

# Test API directly
curl http://localhost:11434/api/tags
```

### Issue 2: "MCP server not found"
**Symptoms**: Filesystem/git tools not available in Cline

**Solutions**:
```bash
# On dgx-spark
# Verify npm global packages
npm list -g --depth=0 | grep modelcontextprotocol

# Reinstall if missing
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-git

# Check npx is in PATH
which npx
echo $PATH
```

### Issue 3: "Permission denied" for file operations
**Symptoms**: Cline cannot read/write files

**Solutions**:
```bash
# On dgx-spark
# Check file permissions
ls -la /home/[your-username]/Documents/git/AI-CoScientist

# Verify you own the directory
stat /home/[your-username]/Documents/git/AI-CoScientist | grep Uid

# Fix permissions if needed
chmod -R u+rw /home/[your-username]/Documents/git/AI-CoScientist
```

### Issue 4: Slow First Response (>30s)
**Symptoms**: First query takes very long, subsequent queries are fast

**Explanation**: This is NORMAL behavior
- First request loads model into GPU memory (cold start)
- Subsequent requests use cached model (warm)
- DeepSeek-R1 32B is ~18GB, takes time to load

**Optimization**:
```bash
# Keep model warm by pinging it
# Create a simple keep-alive script
cat > ~/keep_ollama_warm.sh << 'EOF'
#!/bin/bash
while true; do
  curl -s -X POST http://localhost:11434/api/generate \
    -d '{"model": "deepseek-r1:32b", "prompt": "ping", "stream": false}' \
    > /dev/null
  sleep 300  # Every 5 minutes
done
EOF

chmod +x ~/keep_ollama_warm.sh
# Run in background
nohup ~/keep_ollama_warm.sh > /dev/null 2>&1 &
```

### Issue 5: VS Code Remote SSH Disconnects
**Symptoms**: Frequent disconnections, "Connection lost" messages

**Solutions**:
```bash
# On local machine
# Add to ~/.ssh/config
Host dgx-spark
    HostName [hostname]
    User [username]
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

### Issue 6: Git Integration Not Working
**Symptoms**: Cline cannot execute git commands

**Solutions**:
```bash
# On dgx-spark
# Verify git is installed
git --version

# Check git config
cd /home/[your-username]/Documents/git/AI-CoScientist
git status  # Should work without errors

# Verify MCP git server
npx @modelcontextprotocol/server-git --help
```

---

## 📊 Performance Monitoring

### GPU Usage Monitoring
```bash
# On dgx-spark
# Monitor GPU memory for DeepSeek-R1
watch -n 1 nvidia-smi

# Expected when model loaded:
# GPU 1: ~18-20GB / 24GB used
# GPU 5: Nemotron embedder
# GPU 6: Nemotron reranker
```

### Ollama Logs
```bash
# On dgx-spark
# Follow Ollama logs
tail -f ~/.ollama/logs/server.log

# Look for:
# - Model loading messages
# - Request/response times
# - Error messages
```

### Network Latency Test
```bash
# On local machine
# Test SSH latency to dgx-spark
ping -c 10 [dgx-spark-hostname]

# Expected: <10ms on campus network
# If >50ms: Network issues, check VPN/connection
```

---

## 🔄 Workflow Comparison

### Before: Claude Code (Cloud-based)
```
User → VS Code → Internet → Anthropic API → Claude Sonnet 4.5 → Response
Latency: ~500ms network + ~2s inference = 2.5s total
Cost: $20/month subscription
Privacy: Code sent to cloud
```

### After: Cline + DeepSeek-R1 (Local)
```
User → VS Code → SSH → dgx-spark → Ollama → DeepSeek-R1 → Response
Latency: ~10ms SSH + ~3s inference = 3s total (after warm-up)
Cost: $0/month (using existing infrastructure)
Privacy: Code stays on dgx-spark server
```

### Hybrid Workflow (Recommended)
```
Simple Tasks: Cline + DeepSeek-R1 (fast, local)
Complex Tasks: Claude Code (superior reasoning)
Collaborative: Both (use Cline for editing, Claude for architecture)
```

---

## 🚀 Next Steps After Deployment

### 1. Optimize for AI-CoScientist Workflow
- Create project-specific prompts
- Add custom MCP servers for ArXiv, PubMed APIs
- Integrate with paper ingestion pipeline

### 2. Team Onboarding
- Share this guide with team members
- Create demo video of common workflows
- Set up office hours for questions

### 3. Advanced Features
- Install additional MCP servers (web-search, puppeteer)
- Create Cline presets for different tasks
- Integrate with CI/CD for automated testing

### 4. Monitor and Iterate
- Track usage patterns
- Gather feedback from team
- Optimize configuration based on real usage

---

## 📚 References

- **Cline Documentation**: https://github.com/cline/cline
- **MCP Protocol**: https://modelcontextprotocol.io
- **DeepSeek-R1**: https://github.com/deepseek-ai/DeepSeek-R1
- **Ollama API**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **VS Code Remote SSH**: https://code.visualstudio.com/docs/remote/ssh

---

**Created**: 2025-11-06
**Author**: AI-CoScientist Team
**Status**: Ready for Implementation
**Estimated Setup Time**: 2-3 hours
**Maintenance**: ~30 min/month (updates, monitoring)
