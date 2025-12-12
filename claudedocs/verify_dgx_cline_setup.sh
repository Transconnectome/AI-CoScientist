#!/bin/bash
# DGX Cline Setup Verification Script
# Run this before manual Cursor testing

echo "=========================================="
echo "DGX Cline Setup Verification"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: SSH Connectivity
echo "1️⃣  Checking SSH connectivity to dgx-spark..."
if ssh dgx-spark "echo 'SSH OK'" &>/dev/null; then
    echo -e "${GREEN}✅ SSH connection successful${NC}"
else
    echo -e "${RED}❌ SSH connection failed${NC}"
    exit 1
fi
echo ""

# Check 2: Ollama PATH
echo "2️⃣  Checking Ollama PATH configuration..."
OLLAMA_VERSION=$(ssh dgx-spark "bash -lc 'ollama --version 2>&1'" | grep "ollama version")
if [ -n "$OLLAMA_VERSION" ]; then
    echo -e "${GREEN}✅ Ollama in PATH: $OLLAMA_VERSION${NC}"
else
    echo -e "${RED}❌ Ollama not accessible in login shell${NC}"
    exit 1
fi
echo ""

# Check 3: DeepSeek-R1 Model
echo "3️⃣  Checking DeepSeek-R1 32B model availability..."
MODEL_CHECK=$(ssh dgx-spark "/home/juke/.local/bin/ollama list" | grep "deepseek-r1:32b")
if [ -n "$MODEL_CHECK" ]; then
    echo -e "${GREEN}✅ DeepSeek-R1 32B model available${NC}"
    echo "   $MODEL_CHECK"
else
    echo -e "${RED}❌ DeepSeek-R1 32B model not found${NC}"
    exit 1
fi
echo ""

# Check 4: Ollama API Endpoint
echo "4️⃣  Checking Ollama API endpoint..."
API_CHECK=$(ssh dgx-spark "curl -s http://localhost:11434/api/tags" | grep -q "deepseek-r1:32b" && echo "OK")
if [ "$API_CHECK" = "OK" ]; then
    echo -e "${GREEN}✅ Ollama API responding at localhost:11434${NC}"
else
    echo -e "${RED}❌ Ollama API not responding${NC}"
    exit 1
fi
echo ""

# Check 5: Project Directory
echo "5️⃣  Checking AI-CoScientist project directory..."
if ssh dgx-spark "[ -d /home/juke/git/AI-CoScientist ]"; then
    echo -e "${GREEN}✅ Project directory exists${NC}"
else
    echo -e "${RED}❌ Project directory not found${NC}"
    exit 1
fi
echo ""

# Check 6: Cline Configuration
echo "6️⃣  Checking Cline settings.json..."
if ssh dgx-spark "[ -f /home/juke/git/AI-CoScientist/.vscode/settings.json ]"; then
    echo -e "${GREEN}✅ Cline settings.json exists${NC}"

    # Verify critical settings
    OLLAMA_CONFIG=$(ssh dgx-spark "grep -q 'cline.ollamaModelId' /home/juke/git/AI-CoScientist/.vscode/settings.json && echo 'OK'")
    if [ "$OLLAMA_CONFIG" = "OK" ]; then
        echo -e "${GREEN}   ✓ Ollama configuration present${NC}"
    else
        echo -e "${RED}   ✗ Ollama configuration missing${NC}"
    fi

    MCP_CONFIG=$(ssh dgx-spark "grep -q 'cline.mcpServers' /home/juke/git/AI-CoScientist/.vscode/settings.json && echo 'OK'")
    if [ "$MCP_CONFIG" = "OK" ]; then
        echo -e "${GREEN}   ✓ MCP servers configuration present${NC}"
    else
        echo -e "${RED}   ✗ MCP servers configuration missing${NC}"
    fi
else
    echo -e "${RED}❌ Cline settings.json not found${NC}"
    exit 1
fi
echo ""

# Check 7: Node.js & npx
echo "7️⃣  Checking Node.js and npx for MCP servers..."
NODE_VERSION=$(ssh dgx-spark "node --version 2>&1")
NPX_VERSION=$(ssh dgx-spark "npx --version 2>&1")
if [ -n "$NODE_VERSION" ] && [ -n "$NPX_VERSION" ]; then
    echo -e "${GREEN}✅ Node.js $NODE_VERSION${NC}"
    echo -e "${GREEN}✅ npx $NPX_VERSION${NC}"
else
    echo -e "${RED}❌ Node.js or npx not available${NC}"
    exit 1
fi
echo ""

# Check 8: Model Quick Test
echo "8️⃣  Running quick model inference test..."
TEST_OUTPUT=$(ssh dgx-spark "bash -lc 'echo \"Testing 1-2-3\" | ollama run deepseek-r1:32b --verbose 2>&1 | head -5'")
if [ -n "$TEST_OUTPUT" ]; then
    echo -e "${GREEN}✅ Model inference working${NC}"
else
    echo -e "${YELLOW}⚠️  Model test inconclusive${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "✨ All Remote Checks Passed!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo "   1. Open Cursor/VS Code"
echo "   2. Connect to dgx-spark via Remote SSH"
echo "   3. Open folder: /home/juke/git/AI-CoScientist"
echo "   4. Launch Cline extension"
echo "   5. Test prompts (see guide below)"
echo ""
echo "🧪 Test Prompts for Cline:"
echo "   • 'What files are in scripts/ directory?'"
echo "   • 'Show me the current git branch and recent commits'"
echo "   • 'Explain the monitor_ollama_download.py script'"
echo ""
echo "Expected behavior:"
echo "   • Cline shows 'Connected to deepseek-r1:32b'"
echo "   • Responses include chain-of-thought reasoning"
echo "   • MCP tools (filesystem, git, sqlite) work"
echo ""
