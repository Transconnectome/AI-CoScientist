# MCP Server Setup Guide

This guide explains how to set up and use the Model Context Protocol (MCP) servers provided by AI-CoScientist. These servers allow AI agents (like Claude Desktop, Cursor, or Antigravity) to directly interact with the AI-CoScientist engine for paper reviews, improvements, and academic searches.

## 🚀 Available MCP Servers

1.  **`ai-coscientist`**:
    *   **Core Engine**: Provides tools for detailed paper analysis (`analyze_paper`), section improvement (`improve_paper`), and version comparison (`compare_papers`).
    *   **Powered by**: Local python environment + LLM APIs (Anthropic/OpenAI).

2.  **`academic-search`**:
    *   **Literature Discovery**: Provides real-time search capabilities on **arXiv** and **Semantic Scholar**.
    *   **Tools**: `search_arxiv`, `search_semantic_scholar`.

3.  **`sequential-thinking`** (External):
    *   **Reasoning**: Provides a structured thinking process for complex problem solving.

## 🛠️ Installation & Setup

### 1. Prerequisites

Ensure you have the project dependencies installed:

```bash
cd AI-CoScientist
poetry install
```

### 2. Configure MCP Client

You need to tell your MCP client (Claude Desktop, Cursor, etc.) where to find these servers.

#### Option A: Using `mcp-config.json` (Recommended)

Copy the `mcp-config.json` from the root of this repository to your MCP client's configuration directory.

**Claude Desktop Configuration Location:**
*   **MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
*   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
*   **Linux**: `~/.config/claude-desktop/config.json` (if applicable)

**Configuration Content:**

```json
{
  "mcpServers": {
    "academic-search": {
      "command": "poetry",
      "args": ["run", "python", "-m", "src.mcp.academic_search"],
      "cwd": "/absolute/path/to/AI-CoScientist",
      "env": {}
    },
    "ai-coscientist": {
      "command": "poetry",
      "args": ["run", "python", "-m", "src.mcp.server"],
      "cwd": "/absolute/path/to/AI-CoScientist",
      "env": {
        "ANTHROPIC_API_KEY": "your_api_key_here"
      }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
      "disabled": false
    }
  }
}
```

> **Note**: Replace `/absolute/path/to/AI-CoScientist` with the full path on your machine (e.g., `/home/username/git/AI-CoScientist`).

### 3. Verify Installation

Restart your MCP Client (e.g., Claude Desktop). You should see the following tools available:

*   `analyze_paper`
*   `improve_paper`
*   `search_arxiv`
*   `search_semantic_scholar`

## 💡 Usage Examples

### Semantic Literature Search
"Find recent papers on 'Graph Neural Networks for Brain Connectomics' from arXiv and summarize the top 3 results."

### Deep Paper Analysis
"Analyze the paper at `/path/to/paper.pdf`. Assess its methodology score and suggest 3 key improvements."

### Section Rewrite
"Rewrite the Abstract of `/path/to/draft.md` to be more concise and emphasize the 'Causal Inference' aspect."

---
**Troubleshooting**:
- If tools don't appear, check the logs of your MCP client.
- Ensure `poetry install` ran successfully and all paths in the config JSON are absolute.
