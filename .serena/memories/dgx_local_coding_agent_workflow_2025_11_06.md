# DGX Local Coding Agent Implementation Workflow - Session Memory

**Created**: 2025-11-06
**Context**: Implementation plan for deploying Cline + DeepSeek-R1 32B + MCP servers on dgx-spark

## Overview

Created comprehensive implementation workflow for deploying local coding agent on dgx-spark as alternative to Claude Code.

## Architecture Decision

**Selected Stack**:
- **IDE**: VS Code with Remote SSH to dgx-spark
- **Extension**: Cline (saoudrizwan.claude-dev)
- **Model**: DeepSeek-R1 32B via Ollama (256K context, tool-calling support)
- **MCP Servers**: filesystem, git, sqlite
- **Target**: AI-CoScientist project development

## Workflow Document

Created: `claudedocs/DGX_LOCAL_CODING_AGENT_WORKFLOW.md`

**Contents**:
- 8 implementation phases with detailed steps
- Architecture overview and component diagram
- Success criteria checklist (25 items)
- Common issues and solutions (6 major issues)
- Performance monitoring and optimization
- Quick reference guide for team

## Implementation Phases

1. **Phase 1**: Environment verification (15 min)
   - SSH connectivity, Ollama status, GPU availability
   
2. **Phase 2**: Node.js and MCP installation (30 min)
   - nvm or apt-based Node.js installation
   - Global npm packages for MCP servers
   
3. **Phase 3**: Ollama configuration (15 min)
   - Service verification, model testing
   
4. **Phase 4**: Cline extension (10 min)
   - VS Code extension installation
   
5. **Phase 5**: Remote SSH setup (20 min)
   - VS Code Remote SSH configuration
   - Project folder access
   
6. **Phase 6**: AI-CoScientist configuration (30 min)
   - `.vscode/settings.json` with MCP server paths
   - Custom instructions for project context
   
7. **Phase 7**: Integration testing (45 min)
   - Ollama connection, MCP server validation
   - Code generation tests, performance baseline
   
8. **Phase 8**: Documentation (20 min)
   - Usage guide, troubleshooting, quick reference

**Total Estimated Time**: 2-3 hours

## Key Configuration

`.vscode/settings.json` structure:
```json
{
  "cline.apiProvider": "ollama",
  "cline.ollamaModelId": "deepseek-r1:32b",
  "cline.ollamaBaseUrl": "http://localhost:11434",
  "cline.mcpServers": {
    "filesystem": {...},
    "git": {...},
    "sqlite": {...}
  },
  "cline.customInstructions": "AI-CoScientist project context..."
}
```

## Advantages Over Cloud

| Feature | Local (Cline) | Cloud (Claude Code) |
|---------|---------------|---------------------|
| Cost | $0/month | $20/month |
| Privacy | 100% local | Cloud-based |
| Latency | <100ms | ~500ms |
| Offline | ✅ Yes | ❌ No |

## Success Criteria

- [x] Workflow document created (13KB, comprehensive)
- [x] 8 phases with detailed instructions
- [x] Troubleshooting section (6 common issues)
- [x] Performance monitoring guidance
- [x] Team usage guide included
- [ ] Implementation execution (pending user confirmation)

## Next Steps

1. User reviews workflow document
2. Execute Phase 1-2 (environment setup on dgx-spark)
3. Execute Phase 3-6 (Cline installation and configuration)
4. Execute Phase 7 (testing and validation)
5. Team onboarding and knowledge sharing

## References

- Cline Extension: saoudrizwan.claude-dev
- MCP Protocol: https://modelcontextprotocol.io
- DeepSeek-R1 Model: 18.49 GB already on dgx-spark
- Target Project: ~/Documents/git/AI-CoScientist

## Technical Decisions

1. **Why Cline over Aider?**: Better MCP integration, VS Code UI
2. **Why DeepSeek-R1?**: Strong reasoning, 256K context, already downloaded
3. **Why MCP Servers?**: Standardized tool protocol, extensible
4. **Why Remote SSH?**: Leverage dgx-spark GPUs, local development UX

## Monitoring Notes

- DeepSeek-R1 uses GPU 1 on dgx-spark
- Model warm-up: ~30s first request, <5s subsequent
- Expected GPU memory: ~18-20GB / 24GB
- Network latency: <10ms on campus network

## Files Created

- `claudedocs/DGX_LOCAL_CODING_AGENT_WORKFLOW.md` (13KB)
- Serena memory: This file

## Status

**Current**: Workflow planning complete, ready for implementation
**Next**: User approval → Begin Phase 1 execution
