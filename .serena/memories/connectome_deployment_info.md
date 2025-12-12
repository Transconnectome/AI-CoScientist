# Connectome Server Deployment Information

## SSH Access
- **Alias**: `server`
- **Status**: ✅ Connection verified working
- **Command**: `ssh server`

## Deployment Target
- **Server**: Connectome (8x RTX 3090 GPUs)
- **Deployment Mode**: Hybrid (GPT-4 + Nemotron)
- **Total Services**: 11

## GPU Allocation
- GPU 1: Nemotron LLM (9B model, ~18GB VRAM)
- GPU 5: NeMo Embedder (1B model, ~4GB VRAM)  
- GPU 6: NeMo Reranker (1B model, ~4GB VRAM)

## API Keys
⚠️ REQUIRED: Configure in .env.production before deployment
- NGC_API_KEY: [Your NVIDIA NGC API key from https://org.ngc.nvidia.com/setup/api-key]
- OPENAI_API_KEY: [Your OpenAI API key from https://platform.openai.com/api-keys]
- ANTHROPIC_API_KEY: [Your Anthropic API key from https://console.anthropic.com/settings/keys]

## Deployment Files
- Script: `scripts/deploy_to_connectome_hybrid.sh`
- Compose: `docker-compose.connectome.yml`
- Environment: `.env.production` (to be created)

## Expected Resources
- Docker images: ~15GB
- Model cache: ~20GB
- Total disk: ~40GB
- Deployment time: 10-15 minutes
