#!/usr/bin/env python3
"""
Model Router Proxy for Cline Dynamic Model Selection
Week 2: FastAPI proxy that routes requests between DeepSeek-R1 and Nemotron
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# Configuration
OLLAMA_BASE = "http://localhost:11434"
CONFIG_PATH = Path("/home/juke/git/AI-CoScientist/.vscode/settings.json")
LOG_PATH = Path("/home/juke/git/AI-CoScientist/logs/model_router.log")

# Setup logging
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Router Proxy", version="1.0.0")


class TaskType(Enum):
    """Task classification types"""
    CODE_COMPLETION = "code_completion"
    SIMPLE_EDIT = "simple_edit"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    PERFORMANCE = "performance"
    SECURITY = "security"
    EXPLANATION = "explanation"
    DESIGN = "design"
    GENERAL = "general"


def load_config() -> Dict:
    """Load routing configuration from settings.json"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            return config.get("cline.modelRouting", {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()


def get_default_config() -> Dict:
    """Return default configuration if file load fails"""
    return {
        "enabled": True,
        "policy": "balanced",
        "models": {
            "deepseek-r1:32b": {
                "displayName": "DeepSeek-R1 32B (Reasoning)",
                "capabilities": ["reasoning", "architecture", "debugging", "design"],
                "priority": 2,
                "enabled": True
            },
            "nemotron-nano-9b-v2": {
                "displayName": "Nemotron Nano 9B (Fast)",
                "capabilities": ["code_completion", "refactoring", "documentation"],
                "priority": 1,
                "enabled": True
            }
        },
        "routingPolicies": {
            "balanced": {
                "complexityThreshold": 0.7,
                "defaultModel": "nemotron-nano-9b-v2",
                "fallbackModel": "deepseek-r1:32b",
                "taskTypeMapping": {
                    "code_completion": "nemotron-nano-9b-v2",
                    "simple_edit": "nemotron-nano-9b-v2",
                    "documentation": "nemotron-nano-9b-v2",
                    "refactoring": "nemotron-nano-9b-v2",
                    "explanation": "nemotron-nano-9b-v2",
                    "debugging": "deepseek-r1:32b",
                    "architecture": "deepseek-r1:32b",
                    "design": "deepseek-r1:32b",
                    "security": "deepseek-r1:32b",
                    "performance": "deepseek-r1:32b"
                }
            }
        }
    }


def calculate_complexity(prompt: str, context: Optional[Dict] = None) -> float:
    """
    Calculate task complexity score (0.0-1.0)

    Returns:
        0.0-0.3: Simple (use fast model)
        0.3-0.7: Medium (configurable)
        0.7-1.0: Complex (use reasoning model)
    """
    score = 0.0
    prompt_lower = prompt.lower()

    # High complexity keywords (+0.6)
    high_complexity_keywords = [
        "debug", "architecture", "design", "optimize", "security",
        "refactor complex", "analyze system", "performance issue",
        "race condition", "memory leak", "architectural decision"
    ]
    if any(kw in prompt_lower for kw in high_complexity_keywords):
        score += 0.6

    # Medium complexity keywords (+0.3)
    medium_complexity_keywords = [
        "refactor", "test", "implement", "create class",
        "add feature", "modify function"
    ]
    if any(kw in prompt_lower for kw in medium_complexity_keywords):
        score += 0.3

    # Low complexity indicators (no change, but logged)
    low_complexity_keywords = [
        "complete", "add comment", "format", "rename variable",
        "fix typo", "add docstring"
    ]

    # Prompt length factor (+0.1 if >50 words)
    word_count = len(prompt.split())
    if word_count > 50:
        score += 0.1

    # Context-based adjustments
    if context:
        file_count = context.get("files_count", 0)
        if file_count > 5:
            score += 0.2
        elif file_count > 10:
            score += 0.3

    # Cap at 1.0
    final_score = min(score, 1.0)

    logger.debug(f"Complexity calculation: {final_score:.2f} for prompt: {prompt[:50]}...")
    return final_score


def classify_task(prompt: str) -> TaskType:
    """Classify task type based on prompt content"""
    prompt_lower = prompt.lower()

    # Task type detection patterns
    patterns = {
        TaskType.CODE_COMPLETION: ["complete", "autocomplete", "finish this"],
        TaskType.DEBUGGING: ["debug", "fix bug", "error", "not working", "fails"],
        TaskType.ARCHITECTURE: ["architecture", "system design", "structure"],
        TaskType.DESIGN: ["design", "plan", "approach"],
        TaskType.REFACTORING: ["refactor", "clean up", "reorganize"],
        TaskType.DOCUMENTATION: ["document", "docstring", "comment", "explain code"],
        TaskType.CODE_REVIEW: ["review", "check this code", "is this correct"],
        TaskType.TESTING: ["test", "unit test", "integration test"],
        TaskType.PERFORMANCE: ["optimize", "performance", "faster", "slow"],
        TaskType.SECURITY: ["security", "vulnerability", "secure", "authentication"],
        TaskType.EXPLANATION: ["explain", "what does", "how does"],
    }

    for task_type, keywords in patterns.items():
        if any(kw in prompt_lower for kw in keywords):
            logger.debug(f"Task classified as: {task_type.value}")
            return task_type

    return TaskType.GENERAL


def select_model(prompt: str, config: Dict, context: Optional[Dict] = None) -> str:
    """
    Select optimal model based on prompt and configuration

    Selection logic:
    1. Check if routing is enabled
    2. Check manual override
    3. Try task-type based mapping
    4. Fall back to complexity threshold
    """
    # Check if routing is enabled
    if not config.get("enabled", True):
        default = config.get("routingPolicies", {}).get("balanced", {}).get("defaultModel", "deepseek-r1:32b")
        logger.info(f"Routing disabled, using default: {default}")
        return default

    # Get active policy
    policy_name = config.get("policy", "balanced")
    policy = config.get("routingPolicies", {}).get(policy_name, {})

    # Check manual override
    manual_override = config.get("manualOverride")
    if manual_override:
        logger.info(f"Manual override active: {manual_override}")
        return manual_override

    # Try task-type based mapping
    task_type = classify_task(prompt)
    task_mapping = policy.get("taskTypeMapping", {})

    if task_type.value in task_mapping:
        selected = task_mapping[task_type.value]
        logger.info(f"Selected by task type ({task_type.value}): {selected}")
        return selected

    # Fall back to complexity threshold
    complexity = calculate_complexity(prompt, context)
    threshold = policy.get("complexityThreshold", 0.7)

    if complexity >= threshold:
        selected = policy.get("fallbackModel", "deepseek-r1:32b")
        logger.info(f"Selected by complexity ({complexity:.2f} >= {threshold}): {selected}")
    else:
        selected = policy.get("defaultModel", "nemotron-nano-9b-v2")
        logger.info(f"Selected by default ({complexity:.2f} < {threshold}): {selected}")

    return selected


def log_routing_decision(prompt: str, selected_model: str, complexity: float, task_type: TaskType):
    """Log routing decision for analytics"""
    decision = {
        "timestamp": datetime.now().isoformat(),
        "prompt_preview": prompt[:100],
        "selected_model": selected_model,
        "complexity": complexity,
        "task_type": task_type.value
    }
    logger.info(f"Routing decision: {json.dumps(decision)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "model-router-proxy"}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    """
    Proxy all requests to Ollama with intelligent model routing
    """
    config = load_config()
    body = await request.body()

    # Route generation requests with model selection
    if path == "api/generate" and request.method == "POST":
        try:
            data = json.loads(body)
            prompt = data.get("prompt", "")

            # Select model
            selected_model = select_model(prompt, config)

            # Override model in request
            original_model = data.get("model", "unknown")
            data["model"] = selected_model

            # Log decision
            complexity = calculate_complexity(prompt)
            task_type = classify_task(prompt)
            log_routing_decision(prompt, selected_model, complexity, task_type)

            print(f"🤖 {datetime.now().strftime('%H:%M:%S')} | {task_type.value:20s} | {selected_model:25s} | {prompt[:50]}")

            body = json.dumps(data).encode()
        except Exception as e:
            logger.error(f"Error processing request: {e}")

    # Forward request to Ollama
    async with httpx.AsyncClient(timeout=300.0) as client:
        url = f"{OLLAMA_BASE}/{path}"

        try:
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
        except Exception as e:
            logger.error(f"Error forwarding request: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500
            )


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🚀 Model Router Proxy Starting")
    print("=" * 80)
    print(f"📍 Proxy: http://localhost:11435")
    print(f"🎯 Ollama: {OLLAMA_BASE}")
    print(f"⚙️  Config: {CONFIG_PATH}")
    print(f"📝 Logs: {LOG_PATH}")
    print("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=11435, log_level="info")
