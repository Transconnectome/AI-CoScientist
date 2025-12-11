"""Reusable adversarial (red-team) review utilities.

The original ``scripts/multi_agent_peer_review.py`` script has been
refactored into this module so both the CLI experience and the unified
pipeline can reuse the same review logic.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:  # Optional providers
    import openai  # type: ignore
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - provider optional
    openai = None
    OpenAI = None

try:
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover - provider optional
    Anthropic = None

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - provider optional
    genai = None


MAX_OPENAI_CHARS = 40000
MAX_ANTHROPIC_CHARS = 40000
MAX_GEMINI_CHARS = 30000


@dataclass
class ReviewAgentResult:
    model: str
    review: str
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AdversarialReviewResult:
    reviews: List[ReviewAgentResult]
    synthesis: str
    output_path: Optional[str]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.reviews if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.reviews if not r.success)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "reviews": [r.__dict__ for r in self.reviews],
            "output_path": self.output_path,
        }


async def run_adversarial_review(
    document_text: str,
    prompt: str,
    output_dir: Optional[Path] = None,
    file_prefix: str = "ADVERSARIAL_REVIEW",
) -> AdversarialReviewResult:
    """Execute multi-agent review with premium models (GPT-5.1, Gemini 3.0, Claude Opus 4.5).
    
    Falls back to DeepSeek when premium models are unavailable.
    Never uses downgraded models (3.5, 1.5, etc.) to maintain quality.
    """

    tasks = []

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Try premium models first
    if openai and OpenAI and openai_key:
        tasks.append(_review_with_openai_premium(prompt, document_text, openai_key))

    if Anthropic and anthropic_key:
        tasks.append(_review_with_claude_premium(prompt, document_text, anthropic_key))

    if genai and google_key:
        tasks.append(_review_with_gemini_premium(prompt, document_text, google_key))

    # Execute all review tasks
    reviews = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_reviews = []
    
    for result in reviews:
        if isinstance(result, ReviewAgentResult):
            if result.success:
                successful_reviews.append(result)
    
    # Final fallback to offline critic only if everything failed
    if not successful_reviews:
        successful_reviews.append(await _offline_review(prompt, document_text))
    
    synthesis = synthesize_reviews(successful_reviews)

    output_path: Optional[str] = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        file_path.write_text(synthesis, encoding="utf-8")
        output_path = str(file_path)

    return AdversarialReviewResult(reviews=successful_reviews, synthesis=synthesis, output_path=output_path)


def synthesize_reviews(reviews: List[ReviewAgentResult]) -> str:
    successful = [r for r in reviews if r.success]
    if not successful:
        return "ERROR: All review agents failed. No reviews to synthesize."

    lines = ["# ADVERSARIAL REVIEW SYNTHESIS", "=" * 80]
    lines.append(f"\n**Review Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Successful Agents:** {len(successful)}")
    lines.append(f"**Models:** {', '.join(r.model for r in successful)}")
    lines.append("\n" + "=" * 80 + "\n")

    for idx, review in enumerate(successful, 1):
        lines.append(f"\n## AGENT {idx}: {review.model}")
        lines.append("-" * 80)
        lines.append(review.review)
        lines.append("\n" + "=" * 80 + "\n")

    lines.append("\n## META-ANALYSIS")
    lines.append("-" * 80)
    lines.append(
        "Focus on (1) points of consensus, (2) contradictory findings, "
        "(3) fatal flaws, and (4) required revisions before submission."
    )

    return "\n".join(lines)


async def _review_with_openai_premium(
    prompt: str,
    document_text: str,
    api_key: str,
    deepseek_key: Optional[str],
) -> ReviewAgentResult:
    """Try GPT-5.x → GPT-4o cascade before falling back to DeepSeek."""

    full_prompt = _build_full_prompt(prompt, document_text, MAX_OPENAI_CHARS)
    client = OpenAI(api_key=api_key)  # type: ignore[misc]

    configured = os.getenv("OPENAI_MODEL", "gpt-5.1")
    candidate_models = []
    if configured:
        candidate_models.append(configured)
    candidate_models.extend(["gpt-5", "gpt-4o"])

    tried = set()
    last_error: Optional[str] = None

    for model_name in candidate_models:
        if not model_name or model_name in tried:
            continue
        tried.add(model_name)

        # Skip explicitly banned/down-graded families
        lowered = model_name.lower()
        if any(token in lowered for token in ["3.5", "3.0", "1.5", "turbo", "mini"]):
            continue

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Reviewer #2 focusing on rigor, fatal flaws, and "
                            "reproducibility."
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,
                max_tokens=6000,
            )
            review_text = response.choices[0].message.content
            return ReviewAgentResult(
                model=f"OpenAI {model_name}", review=review_text, success=True
            )
        except Exception as exc:
            last_error = str(exc)
            continue

    # Fallback to DeepSeek if available
    if deepseek_key and OpenAI:
        return await _review_with_deepseek(prompt, document_text, deepseek_key)

    return ReviewAgentResult(model="OpenAI", review="", success=False, error=last_error)


async def _review_with_claude_premium(
    prompt: str,
    document_text: str,
    api_key: str,
    deepseek_key: Optional[str],
) -> ReviewAgentResult:
    """Try Claude Opus 4.5 cascade before falling back to DeepSeek."""

    full_prompt = _build_full_prompt(prompt, document_text, MAX_ANTHROPIC_CHARS)
    client = Anthropic(api_key=api_key)  # type: ignore[misc]

    configured = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20250929")
    candidate_models = []
    if configured:
        candidate_models.append(configured)
    candidate_models.extend(
        [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
        ]
    )

    tried = set()
    last_error: Optional[str] = None

    for model_name in candidate_models:
        if not model_name or model_name in tried:
            continue
        tried.add(model_name)

        lowered = model_name.lower()
        if any(token in lowered for token in ["3.5", "instant", "haiku", "lite"]):
            continue

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=6000,
                temperature=0.3,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return ReviewAgentResult(
                model=f"Anthropic {model_name}",
                review=response.content[0].text,
                success=True,
            )
        except Exception as exc:
            last_error = str(exc)
            continue

    if deepseek_key and OpenAI:
        return await _review_with_deepseek(prompt, document_text, deepseek_key)

    return ReviewAgentResult(model="Anthropic", review="", success=False, error=last_error)


async def _review_with_gemini_premium(
    prompt: str,
    document_text: str,
    api_key: str,
    deepseek_key: Optional[str],
) -> ReviewAgentResult:
    """Try Gemini 3.x cascade before falling back to DeepSeek."""

    full_prompt = _build_full_prompt(prompt, document_text, MAX_GEMINI_CHARS)
    genai.configure(api_key=api_key)  # type: ignore[attr-defined]

    configured = os.getenv("GEMINI_MODEL", "gemini-3.0-pro")
    candidate_models = []
    if configured:
        candidate_models.append(configured)
    candidate_models.extend(
        [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
        ]
    )

    tried = set()
    last_error: Optional[str] = None

    for model_name in candidate_models:
        if not model_name or model_name in tried:
            continue
        tried.add(model_name)

        lowered = model_name.lower()
        if any(token in lowered for token in ["1.0", "nano", "lite"]):
            continue

        try:
            model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(  # type: ignore[attr-defined]
                    temperature=0.3,
                    max_output_tokens=6000,
                ),
            )
            return ReviewAgentResult(
                model=f"Google {model_name}", review=response.text, success=True
            )
        except Exception as exc:
            last_error = str(exc)
            continue

    if deepseek_key and OpenAI:
        return await _review_with_deepseek(prompt, document_text, deepseek_key)

    return ReviewAgentResult(model="Google Gemini", review="", success=False, error=last_error)


async def _review_with_deepseek(prompt: str, document_text: str, api_key: str) -> ReviewAgentResult:
    """DeepSeek fallback when premium models are unavailable."""
    full_prompt = _build_full_prompt(prompt, document_text, MAX_OPENAI_CHARS)

    try:
        # DeepSeek uses OpenAI-compatible API
        # Base URL should be https://api.deepseek.com (without /v1)
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # type: ignore[misc]
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Reviewer #2 focusing on rigor, fatal flaws, and "
                        "reproducibility."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.3,
            max_tokens=6000,
        )
        review_text = response.choices[0].message.content
        return ReviewAgentResult(model=f"DeepSeek {model_name}", review=review_text, success=True)
    except Exception as exc:
        return ReviewAgentResult(model="DeepSeek", review="", success=False, error=str(exc))


async def _offline_review(prompt: str, document_text: str) -> ReviewAgentResult:
    """Heuristic reviewer used when APIs are unavailable."""

    findings = []
    lower_text = document_text.lower()

    if "aim" not in lower_text:
        findings.append("Missing explicit Specific Aims section.")
    if "power" not in lower_text and "sample" not in lower_text:
        findings.append("No statistical power/sample size discussion detected.")
    if "validation" not in lower_text and "benchmark" not in lower_text:
        findings.append("Lacks quantitative validation or benchmarking claims.")
    if "risk" not in lower_text and "mitigation" not in lower_text:
        findings.append("Project risks and mitigation strategies are absent.")

    critique = "\n".join(f"- {item}" for item in findings) or "- Heuristic critic could not find obvious gaps."

    review = (
        "### Offline Reviewer (Red-Team Simulation)\n"
        "This fallback critic inspects the draft when API keys are not available.\n"
        f"Prompt focus: {prompt[:200]}...\n\n"
        "Key Issues:\n"
        f"{critique}\n"
        "Recommendation: Address the above items before submission."
    )

    return ReviewAgentResult(model="OfflineCritic", review=review, success=True)


def _build_full_prompt(prompt: str, document_text: str, limit: int) -> str:
    truncated = document_text
    if len(truncated) > limit:
        truncated = truncated[:limit] + "\n\n[... truncated for length ...]"

    return f"{prompt}\n\n---\n\nDOCUMENT TEXT:\n\n{truncated}"


