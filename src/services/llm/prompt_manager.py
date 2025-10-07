"""Prompt template management."""

from typing import Any, Dict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


class PromptManager:
    """Manage prompt templates and optimization."""

    def __init__(self, templates_dir: str = "prompts"):
        """Initialize prompt manager."""
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False
        )
        self.cache: Dict[str, Template] = {}

    def render_prompt(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Render prompt from template."""
        template = self._get_template(template_name)
        return template.render(**context)

    def validate_prompt(self, prompt: str, max_tokens: int = 4000) -> bool:
        """Validate prompt doesn't exceed token limit."""
        # Rough estimation: 4 chars ≈ 1 token
        estimated_tokens = len(prompt) // 4
        return estimated_tokens <= max_tokens

    def optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for token efficiency."""
        # Remove extra whitespace
        optimized = " ".join(prompt.split())

        # Remove redundant phrases
        redundant_phrases = [
            "please note that",
            "it is important to",
            "you should be aware that"
        ]
        for phrase in redundant_phrases:
            optimized = optimized.replace(phrase, "")

        return optimized.strip()

    def _get_template(self, template_name: str) -> Template:
        """Get template from cache or load."""
        if template_name not in self.cache:
            template_path = f"{template_name}.j2"
            self.cache[template_name] = self.env.get_template(template_path)
        return self.cache[template_name]

    def create_template(self, name: str, content: str) -> None:
        """Create a new prompt template."""
        template_path = self.templates_dir / f"{name}.j2"
        template_path.write_text(content)
        # Clear cache for this template
        if name in self.cache:
            del self.cache[name]
