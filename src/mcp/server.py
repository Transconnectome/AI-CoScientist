"""AI-CoScientist MCP Server for paper analysis and improvement."""

import asyncio
import json
import logging
from typing import Any, Dict, List
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from src.mcp.simple_llm import SimpleLLMService
from src.mcp.file_utils import read_file, validate_file_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperReviewerServer:
    """MCP Server for AI-CoScientist paper analysis and improvement."""

    def __init__(self):
        """Initialize server."""
        self.server = Server("ai-coscientist")
        self.llm_service: SimpleLLMService = None
        self.setup_handlers()

    def _ensure_llm_service(self):
        """Lazy initialization of LLM service."""
        if self.llm_service is None:
            self.llm_service = SimpleLLMService()

    def setup_handlers(self):
        """Setup MCP server handlers."""

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[types.Prompt]:
            """List available prompts for @ menu."""
            return [
                types.Prompt(
                    name="analyze-paper",
                    description="📊 Analyze paper quality (4D scoring)",
                    arguments=[
                        types.PromptArgument(
                            name="file_path",
                            description="Path to paper file (.docx or .txt)",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="improve-section",
                    description="✨ Improve specific paper section",
                    arguments=[
                        types.PromptArgument(
                            name="file_path",
                            description="Path to paper file",
                            required=True
                        ),
                        types.PromptArgument(
                            name="section",
                            description="Section to improve (abstract/introduction/methods/results/discussion)",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="get-suggestions",
                    description="💡 Get improvement suggestions",
                    arguments=[
                        types.PromptArgument(
                            name="file_path",
                            description="Path to paper file",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="compare-versions",
                    description="🔄 Compare two paper versions",
                    arguments=[
                        types.PromptArgument(
                            name="original_path",
                            description="Path to original paper",
                            required=True
                        ),
                        types.PromptArgument(
                            name="revised_path",
                            description="Path to revised paper",
                            required=True
                        )
                    ]
                ),
                types.Prompt(
                    name="ask-question",
                    description="💬 Ask a question about the paper",
                    arguments=[
                        types.PromptArgument(
                            name="file_path",
                            description="Path to paper file",
                            required=True
                        ),
                        types.PromptArgument(
                            name="question",
                            description="Your question",
                            required=True
                        )
                    ]
                )
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str,
            arguments: dict[str, str] | None
        ) -> types.GetPromptResult:
            """Handle prompt requests from @ menu."""
            if not arguments:
                arguments = {}

            if name == "analyze-paper":
                file_path = arguments.get("file_path", "")
                return types.GetPromptResult(
                    description="Analyze academic paper quality",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"Please use the analyze_paper tool to analyze this paper:\n\nFile: {file_path}\n\nProvide comprehensive quality assessment with 4-dimensional scoring (Novelty, Methodology, Clarity, Significance)."
                            )
                        )
                    ]
                )

            elif name == "improve-section":
                file_path = arguments.get("file_path", "")
                section = arguments.get("section", "abstract")
                return types.GetPromptResult(
                    description=f"Improve {section} section",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"Please use the improve_paper tool to improve the {section} section of this paper:\n\nFile: {file_path}\nSection: {section}\n\nProvide improved text with clear explanation of changes."
                            )
                        )
                    ]
                )

            elif name == "get-suggestions":
                file_path = arguments.get("file_path", "")
                return types.GetPromptResult(
                    description="Get improvement suggestions",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"Please use the get_improvement_suggestions tool to get top 5 improvement suggestions for this paper:\n\nFile: {file_path}\n\nPrioritize by expected impact."
                            )
                        )
                    ]
                )

            elif name == "compare-versions":
                original_path = arguments.get("original_path", "")
                revised_path = arguments.get("revised_path", "")
                return types.GetPromptResult(
                    description="Compare paper versions",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"Please use the compare_papers tool to compare these two versions:\n\nOriginal: {original_path}\nRevised: {revised_path}\n\nAnalyze score changes and improvement effectiveness."
                            )
                        )
                    ]
                )

            elif name == "ask-question":
                file_path = arguments.get("file_path", "")
                question = arguments.get("question", "")
                return types.GetPromptResult(
                    description="Answer question about paper",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text",
                                text=f"Please use the chat_review tool to answer this question about the paper:\n\nFile: {file_path}\nQuestion: {question}\n\nProvide a detailed, specific answer."
                            )
                        )
                    ]
                )

            else:
                raise ValueError(f"Unknown prompt: {name}")

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return [
                types.Tool(
                    name="analyze_paper",
                    description=(
                        "Analyze academic paper quality with 4-dimensional scoring: "
                        "Novelty, Methodology, Clarity, Significance. "
                        "Provides comprehensive quality assessment with suggestions."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to paper file (.docx or .txt)"
                            },
                            "include_suggestions": {
                                "type": "boolean",
                                "description": "Include improvement suggestions",
                                "default": True
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                types.Tool(
                    name="improve_paper",
                    description=(
                        "Improve specific paper section or entire paper. "
                        "Sections: abstract, introduction, methods, results, discussion, all. "
                        "Returns improved text with change summary."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to paper file"
                            },
                            "section": {
                                "type": "string",
                                "description": "Section to improve (abstract/introduction/methods/results/discussion/all)",
                                "enum": ["abstract", "introduction", "methods", "results", "discussion", "all"]
                            },
                            "strategy": {
                                "type": "string",
                                "description": "Improvement strategy (optional)",
                                "enum": ["clarity", "conciseness", "technical_depth", "general"]
                            }
                        },
                        "required": ["file_path", "section"]
                    }
                ),
                types.Tool(
                    name="get_improvement_suggestions",
                    description=(
                        "Generate prioritized improvement suggestions for the paper. "
                        "Returns top N suggestions ranked by expected impact."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to paper file"
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Number of top suggestions to return",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                types.Tool(
                    name="compare_papers",
                    description=(
                        "Compare two versions of a paper (original vs revised). "
                        "Analyzes score changes and improvement effectiveness."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "original_path": {
                                "type": "string",
                                "description": "Path to original paper file"
                            },
                            "revised_path": {
                                "type": "string",
                                "description": "Path to revised paper file"
                            }
                        },
                        "required": ["original_path", "revised_path"]
                    }
                ),
                types.Tool(
                    name="chat_review",
                    description=(
                        "Interactive chat-based review. Ask questions about the paper "
                        "and get natural language answers."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to paper file"
                            },
                            "query": {
                                "type": "string",
                                "description": "Your question about the paper"
                            }
                        },
                        "required": ["file_path", "query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict
        ) -> list[types.TextContent]:
            """Handle tool calls."""
            try:
                # Ensure LLM service is initialized
                self._ensure_llm_service()

                # Route to appropriate handler
                if name == "analyze_paper":
                    result = await self._analyze_paper(arguments)
                elif name == "improve_paper":
                    result = await self._improve_paper(arguments)
                elif name == "get_improvement_suggestions":
                    result = await self._get_improvement_suggestions(arguments)
                elif name == "compare_papers":
                    result = await self._compare_papers(arguments)
                elif name == "chat_review":
                    result = await self._chat_review(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")

                return [types.TextContent(type="text", text=result)]

            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                error_msg = f"❌ Error: {str(e)}"
                return [types.TextContent(type="text", text=error_msg)]

    async def _analyze_paper(self, args: Dict[str, Any]) -> str:
        """Analyze paper quality."""
        file_path = args["file_path"]
        include_suggestions = args.get("include_suggestions", True)

        # Validate file
        error = validate_file_path(file_path)
        if error:
            return f"❌ {error}"

        # Read paper
        try:
            text = read_file(file_path)
        except Exception as e:
            return f"❌ Failed to read file: {str(e)}"

        # Analyze with LLM
        prompt = f"""Analyze this academic paper and provide comprehensive quality assessment.

Paper content:
{text[:8000]}  # Limit to avoid token overflow

Provide analysis as JSON with these fields:
{{
    "novelty_score": <float 0-10, how novel and innovative>,
    "methodology_score": <float 0-10, rigor and appropriateness>,
    "clarity_score": <float 0-10, writing clarity and readability>,
    "significance_score": <float 0-10, potential impact>,
    "overall_score": <float 0-10, weighted average>,
    "strengths": [<list of 2-4 key strengths>],
    "weaknesses": [<list of 2-4 key weaknesses>],
    "suggestions": [<list of 3-5 improvement suggestions>]
}}

Be specific and constructive. Return only valid JSON."""

        try:
            response = await self.llm_service.complete(prompt, max_tokens=2000)
            analysis = json.loads(response["content"])

            # Format output
            output = "📊 **Paper Analysis Results**\n\n"
            output += f"**Overall Score**: {analysis.get('overall_score', 0):.1f}/10\n\n"
            output += "**Dimension Scores**:\n"
            output += f"  • Novelty: {analysis.get('novelty_score', 0):.1f}/10\n"
            output += f"  • Methodology: {analysis.get('methodology_score', 0):.1f}/10\n"
            output += f"  • Clarity: {analysis.get('clarity_score', 0):.1f}/10\n"
            output += f"  • Significance: {analysis.get('significance_score', 0):.1f}/10\n\n"

            output += "**Strengths** ✅:\n"
            for i, strength in enumerate(analysis.get('strengths', []), 1):
                output += f"  {i}. {strength}\n"

            output += "\n**Weaknesses** ⚠️:\n"
            for i, weakness in enumerate(analysis.get('weaknesses', []), 1):
                output += f"  {i}. {weakness}\n"

            if include_suggestions and 'suggestions' in analysis:
                output += "\n**Improvement Suggestions** 💡:\n"
                for i, suggestion in enumerate(analysis.get('suggestions', []), 1):
                    output += f"  {i}. {suggestion}\n"

            return output

        except json.JSONDecodeError:
            return "❌ Failed to parse analysis results"
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"

    async def _improve_paper(self, args: Dict[str, Any]) -> str:
        """Improve paper section."""
        file_path = args["file_path"]
        section = args["section"]
        strategy = args.get("strategy", "general")

        # Validate file
        error = validate_file_path(file_path)
        if error:
            return f"❌ {error}"

        # Read paper
        try:
            text = read_file(file_path)
        except Exception as e:
            return f"❌ Failed to read file: {str(e)}"

        # Strategy-specific instructions
        strategy_instructions = {
            "clarity": "Focus on improving clarity, readability, and logical flow.",
            "conciseness": "Make the text more concise while preserving key information.",
            "technical_depth": "Add more technical depth and detailed explanations.",
            "general": "Improve overall quality, clarity, and impact."
        }

        instruction = strategy_instructions.get(strategy, strategy_instructions["general"])

        # Build improvement prompt
        if section == "all":
            content_to_improve = text[:8000]
            section_desc = "entire paper"
        else:
            # Try to extract section
            # Simple heuristic: look for section headers
            content_to_improve = text[:8000]
            section_desc = f"{section} section"

        prompt = f"""Improve the following {section_desc} of an academic paper.

Original content:
{content_to_improve}

Improvement strategy: {instruction}

Guidelines:
- Maintain academic tone and rigor
- Enhance clarity and readability
- Strengthen arguments and evidence
- Remove redundancy
- Improve logical flow

Return JSON:
{{
    "improved_content": "<full improved text>",
    "changes_summary": "<summary of key changes made>",
    "improvement_areas": [<list of specific improvements>]
}}

Return only valid JSON."""

        try:
            response = await self.llm_service.complete(prompt, max_tokens=4000)
            result = json.loads(response["content"])

            # Format output
            output = f"✨ **{section.capitalize()} Improvement Results**\n\n"
            output += "**Changes Summary**:\n"
            output += f"{result.get('changes_summary', 'No summary available')}\n\n"

            if 'improvement_areas' in result:
                output += "**Improvement Areas**:\n"
                for i, area in enumerate(result.get('improvement_areas', []), 1):
                    output += f"  {i}. {area}\n"
                output += "\n"

            output += "**Improved Content**:\n"
            output += "─" * 50 + "\n"
            output += result.get('improved_content', 'No improved content available')
            output += "\n" + "─" * 50

            return output

        except json.JSONDecodeError:
            return "❌ Failed to parse improvement results"
        except Exception as e:
            return f"❌ Improvement failed: {str(e)}"

    async def _get_improvement_suggestions(self, args: Dict[str, Any]) -> str:
        """Get prioritized improvement suggestions."""
        file_path = args["file_path"]
        top_n = args.get("top_n", 5)

        # Validate file
        error = validate_file_path(file_path)
        if error:
            return f"❌ {error}"

        # Read paper
        try:
            text = read_file(file_path)
        except Exception as e:
            return f"❌ Failed to read file: {str(e)}"

        prompt = f"""Analyze this paper and provide {top_n} most impactful improvement suggestions.

Paper content:
{text[:8000]}

For each suggestion, provide:
- Title: Brief description
- Expected impact: High/Medium/Low
- Time required: Short/Medium/Long
- Specific action: What to do

Return JSON array:
[
    {{
        "title": "<suggestion title>",
        "impact": "high|medium|low",
        "time_required": "short|medium|long",
        "action": "<specific action to take>",
        "expected_improvement": "<what will improve>"
    }},
    ...
]

Prioritize by expected impact. Return only valid JSON array."""

        try:
            response = await self.llm_service.complete(prompt, max_tokens=2000)
            suggestions = json.loads(response["content"])

            # Format output
            output = f"💡 **Top {top_n} Improvement Suggestions**\n\n"

            for i, suggestion in enumerate(suggestions[:top_n], 1):
                impact = suggestion.get('impact', 'unknown').upper()
                time = suggestion.get('time_required', 'unknown')

                # Impact emoji
                impact_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💫"}.get(impact, "•")

                output += f"{i}. {impact_emoji} **{suggestion.get('title', 'Untitled')}**\n"
                output += f"   Impact: {impact} | Time: {time}\n"
                output += f"   Action: {suggestion.get('action', 'No action specified')}\n"
                output += f"   Expected: {suggestion.get('expected_improvement', 'No description')}\n\n"

            return output

        except json.JSONDecodeError:
            return "❌ Failed to parse suggestions"
        except Exception as e:
            return f"❌ Suggestion generation failed: {str(e)}"

    async def _compare_papers(self, args: Dict[str, Any]) -> str:
        """Compare two paper versions."""
        original_path = args["original_path"]
        revised_path = args["revised_path"]

        # Validate files
        for path in [original_path, revised_path]:
            error = validate_file_path(path)
            if error:
                return f"❌ {error}"

        # Read both papers
        try:
            original_text = read_file(original_path)
            revised_text = read_file(revised_path)
        except Exception as e:
            return f"❌ Failed to read files: {str(e)}"

        # Analyze both versions
        prompt = f"""Compare these two versions of an academic paper and analyze improvements.

**Original Version**:
{original_text[:4000]}

**Revised Version**:
{revised_text[:4000]}

Provide comparison as JSON:
{{
    "original_score": <float 0-10>,
    "revised_score": <float 0-10>,
    "improvement_percentage": <float percentage>,
    "key_improvements": [<list of key improvements made>],
    "remaining_issues": [<list of issues still present>],
    "dimension_changes": {{
        "novelty": <float change>,
        "methodology": <float change>,
        "clarity": <float change>,
        "significance": <float change>
    }}
}}

Return only valid JSON."""

        try:
            response = await self.llm_service.complete(prompt, max_tokens=2000)
            comparison = json.loads(response["content"])

            # Format output
            output = "📊 **Paper Comparison Results**\n\n"
            output += f"**Original Score**: {comparison.get('original_score', 0):.1f}/10\n"
            output += f"**Revised Score**: {comparison.get('revised_score', 0):.1f}/10\n"
            output += f"**Improvement**: +{comparison.get('improvement_percentage', 0):.1f}%\n\n"

            if 'dimension_changes' in comparison:
                output += "**Dimension Changes**:\n"
                for dim, change in comparison['dimension_changes'].items():
                    emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    output += f"  {emoji} {dim.capitalize()}: {change:+.1f}\n"
                output += "\n"

            output += "**Key Improvements** ✨:\n"
            for i, improvement in enumerate(comparison.get('key_improvements', []), 1):
                output += f"  {i}. {improvement}\n"

            if comparison.get('remaining_issues'):
                output += "\n**Remaining Issues** ⚠️:\n"
                for i, issue in enumerate(comparison.get('remaining_issues', []), 1):
                    output += f"  {i}. {issue}\n"

            return output

        except json.JSONDecodeError:
            return "❌ Failed to parse comparison results"
        except Exception as e:
            return f"❌ Comparison failed: {str(e)}"

    async def _chat_review(self, args: Dict[str, Any]) -> str:
        """Interactive chat review."""
        file_path = args["file_path"]
        query = args["query"]

        # Validate file
        error = validate_file_path(file_path)
        if error:
            return f"❌ {error}"

        # Read paper
        try:
            text = read_file(file_path)
        except Exception as e:
            return f"❌ Failed to read file: {str(e)}"

        prompt = f"""You are an expert academic paper reviewer. Answer the following question about this paper.

Paper content:
{text[:8000]}

Question: {query}

Provide a detailed, helpful answer. Be specific and reference relevant parts of the paper."""

        try:
            response = await self.llm_service.complete(prompt, max_tokens=1500)

            # Format output
            output = f"💬 **Review Question**: {query}\n\n"
            output += "**Answer**:\n"
            output += response["content"]

            return output

        except Exception as e:
            return f"❌ Chat review failed: {str(e)}"

    async def run(self):
        """Run the MCP server."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("AI-CoScientist MCP server starting...")
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Entry point for MCP server."""
    server = PaperReviewerServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
