#!/usr/bin/env python3
"""Enhanced interactive chatbot with Rich UI and conversation history.

This version includes:
- Rich terminal UI with colors, tables, and progress bars
- Conversation history save/load functionality
- Improved visual presentation
- Session persistence

Usage:
    python scripts/chat_reviewer_enhanced.py
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import box

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()


class ConversationHistory:
    """Manage conversation history with save/load functionality."""

    def __init__(self, history_dir: Path = None):
        """Initialize conversation history manager.

        Args:
            history_dir: Directory to store conversation histories
        """
        if history_dir is None:
            history_dir = Path.home() / ".ai-coscientist" / "chat_history"

        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session_data: Dict) -> str:
        """Save a conversation session.

        Args:
            session_data: Dictionary containing session information

        Returns:
            Session ID
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = self.history_dir / f"session_{session_id}.json"

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return session_id

    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        session_file = self.history_dir / f"session_{session_id}.json"

        if not session_file.exists():
            return None

        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_sessions(self, limit: int = 10) -> List[Dict]:
        """List recent conversation sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata
        """
        sessions = []

        for session_file in sorted(self.history_dir.glob("session_*.json"), reverse=True)[:limit]:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        'id': session_file.stem.replace('session_', ''),
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'paper': data.get('paper_path', 'N/A'),
                        'messages': len(data.get('messages', []))
                    })
            except Exception:
                continue

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False otherwise
        """
        session_file = self.history_dir / f"session_{session_id}.json"

        if session_file.exists():
            session_file.unlink()
            return True

        return False


class PaperReviewChatbot:
    """Enhanced chatbot with Rich UI and conversation history."""

    def __init__(self):
        """Initialize the enhanced chatbot."""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.current_paper_path: Optional[str] = None
        self.current_scores: Optional[Dict] = None
        self.conversation_history: List[Dict] = []
        self.enhanced_versions: List[str] = []
        self.history_manager = ConversationHistory()
        self.current_session_id: Optional[str] = None

        # System prompt
        self.system_prompt = """You are an expert scientific paper reviewer and writing coach.
You help researchers improve their papers by:
1. Evaluating papers across 4 dimensions: Novelty, Methodology, Clarity, Significance
2. Providing specific, actionable improvement suggestions
3. Explaining scores in a friendly, encouraging way
4. Guiding users through the enhancement process step-by-step

You have access to a paper evaluation system that scores papers 0-10 using an ensemble of AI models.
You can also apply automated enhancements like adding theoretical justification, impact quantification, etc.

Be conversational, encouraging, and specific. Always ask clarifying questions when needed."""

    def display_scores_table(self, scores: Dict) -> None:
        """Display scores in a Rich table.

        Args:
            scores: Score dictionary
        """
        # Create overall score panel
        overall = scores['overall']
        confidence = scores.get('confidence', 0.0)

        # Determine color based on score
        if overall >= 9.0:
            color = "green"
            quality = "Exceptional"
        elif overall >= 8.5:
            color = "bright_green"
            quality = "Excellent"
        elif overall >= 8.0:
            color = "yellow"
            quality = "Very Good"
        elif overall >= 7.5:
            color = "bright_yellow"
            quality = "Good"
        elif overall >= 7.0:
            color = "orange1"
            quality = "Acceptable"
        else:
            color = "red"
            quality = "Needs Work"

        # Overall score panel
        overall_panel = Panel(
            f"[bold {color}]{overall}/10[/bold {color}] ({quality})\n"
            f"Confidence: {confidence:.2f}",
            title="📊 Overall Score",
            border_style=color
        )
        console.print(overall_panel)

        # Dimensional scores table
        dim_table = Table(title="📋 Dimensional Scores", box=box.ROUNDED)
        dim_table.add_column("Dimension", style="cyan", no_wrap=True)
        dim_table.add_column("Score", style="magenta")
        dim_table.add_column("Status", justify="right")

        dimensions = [
            ("Novelty", scores.get('novelty', 0)),
            ("Methodology", scores.get('methodology', 0)),
            ("Clarity", scores.get('clarity', 0)),
            ("Significance", scores.get('significance', 0))
        ]

        for dim_name, dim_score in dimensions:
            status = "✅ Strong" if dim_score >= 8.0 else "⚠️ Improve"
            status_color = "green" if dim_score >= 8.0 else "yellow"
            dim_table.add_row(
                dim_name,
                f"{dim_score:.2f}/10",
                f"[{status_color}]{status}[/{status_color}]"
            )

        console.print(dim_table)

        # Model contributions table
        model_table = Table(title="🤖 Model Contributions", box=box.ROUNDED)
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Weight", style="dim")
        model_table.add_column("Score", style="magenta")
        model_table.add_column("Focus", style="italic")

        models = [
            ("GPT-4", "40%", scores.get('gpt4', 0), "Narrative quality"),
            ("Hybrid", "30%", scores.get('hybrid', 0), "Technical depth"),
            ("Multi-task", "30%", scores.get('multitask', 0), "Novelty assessment")
        ]

        for model_name, weight, model_score, focus in models:
            model_table.add_row(model_name, weight, f"{model_score:.2f}/10", focus)

        console.print(model_table)

        # Display LLM feedback if available
        if scores.get('llm_evaluation', False):
            console.print()  # Empty line

            # Strengths panel
            if scores.get('strengths'):
                strengths_text = "\n".join([f"✓ {s}" for s in scores['strengths']])
                strengths_panel = Panel(
                    strengths_text,
                    title="💪 Strengths",
                    border_style="green"
                )
                console.print(strengths_panel)

            # Weaknesses panel
            if scores.get('weaknesses'):
                weaknesses_text = "\n".join([f"• {w}" for w in scores['weaknesses']])
                weaknesses_panel = Panel(
                    weaknesses_text,
                    title="⚠️ Areas for Improvement",
                    border_style="yellow"
                )
                console.print(weaknesses_panel)

            # Dimensional justifications
            if scores.get('novelty_justification'):
                console.print()
                justifications = []
                if scores.get('novelty_justification'):
                    justifications.append(f"[cyan]Novelty:[/cyan] {scores['novelty_justification']}")
                if scores.get('methodology_justification'):
                    justifications.append(f"[cyan]Methodology:[/cyan] {scores['methodology_justification']}")
                if scores.get('clarity_justification'):
                    justifications.append(f"[cyan]Clarity:[/cyan] {scores['clarity_justification']}")
                if scores.get('significance_justification'):
                    justifications.append(f"[cyan]Significance:[/cyan] {scores['significance_justification']}")

                if justifications:
                    justifications_panel = Panel(
                        "\n\n".join(justifications),
                        title="📊 Score Justifications",
                        border_style="blue"
                    )
                    console.print(justifications_panel)

            # Overall assessment
            if scores.get('overall_assessment'):
                console.print()
                assessment_panel = Panel(
                    scores['overall_assessment'],
                    title="📝 Overall Assessment",
                    border_style="magenta"
                )
                console.print(assessment_panel)

    def evaluate_paper(self, file_path: str, use_llm: bool = True) -> Dict:
        """Evaluate a paper with progress indicator.

        Args:
            file_path: Path to the paper file
            use_llm: Whether to use LLM evaluation (default: True)

        Returns:
            Dictionary with scores
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            eval_method = "LLM-based analysis" if use_llm else "heuristic analysis"
            task = progress.add_task(f"Analyzing paper with {eval_method}...", total=None)

            # Use the LLM paper evaluator module
            from paper_evaluator_llm import evaluate_paper_file

            try:
                scores = evaluate_paper_file(file_path, use_llm=use_llm)
                progress.update(task, completed=True)
                return scores
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]Error evaluating paper: {e}[/red]")
                console.print("[yellow]Falling back to heuristic evaluation...[/yellow]")
                # Return default scores on error
                try:
                    scores = evaluate_paper_file(file_path, use_llm=False)
                    return scores
                except:
                    return {
                        'overall': 7.0,
                        'novelty': 7.0,
                        'methodology': 7.0,
                        'clarity': 7.0,
                        'significance': 7.0,
                        'confidence': 0.60,
                        'gpt4': 7.0,
                        'hybrid': 7.0,
                        'multitask': 7.0,
                        'llm_evaluation': False
                    }

    def get_improvement_suggestions(self, current_score: float, target_score: float = None) -> List[Dict]:
        """Get improvement suggestions based on current score."""
        suggestions = [
            {
                'title': 'Transform Title with Crisis Framing',
                'description': 'Reframe title from incremental to paradigm shift',
                'time': '30 minutes',
                'expected_gain': 0.3,
                'difficulty': 'Easy',
                'script': None
            },
            {
                'title': 'Add Theoretical Justification Section',
                'description': 'Add ~1200 word theoretical foundations section',
                'time': '2 hours',
                'expected_gain': 0.3,
                'difficulty': 'Medium',
                'script': 'insert_theoretical_justification.py'
            },
            {
                'title': 'Quantify All Impact Statements',
                'description': 'Replace vague statements with specific numbers',
                'time': '1-2 hours',
                'expected_gain': 0.2,
                'difficulty': 'Easy',
                'script': None
            },
            {
                'title': 'Add Method Comparison Table',
                'description': 'Add systematic comparison with alternative methods',
                'time': '1 hour',
                'expected_gain': 0.1,
                'difficulty': 'Easy',
                'script': 'add_comparison_table.py'
            },
            {
                'title': 'Add Impact Boxes',
                'description': 'Add visual impact boxes with quantified metrics',
                'time': '30 minutes',
                'expected_gain': 0.05,
                'difficulty': 'Easy',
                'script': 'add_impact_boxes.py'
            }
        ]

        # Filter suggestions based on target
        if target_score:
            gap = target_score - current_score
            selected = []
            cumulative_gain = 0
            for sug in suggestions:
                if cumulative_gain < gap:
                    selected.append(sug)
                    cumulative_gain += sug['expected_gain']
            return selected

        return suggestions[:3]

    def display_suggestions(self, suggestions: List[Dict]) -> None:
        """Display improvement suggestions in a Rich table.

        Args:
            suggestions: List of improvement suggestions
        """
        table = Table(title="💡 Improvement Suggestions", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Suggestion", style="green")
        table.add_column("Time", style="yellow")
        table.add_column("Gain", style="magenta")
        table.add_column("Difficulty", style="blue")

        for i, sug in enumerate(suggestions, 1):
            table.add_row(
                str(i),
                sug['title'],
                sug['time'],
                f"+{sug['expected_gain']:.2f}",
                sug['difficulty']
            )

        console.print(table)

    def parse_user_intent(self, message: str) -> tuple[str, dict]:
        """Parse user message to determine intent."""
        msg_lower = message.lower()

        # Check for file path
        file_pattern = r'[\w\-./]+\.(?:docx|txt|pdf)'
        file_match = re.search(file_pattern, message)

        # History commands
        if any(word in msg_lower for word in ['save', 'save conversation', 'save session']):
            return 'save_history', {}
        elif any(word in msg_lower for word in ['load', 'load conversation', 'load session']):
            return 'load_history', {}
        elif any(word in msg_lower for word in ['show history', 'list sessions', 'list conversations']):
            return 'list_history', {}

        # Detect intents
        if any(word in msg_lower for word in ['review', 'evaluate', 'analyze', 'score']):
            if file_match:
                return 'evaluate', {'file_path': file_match.group(0)}
            else:
                return 'evaluate_request', {}

        elif any(word in msg_lower for word in ['improve', 'enhance', 'better', 'increase']):
            score_match = re.search(r'(\d+\.?\d*)', message)
            target = float(score_match.group(1)) if score_match else None
            return 'improve', {'target_score': target}

        elif any(word in msg_lower for word in ['apply', 'add', 'insert']):
            if 'theoretical' in msg_lower or 'theory' in msg_lower:
                return 'apply_enhancement', {'type': 'theoretical'}
            elif 'comparison' in msg_lower or 'table' in msg_lower:
                return 'apply_enhancement', {'type': 'comparison'}
            elif 'impact' in msg_lower or 'box' in msg_lower:
                return 'apply_enhancement', {'type': 'impact_boxes'}
            else:
                return 'apply_request', {}

        elif any(word in msg_lower for word in ['why', 'explain', 'how']):
            return 'explain', {}

        elif any(word in msg_lower for word in ['next', 'what now', 'then']):
            return 'next_steps', {}

        else:
            return 'general', {}

    def save_current_session(self) -> str:
        """Save current conversation session.

        Returns:
            Session ID
        """
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'paper_path': self.current_paper_path,
            'scores': self.current_scores,
            'messages': self.conversation_history,
            'enhanced_versions': self.enhanced_versions
        }

        session_id = self.history_manager.save_session(session_data)
        self.current_session_id = session_id

        return session_id

    def load_session_by_id(self, session_id: str) -> bool:
        """Load a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            True if loaded successfully
        """
        session_data = self.history_manager.load_session(session_id)

        if session_data:
            self.current_paper_path = session_data.get('paper_path')
            self.current_scores = session_data.get('scores')
            self.conversation_history = session_data.get('messages', [])
            self.enhanced_versions = session_data.get('enhanced_versions', [])
            self.current_session_id = session_id
            return True

        return False

    def display_session_list(self) -> None:
        """Display list of available sessions."""
        sessions = self.history_manager.list_sessions()

        if not sessions:
            console.print("[yellow]No saved sessions found.[/yellow]")
            return

        table = Table(title="💾 Saved Sessions", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Session ID", style="green")
        table.add_column("Date", style="yellow")
        table.add_column("Paper", style="magenta")
        table.add_column("Messages", style="blue")

        for i, session in enumerate(sessions, 1):
            timestamp = session['timestamp']
            if 'T' in timestamp:
                timestamp = timestamp.split('T')[0]

            table.add_row(
                str(i),
                session['id'],
                timestamp,
                Path(session['paper']).name if session['paper'] else 'N/A',
                str(session['messages'])
            )

        console.print(table)

    def generate_response(self, user_message: str) -> str:
        """Generate chatbot response using Claude."""
        # Parse intent
        intent, params = self.parse_user_intent(user_message)

        # Handle history intents
        if intent == 'save_history':
            session_id = self.save_current_session()
            return f"✅ Session saved! ID: {session_id}"

        elif intent == 'load_history':
            self.display_session_list()
            session_id = Prompt.ask("\n[cyan]Enter session ID to load[/cyan]")
            if self.load_session_by_id(session_id):
                console.print(f"[green]✅ Session {session_id} loaded![/green]")
                if self.current_scores:
                    self.display_scores_table(self.current_scores)
                return "Session restored. You can continue from where you left off."
            else:
                return f"❌ Session {session_id} not found."

        elif intent == 'list_history':
            self.display_session_list()
            return "Use 'load conversation' to restore a session."

        # Handle specific intents
        context = ""

        if intent == 'evaluate':
            file_path = params['file_path']
            console.print(f"\n[cyan]📄 Evaluating paper: {file_path}[/cyan]\n")

            # Evaluate paper
            self.current_paper_path = file_path
            self.current_scores = self.evaluate_paper(file_path)

            # Display scores with Rich
            self.display_scores_table(self.current_scores)

            # Auto-save after evaluation
            session_id = self.save_current_session()
            console.print(f"\n[dim]Session auto-saved: {session_id}[/dim]\n")

            context = f"""The user wants to evaluate their paper: {file_path}

Paper Evaluation Results:
- Overall Score: {self.current_scores['overall']}/10
- Novelty: {self.current_scores['novelty']}/10
- Methodology: {self.current_scores['methodology']}/10
- Clarity: {self.current_scores['clarity']}/10
- Significance: {self.current_scores['significance']}/10
- Confidence: {self.current_scores['confidence']}

Model Contributions:
- GPT-4 (40%): {self.current_scores['gpt4']}/10
- Hybrid (30%): {self.current_scores['hybrid']}/10
- Multi-task (30%): {self.current_scores['multitask']}/10

Provide an encouraging analysis of these scores, highlighting strengths and areas for improvement.
Ask the user what they'd like to work on."""

        elif intent == 'improve':
            target = params.get('target_score')
            if not self.current_scores:
                context = "The user wants to improve their paper, but no paper has been evaluated yet. Ask them to provide a paper to review first."
            else:
                suggestions = self.get_improvement_suggestions(
                    self.current_scores['overall'],
                    target
                )

                # Display suggestions with Rich
                self.display_suggestions(suggestions)

                suggestions_text = "\n".join([
                    f"{i+1}. {s['title']}"
                    f"\n   - Time: {s['time']}"
                    f"\n   - Expected gain: +{s['expected_gain']} points"
                    f"\n   - Difficulty: {s['difficulty']}"
                    for i, s in enumerate(suggestions)
                ])

                context = f"""The user wants to improve their paper. Current score: {self.current_scores['overall']}/10
Target score: {target or 'not specified'}

Top improvement suggestions:
{suggestions_text}

Present these suggestions in a friendly way and ask which one they'd like to start with."""

        else:
            # General conversation
            if self.current_scores:
                context = f"Current paper score: {self.current_scores['overall']}/10. The user is asking a general question: {user_message}"
            else:
                context = f"No paper evaluated yet. The user is asking: {user_message}"

        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })

        # Generate response using Claude
        messages = self.conversation_history.copy()
        if context:
            messages[-1]['content'] = f"{context}\n\nUser message: {user_message}"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Thinking...", total=None)

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=self.system_prompt,
                messages=messages
            )

            progress.update(task, completed=True)

        assistant_message = response.content[0].text

        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })

        return assistant_message

    def chat(self):
        """Start the interactive chat session with Rich UI."""
        # Welcome banner
        welcome = Panel(
            "[bold cyan]Paper Review Chatbot[/bold cyan]\n\n"
            "I can help you evaluate and improve your scientific papers.\n\n"
            "[yellow]Commands:[/yellow]\n"
            "  • Review a paper: 'Review my paper: /path/to/paper.docx'\n"
            "  • Improve scores: 'Help me get to 8.5+'\n"
            "  • Save session: 'save conversation'\n"
            "  • Load session: 'load conversation'\n"
            "  • Show history: 'show history'\n"
            "  • Exit: 'quit' or 'exit'",
            border_style="bright_blue",
            box=box.DOUBLE
        )
        console.print(welcome)

        while True:
            try:
                console.print()  # Empty line
                user_input = Prompt.ask("[bold green]💬 You[/bold green]").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    # Offer to save before exiting
                    if self.conversation_history:
                        save_prompt = Prompt.ask(
                            "[yellow]Save conversation before exiting?[/yellow]",
                            choices=["y", "n"],
                            default="y"
                        )
                        if save_prompt == "y":
                            session_id = self.save_current_session()
                            console.print(f"[green]✅ Session saved: {session_id}[/green]")

                    console.print("\n[bold cyan]👋 Goodbye! Good luck with your paper![/bold cyan]\n")
                    break

                # Generate response
                console.print()  # Empty line
                response = self.generate_response(user_input)

                # Display response as Markdown
                console.print(Panel(
                    Markdown(response),
                    title="🤖 Assistant",
                    border_style="blue"
                ))

            except KeyboardInterrupt:
                console.print("\n\n[bold cyan]👋 Chat interrupted. Goodbye![/bold cyan]\n")
                break
            except Exception as e:
                console.print(f"\n[red]❌ Error: {e}[/red]")
                console.print("[yellow]Please try again or type 'quit' to exit.[/yellow]")


def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[red]❌ Error: ANTHROPIC_API_KEY not found in environment variables.[/red]")
        console.print("Please set it in your .env file or export it:")
        console.print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    # Start chatbot
    chatbot = PaperReviewChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()
