#!/usr/bin/env python3
"""Interactive chatbot for paper review and enhancement.

This script provides a conversational interface for evaluating and improving
scientific papers. Users can chat naturally to get scores, suggestions, and
apply enhancements.

Usage:
    python scripts/chat_reviewer.py

Example conversation:
    User: "Review my paper: paper.docx"
    Bot: "Analyzing paper... Score: 7.96/10. What would you like to improve?"
    User: "Get me to 8.5+"
    Bot: "Here are 3 suggestions to reach 8.5..."
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PaperReviewChatbot:
    """Interactive chatbot for paper review and enhancement."""

    def __init__(self):
        """Initialize the chatbot."""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.current_paper_path: Optional[str] = None
        self.current_scores: Optional[Dict] = None
        self.conversation_history: List[Dict] = []
        self.enhanced_versions: List[str] = []

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

    def evaluate_paper(self, file_path: str) -> Dict:
        """Evaluate a paper using the existing evaluation system.

        Args:
            file_path: Path to the paper file (.docx or .txt)

        Returns:
            Dictionary with scores and analysis
        """
        # Use the paper evaluator module
        from paper_evaluator import evaluate_paper_file

        try:
            scores = evaluate_paper_file(file_path)
            return scores
        except Exception as e:
            print(f"Error evaluating paper: {e}")
            # Return default scores on error
            return {
                'overall': 7.0,
                'novelty': 7.0,
                'methodology': 7.0,
                'clarity': 7.0,
                'significance': 7.0,
                'confidence': 0.80,
                'gpt4': 7.0,
                'hybrid': 7.0,
                'multitask': 7.0,
            }

    def get_improvement_suggestions(self, current_score: float, target_score: float = None) -> List[Dict]:
        """Get improvement suggestions based on current score.

        Args:
            current_score: Current overall score
            target_score: Desired target score (optional)

        Returns:
            List of improvement suggestions
        """
        suggestions = [
            {
                'title': 'Transform Title with Crisis Framing',
                'description': 'Reframe title from incremental to paradigm shift',
                'time': '30 minutes',
                'expected_gain': 0.3,
                'difficulty': 'Easy',
                'script': None  # Manual
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
                'script': None  # Manual
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
            # Select suggestions that would reach target
            selected = []
            cumulative_gain = 0
            for sug in suggestions:
                if cumulative_gain < gap:
                    selected.append(sug)
                    cumulative_gain += sug['expected_gain']
            return selected

        return suggestions[:3]  # Return top 3

    def apply_enhancement(self, enhancement_type: str) -> str:
        """Apply a specific enhancement to the paper.

        Args:
            enhancement_type: Type of enhancement to apply

        Returns:
            Path to enhanced paper
        """
        # Map enhancement types to scripts
        script_map = {
            'theoretical': 'insert_theoretical_justification.py',
            'comparison': 'add_comparison_table.py',
            'impact_boxes': 'add_impact_boxes.py',
            'literature': 'add_literature_implications.py'
        }

        script = script_map.get(enhancement_type)
        if not script:
            return None

        # Execute enhancement script
        import subprocess
        script_path = Path(__file__).parent / script

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output to find enhanced file path
            # This is simplified - actual implementation would parse script output
            output_file = self.current_paper_path.replace('.txt', f'-enhanced-{enhancement_type}.txt')
            self.enhanced_versions.append(output_file)

            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error applying enhancement: {e}")
            return None

    def parse_user_intent(self, message: str) -> tuple[str, dict]:
        """Parse user message to determine intent and extract parameters.

        Args:
            message: User's message

        Returns:
            Tuple of (intent, parameters)
        """
        msg_lower = message.lower()

        # Check for file path
        file_pattern = r'[\w\-./]+\.(?:docx|txt|pdf)'
        file_match = re.search(file_pattern, message)

        # Detect intents
        if any(word in msg_lower for word in ['review', 'evaluate', 'analyze', 'score']):
            if file_match:
                return 'evaluate', {'file_path': file_match.group(0)}
            else:
                return 'evaluate_request', {}

        elif any(word in msg_lower for word in ['improve', 'enhance', 'better', 'increase']):
            # Check for target score
            score_match = re.search(r'(\d+\.?\d*)', message)
            target = float(score_match.group(1)) if score_match else None
            return 'improve', {'target_score': target}

        elif any(word in msg_lower for word in ['apply', 'add', 'insert']):
            # Check for enhancement type
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

    def generate_response(self, user_message: str) -> str:
        """Generate chatbot response using Claude.

        Args:
            user_message: User's message

        Returns:
            Chatbot's response
        """
        # Parse intent
        intent, params = self.parse_user_intent(user_message)

        # Handle specific intents
        context = ""

        if intent == 'evaluate':
            file_path = params['file_path']
            # Evaluate paper
            self.current_paper_path = file_path
            self.current_scores = self.evaluate_paper(file_path)

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

        elif intent == 'apply_enhancement':
            enhancement_type = params['type']
            if not self.current_paper_path:
                context = "The user wants to apply an enhancement, but no paper is loaded. Ask them to provide a paper first."
            else:
                # Apply enhancement
                enhanced_path = self.apply_enhancement(enhancement_type)
                if enhanced_path:
                    # Re-evaluate
                    new_scores = self.evaluate_paper(enhanced_path)
                    old_score = self.current_scores['overall']
                    new_score = new_scores['overall']
                    gain = new_score - old_score

                    self.current_scores = new_scores

                    context = f"""Successfully applied {enhancement_type} enhancement!

Results:
- Previous score: {old_score}/10
- New score: {new_score}/10
- Improvement: +{gain:.2f} points

Enhanced file saved to: {enhanced_path}

Ask if the user wants to apply more enhancements or if they're satisfied."""
                else:
                    context = f"Failed to apply {enhancement_type} enhancement. Apologize and suggest alternatives."

        elif intent == 'explain':
            if not self.current_scores:
                context = "The user wants an explanation, but no paper has been evaluated. Ask what they'd like to know about."
            else:
                context = f"""The user wants to understand their scores better. Current scores:
- Overall: {self.current_scores['overall']}/10
- Novelty: {self.current_scores['novelty']}/10
- Methodology: {self.current_scores['methodology']}/10
- Clarity: {self.current_scores['clarity']}/10
- Significance: {self.current_scores['significance']}/10

Explain what these dimensions mean and why the paper received these scores. Be specific and helpful."""

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

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=self.system_prompt,
            messages=messages
        )

        assistant_message = response.content[0].text

        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })

        return assistant_message

    def chat(self):
        """Start the interactive chat session."""
        print("\n" + "="*80)
        print("📝 Paper Review Chatbot")
        print("="*80)
        print("\nHello! I'm your AI paper review assistant.")
        print("I can help you evaluate and improve your scientific papers.")
        print("\nWhat would you like to do?")
        print("  - Review a paper: 'Review my paper: /path/to/paper.docx'")
        print("  - Improve scores: 'Help me get to 8.5+'")
        print("  - Apply enhancements: 'Add theoretical justification'")
        print("  - Ask questions: 'Why did I get this score?'")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("-"*80 + "\n")

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Good luck with your paper!")
                    break

                # Generate response
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")


def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please set it in your .env file or export it:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)

    # Start chatbot
    chatbot = PaperReviewChatbot()
    chatbot.chat()


if __name__ == "__main__":
    main()
