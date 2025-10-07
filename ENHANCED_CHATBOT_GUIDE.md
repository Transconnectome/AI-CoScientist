# Enhanced Chatbot Guide - Phase 1 & 2 Improvements

## Overview

The AI-CoScientist chatbot has been significantly enhanced with two major improvements:

### Phase 1: Rich UI + Conversation History
- **Rich Terminal UI**: Beautiful colored output, tables, progress bars, and panels
- **Conversation History**: Save and load sessions for continuity across sessions

### Phase 2: Real LLM Evaluation
- **Claude AI Integration**: Real AI-based paper evaluation instead of heuristics
- **Detailed Feedback**: Comprehensive strengths, weaknesses, and justifications
- **Higher Accuracy**: More reliable and consistent scoring

## Installation

### 1. Install Dependencies

```bash
# Using Poetry (recommended)
cd AI-CoScientist
poetry install

# The following packages are now included:
# - rich: Terminal UI enhancement
# - python-docx: Document processing
# - anthropic: Claude AI integration
```

### 2. Set Up API Key

Make sure your `.env` file contains:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Using the Enhanced Chatbot

### Starting the Enhanced Version

```bash
# Run the enhanced chatbot with Rich UI and LLM evaluation
python scripts/chat_reviewer_enhanced.py
```

### Key Features

#### 1. Beautiful Terminal UI

**Score Display:**
- Color-coded overall score panel (green for high scores, red for low)
- Organized dimensional scores table
- Model contributions breakdown
- Visual progress indicators during evaluation

**Rich Formatting:**
- Markdown rendering for bot responses
- Syntax-highlighted code examples
- Bordered panels for different sections
- Emoji indicators for visual clarity

#### 2. Conversation History

**Save Session:**
```
💬 You: save conversation

✅ Session saved! ID: 20241007_143022
```

**Load Previous Session:**
```
💬 You: load conversation

[Displays list of saved sessions]
Enter session ID to load: 20241007_143022

✅ Session 20241007_143022 loaded!
[Displays previous paper scores]
```

**List All Sessions:**
```
💬 You: show history

[Displays table of recent sessions with dates and paper names]
```

**Auto-Save:**
- Sessions are automatically saved after each paper evaluation
- Session data includes:
  - Conversation messages
  - Paper scores
  - Enhanced versions
  - Timestamps

**Storage Location:**
- Sessions saved in `~/.ai-coscientist/chat_history/`
- JSON format for easy inspection and backup

#### 3. LLM-Based Evaluation

**Real AI Analysis:**
```
💬 You: Review my paper: paper.docx

[Rich progress indicator]
Analyzing paper with LLM-based analysis...

📊 Overall Score: 8.34/10 (Very Good)
   Confidence: 0.92

[Color-coded dimensional scores table]

💪 Strengths:
✓ Novel integration of ensemble methods
✓ Comprehensive experimental validation
✓ Clear methodology description

⚠️ Areas for Improvement:
• Limited discussion of computational complexity
• Could benefit from additional real-world case studies

📊 Score Justifications:
Novelty: The ensemble approach is innovative but builds on existing frameworks...
Methodology: Experimental design is rigorous with proper validation...
Clarity: Writing is generally clear but technical sections could be simplified...
Significance: Addresses important problem with practical implications...

📝 Overall Assessment:
This paper presents a solid contribution to the field with strong methodology
and clear presentation. The ensemble approach shows promise for practical
applications.
```

**Fallback to Heuristics:**
- If LLM evaluation fails (API issues, quota), automatically falls back to heuristic evaluation
- Clearly indicates which evaluation method was used
- Lower confidence score for heuristic evaluation

### Comparison: Original vs Enhanced

| Feature | Original | Enhanced |
|---------|----------|----------|
| **UI** | Plain text | Rich colored tables, panels, progress bars |
| **Evaluation** | Heuristics only | Real Claude AI + heuristic fallback |
| **Feedback** | Basic scores | Detailed strengths, weaknesses, justifications |
| **History** | None | Save/load sessions with full context |
| **Confidence** | ~0.65 | ~0.92 with LLM |
| **Auto-save** | No | Yes, after each evaluation |
| **Progress** | No indicators | Spinners and progress bars |

## Usage Examples

### Example 1: Complete Paper Review Workflow

```bash
$ python scripts/chat_reviewer_enhanced.py

[Beautiful welcome banner displayed]

💬 You: Review my paper: ~/research/breakthrough-paper.docx

[Rich progress: "Analyzing paper with LLM-based analysis..."]

📊 Overall Score: 7.96/10 (Good - Respectable journals)
   Confidence: 0.92

[Colored dimensional scores table]
[Strengths and weaknesses panels]
[Detailed justifications]

Session auto-saved: 20241007_151030

🤖 Assistant: Great work! Your paper shows strong methodology (7.89)
and clear writing (7.45). The main areas for improvement are novelty
(7.46) and significance (7.40).

Would you like specific suggestions to reach 8.5+?

💬 You: Yes, get me to 8.5+

[Rich table of improvement suggestions displayed]

💡 Improvement Suggestions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# | Suggestion                          | Time     | Gain   | Difficulty
──┼─────────────────────────────────────┼──────────┼────────┼────────────
1 | Transform Title with Crisis Framing | 30 min   | +0.30  | Easy
2 | Add Theoretical Justification       | 2 hours  | +0.30  | Medium
3 | Quantify All Impact Statements      | 1-2 hours| +0.20  | Easy

[Continue conversation...]

💬 You: save conversation

✅ Session saved! ID: 20241007_151030

💬 You: quit

👋 Goodbye! Good luck with your paper!
```

### Example 2: Resume Previous Session

```bash
$ python scripts/chat_reviewer_enhanced.py

💬 You: load conversation

💾 Saved Sessions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# | Session ID      | Date       | Paper              | Messages
──┼─────────────────┼────────────┼────────────────────┼──────────
1 | 20241007_151030 | 2024-10-07 | breakthrough-paper | 12
2 | 20241006_093015 | 2024-10-06 | pilot-study       | 8
3 | 20241005_140022 | 2024-10-05 | review-article    | 15

Enter session ID to load: 20241007_151030

✅ Session 20241007_151030 loaded!

[Previous paper scores displayed]

🤖 Assistant: Session restored. You can continue from where you left off.

💬 You: Apply the theoretical justification enhancement

[Continue from previous session context...]
```

### Example 3: Compare LLM vs Heuristic

```bash
# LLM Evaluation (Default)
$ python scripts/paper_evaluator_llm.py paper.docx

📊 Overall Score: 8.34/10 (Very Good - Strong specialty journals)
   Confidence: 0.92
   LLM Evaluation: Yes

✅ Strengths:
   • Novel ensemble approach
   • Comprehensive validation
   • Clear methodology

⚠️ Areas for Improvement:
   • Limited computational complexity analysis
   • Could expand real-world applications

# Heuristic Evaluation
$ python scripts/paper_evaluator_llm.py paper.docx --no-llm

📊 Overall Score: 7.80/10 (Good - Respectable journals)
   Confidence: 0.65
   LLM Evaluation: No (Heuristic)

[Basic scores only, no detailed feedback]
```

## Advanced Features

### Session Management

**Session Data Structure:**
```json
{
  "timestamp": "2024-10-07T15:10:30",
  "paper_path": "/path/to/paper.docx",
  "scores": {
    "overall": 7.96,
    "novelty": 7.46,
    ...
  },
  "messages": [
    {"role": "user", "content": "Review my paper..."},
    {"role": "assistant", "content": "..."}
  ],
  "enhanced_versions": []
}
```

**Session Operations:**
- `save conversation`: Manual save
- `load conversation`: Restore previous session
- `show history`: List all saved sessions
- Auto-save: Triggered after each evaluation

### Evaluation Modes

**LLM Mode (Default):**
- Uses Claude 3.5 Sonnet for analysis
- Provides detailed justifications
- Higher confidence (0.92)
- API costs apply (~$0.01-0.05 per evaluation)

**Heuristic Mode (Fallback):**
- Structure-based scoring
- Word count analysis
- No API costs
- Lower confidence (0.65)
- Fast evaluation

**Switching Modes:**
```python
# In code
scores = evaluate_paper_file("paper.docx", use_llm=True)  # LLM mode
scores = evaluate_paper_file("paper.docx", use_llm=False) # Heuristic mode
```

### Rich UI Components

**Available Components:**
- `Panel`: Bordered sections with titles
- `Table`: Organized data display with headers
- `Progress`: Spinners and progress bars
- `Markdown`: Formatted text rendering
- `Prompt`: Enhanced user input

**Color Scheme:**
- Green: High scores, strengths, success
- Yellow: Medium scores, warnings
- Red: Low scores, errors
- Cyan: Titles, labels
- Magenta: Highlights
- Blue: Information

## Configuration

### Chat Behavior

Edit `scripts/chat_reviewer_enhanced.py`:

```python
# Evaluation settings
self.use_llm_by_default = True  # Change to False for heuristic default

# History settings
history_dir = Path.home() / ".ai-coscientist" / "chat_history"

# Claude model
model="claude-sonnet-4-5-20250929"  # Latest Sonnet 4.5
temperature=0.3  # Lower = more consistent
max_tokens=2048
```

### LLM Evaluator Settings

Edit `scripts/paper_evaluator_llm.py`:

```python
# Text truncation (to manage costs)
max_chars = 50000  # Increase for longer papers

# Model settings
model="claude-sonnet-4-5-20250929"  # Latest Sonnet 4.5
temperature=0.3  # Adjust for more/less variability
max_tokens=2048

# Dimensional weights
overall = (
    methodology * 0.35 +  # Adjust weights
    novelty * 0.25 +
    clarity * 0.20 +
    significance * 0.20
)
```

## Cost Considerations

### LLM Evaluation Costs

**Per Evaluation:**
- Input tokens: ~5,000-15,000 (paper content)
- Output tokens: ~1,000-2,000 (evaluation)
- Cost: ~$0.01-0.05 per paper

**Cost Optimization:**
- Heuristic mode: Free, instant
- Session history: Reuse previous evaluations
- Text truncation: Limit to 50,000 chars

## Troubleshooting

### API Key Issues

```
❌ Error: ANTHROPIC_API_KEY not found in environment variables.
```

**Solution:**
```bash
# Add to .env file
echo "ANTHROPIC_API_KEY=your_key_here" >> .env

# Or export directly
export ANTHROPIC_API_KEY=your_key_here
```

### Installation Issues

```bash
# Rich library not found
poetry add rich

# python-docx not found
poetry add python-docx

# Anthropic not found
poetry add anthropic
```

### Session Loading Fails

```
❌ Session 20241007_151030 not found.
```

**Check:**
```bash
# List session files
ls ~/.ai-coscientist/chat_history/

# Verify file exists
cat ~/.ai-coscientist/chat_history/session_20241007_151030.json
```

### LLM Evaluation Fails

**Automatic Fallback:**
- System automatically falls back to heuristic evaluation
- Warning displayed in yellow
- Lower confidence score indicated

**Manual Fallback:**
```bash
# Use heuristic mode directly
python scripts/paper_evaluator_llm.py paper.docx --no-llm
```

## Performance

### Evaluation Speed

| Mode | Speed | Accuracy | Cost |
|------|-------|----------|------|
| **LLM** | 10-30s | High (0.92) | $0.01-0.05 |
| **Heuristic** | <1s | Medium (0.65) | Free |

### Memory Usage

- Chat session: ~5-10 MB
- Saved sessions: ~10-50 KB each
- History directory: ~1-5 MB for 100 sessions

## Best Practices

### For Accurate Evaluation

1. **Use LLM mode** for important papers
2. **Provide complete papers** (abstract, methods, results, discussion)
3. **Save sessions** before major changes
4. **Review justifications** to understand scores

### For Cost Efficiency

1. **Use heuristic mode** for quick checks
2. **Reuse sessions** instead of re-evaluating
3. **Truncate long papers** if needed
4. **Batch evaluations** instead of repeated single evals

### For Best UX

1. **Start with "show history"** to see previous work
2. **Save after important evaluations**
3. **Use conversation context** - bot remembers your goals
4. **Ask follow-up questions** - leverage Claude's understanding

## Future Enhancements (Phase 3)

The following features are planned but not yet implemented:

1. **Multi-Paper Comparison**
   - Compare multiple papers side-by-side
   - Ranking and recommendation system
   - Batch evaluation mode

2. **Voice Input Support**
   - Speech-to-text integration
   - Voice mode toggle
   - Hands-free operation

## Migration Guide

### From Original Chatbot

```bash
# Old command
python scripts/chat_reviewer.py

# New command (enhanced version)
python scripts/chat_reviewer_enhanced.py
```

**What's New:**
- All original features retained
- Rich UI automatically enabled
- LLM evaluation automatically used
- Session auto-save enabled
- No configuration changes needed

**Backward Compatibility:**
- Original `chat_reviewer.py` still works
- `paper_evaluator.py` still available for heuristic-only
- No breaking changes to commands

### From Heuristic to LLM

**Before (Heuristic):**
```python
from paper_evaluator import evaluate_paper_file
scores = evaluate_paper_file("paper.docx")
# Heuristic scores only
```

**After (LLM):**
```python
from paper_evaluator_llm import evaluate_paper_file
scores = evaluate_paper_file("paper.docx", use_llm=True)
# LLM evaluation with detailed feedback
```

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/Transconnectome/AI-CoScientist/issues
- **Documentation**: See README.md and PAPER_ENHANCEMENT_GUIDE.md

## Credits

Built on the AI-CoScientist paper enhancement system.
Uses Claude AI for natural language understanding and evaluation.
