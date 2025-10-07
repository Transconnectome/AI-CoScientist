# Paper Review Chatbot Guide

The AI-CoScientist Paper Review Chatbot provides an interactive, conversational interface for evaluating and improving scientific papers.

## Quick Start

```bash
# Run the chatbot
python scripts/chat_reviewer.py
```

## Features

### 1. Natural Language Interaction

Chat naturally with the bot to:
- ✅ Evaluate paper quality
- ✅ Get improvement suggestions
- ✅ Apply automated enhancements
- ✅ Ask questions about scores
- ✅ Get step-by-step guidance

### 2. Conversational Workflow

```
You: "Review my paper: paper.docx"
Bot: "Analyzing... Score: 7.96/10. Your methodology is strong (7.89)
     but novelty could be improved (7.46). What would you like to
     focus on?"

You: "Help me get to 8.5+"
Bot: "To reach 8.5, I recommend:
     1. Transform title (30 min, +0.3 points)
     2. Add theoretical justification (2 hours, +0.3 points)
     3. Quantify impact (1 hour, +0.2 points)
     Which would you like to start with?"

You: "Do number 2"
Bot: "Adding theoretical justification section...
     Done! New score: 8.34/10 (+0.38 improvement)
     Would you like to apply more enhancements?"

You: "Explain why my novelty score is low"
Bot: "Your novelty score of 7.46 suggests the contribution is
     positioned as incremental rather than transformative.
     Consider:
     - Reframing title as solving a crisis
     - Adding paradigm shift language
     - Emphasizing what's uniquely new
     Would you like help with this?"
```

## Example Conversations

### Example 1: First-Time User

```
You: "Hi, I have a paper I want to improve"

Bot: "Great! I'd be happy to help. To get started, please share
     the path to your paper file. For example:
     'Review my paper: /path/to/paper.docx'"

You: "Review my paper: ~/Desktop/my-paper.docx"

Bot: "📊 Analysis complete!

     Overall Score: 7.85/10 (Good - Respectable journals)

     Strengths:
     ✅ Strong methodology (7.90/10)
     ✅ Clear writing (7.75/10)

     Areas for improvement:
     ⚠️ Novelty (7.40/10) - Positioning could be stronger
     ⚠️ Significance (7.35/10) - Impact needs quantification

     What would you like to work on?"

You: "What's wrong with my novelty score?"

Bot: "A novelty score of 7.40 indicates your contribution is
     perceived as a solid advance but not groundbreaking.

     Common issues:
     • Title uses incremental language ('improved', 'better')
     • Abstract doesn't position as paradigm shift
     • Introduction lacks crisis framing

     To improve this, you could:
     1. Transform title to emphasize solving a problem
     2. Add crisis framing to abstract
     3. Strengthen theoretical contributions

     Would you like specific suggestions for your paper?"
```

### Example 2: Target-Driven Improvement

```
You: "I need this paper to score 8.5+ for a top journal"

Bot: "Current score: 7.85/10
     Gap to target: 0.65 points

     Here's a roadmap to reach 8.5:

     Phase 1 (Quick wins - 2 hours):
     1. Transform title → +0.3 points
     2. Quantify all impact → +0.2 points

     Phase 2 (Deeper work - 3 hours):
     3. Add theoretical section → +0.3 points
     4. Improve introduction → +0.2 points

     Total expected: +1.0 points → 8.85/10

     Shall we start with Phase 1?"

You: "Yes, let's do it"

Bot: "Perfect! Let's start with transforming your title.

     Current title (if I can detect it):
     'An Improved Method for...'

     This sounds incremental. For higher impact, consider:
     'Solving the [Crisis] in [Field]: A [Framework]'

     For example:
     'Solving the Reproducibility Crisis in Treatment Effects:
      A Robust Ensemble Framework'

     Would you like me to help craft a new title, or shall
     we move to quantifying impact?"
```

### Example 3: Understanding Scores

```
You: "Why did GPT-4 give me 8.0 but the overall is 7.85?"

Bot: "Good question! Let me explain the ensemble scoring:

     Your scores:
     • GPT-4 (40% weight): 8.00/10 - Narrative quality
     • Hybrid (30% weight): 7.75/10 - Technical depth
     • Multi-task (30% weight): 7.70/10 - Novelty

     Overall = 0.40×8.00 + 0.30×7.75 + 0.30×7.70 = 7.85

     GPT-4 likes your paper's communication and positioning,
     but the local models see room for improvement in
     technical depth and novelty.

     To raise the overall score:
     • Improve technical content → affects Hybrid score
     • Strengthen novelty → affects Multi-task score

     Would you like specific suggestions for either?"
```

### Example 4: Applying Enhancements

```
You: "Add the theoretical justification"

Bot: "Adding theoretical justification section...

     ✅ Created ~1200-word theoretical foundations section
     ✅ Inserted after main results
     ✅ Covers 3 mathematical mechanisms

     Re-evaluating paper...

     📊 New Scores:
     Previous: 7.85/10
     Current:  8.28/10

     Improvement: +0.43 points! 🎉

     Dimensional changes:
     • Methodology: 7.90 → 8.15 (+0.25)
     • Novelty: 7.40 → 7.85 (+0.45)

     Great progress! You're almost at 8.5.
     Want to apply one more enhancement?"

You: "What's the quickest way to hit 8.5?"

Bot: "You need +0.22 more points. Quickest options:

     1. Add comparison table (30 min) → +0.10 expected
     2. Add impact boxes (30 min) → +0.05 expected
     3. Quantify 3-5 impact statements (45 min) → +0.15 expected

     Doing #1 and #3 should get you there!

     Shall I add the comparison table first?"
```

## Available Commands

### Evaluation Commands

```
"Review my paper: <file_path>"
"Evaluate this: <file_path>"
"Analyze my paper: <file_path>"
"Score this paper: <file_path>"
```

### Improvement Commands

```
"Help me improve this"
"Get me to 8.5+"
"I need a score of 9.0"
"Make this better"
"Increase my score"
```

### Enhancement Commands

```
"Add theoretical justification"
"Add the comparison table"
"Add impact boxes"
"Apply [enhancement_name]"
```

### Question Commands

```
"Why did I get this score?"
"Explain my novelty score"
"What does clarity measure?"
"How can I improve methodology?"
"What should I do next?"
```

### Navigation Commands

```
"What's next?"
"Show me options"
"Help"
"Quit" / "Exit"
```

## Understanding Scores

### Overall Score Ranges

| Score | Interpretation | Typical Publication |
|-------|----------------|---------------------|
| 9.0-10.0 | Exceptional | Nature, Science, Cell |
| 8.5-8.9 | Excellent | Top specialty journals |
| 8.0-8.4 | Very Good | Strong specialty journals |
| 7.5-7.9 | Good | Respectable journals |
| 7.0-7.4 | Acceptable | Mid-tier journals |
| <7.0 | Needs Work | Major revisions required |

### Dimensional Scores

**Novelty (7.0-8.0 typical)**:
- Originality of contribution
- Paradigm shift vs incremental
- Theoretical advancement

**Methodology (7.5-8.5 typical)**:
- Experimental rigor
- Validation completeness
- Reproducibility

**Clarity (7.0-8.0 typical)**:
- Writing quality
- Organization
- Communication effectiveness

**Significance (7.0-8.0 typical)**:
- Real-world impact
- Clinical/practical value
- Field advancement

### Model Contributions

**GPT-4 (40% weight)**:
- Evaluates narrative quality
- Assesses communication
- Sensitive to positioning
- Usually highest score

**Hybrid (30% weight)**:
- Evaluates technical depth
- Assesses methodology
- Balanced perspective
- Middle score

**Multi-task (30% weight)**:
- Evaluates novelty
- Assesses contribution
- Most conservative
- Usually lowest score

## Tips for Best Results

### 1. Be Specific

❌ "Make my paper better"
✅ "Help me improve my novelty score to 8.0"

❌ "Review this"
✅ "Review my paper and focus on methodology: paper.docx"

### 2. Provide Context

Good: "I'm submitting to Nature Neuroscience, need 8.5+"
Better: "Target journal requires strong methodology. My current methodology score is 7.8. How can I improve it?"

### 3. Ask Follow-up Questions

```
Bot: "Your novelty score is 7.4"
You: "Why?" ✅
You: "How can I improve it?" ✅
You: "What specific changes would help?" ✅
```

### 4. Use Iterative Improvement

```
Evaluate → Understand → Improve → Re-evaluate → Repeat
```

Don't try to fix everything at once. Focus on one dimension at a time.

### 5. Leverage Bot's Knowledge

Ask questions like:
- "What do top papers in my field do differently?"
- "Show me examples of strong novelty framing"
- "What's the most impactful enhancement I can apply?"

## Troubleshooting

### Bot doesn't understand my request

Try rephrasing:
```
❌ "Do the thing with the theory"
✅ "Add theoretical justification section"

❌ "Make it better"
✅ "Suggest improvements for novelty score"
```

### Scores seem wrong

The bot uses heuristic-based scoring as a demo. For production use, integrate with full ensemble models.

### Enhancement fails

Check:
- File path is correct
- File is readable
- Script has necessary permissions
- Dependencies are installed

### Can't find paper file

Use absolute paths:
```
❌ "paper.docx"
✅ "/Users/username/Desktop/paper.docx"
✅ "~/Desktop/paper.docx"
```

## Advanced Usage

### Batch Processing

```
You: "I have 3 papers to review"
Bot: "Great! Let's review them one by one. Share the first one."
You: "Paper 1: paper1.docx"
[Review and improve]
You: "Next paper: paper2.docx"
[Continue...]
```

### Comparative Analysis

```
You: "Compare paper-v1.docx and paper-v2.docx"
Bot: [Evaluates both and shows improvements]
```

### Custom Target Scores

```
You: "I need novelty 8.0, methodology 8.5, others can stay"
Bot: "Understood. Let's focus on novelty and methodology..."
```

## Integration with Claude Code

To use in Claude Code terminal:

```bash
# 1. Navigate to project
cd /path/to/AI-CoScientist

# 2. Run chatbot
python scripts/chat_reviewer.py

# 3. Chat naturally
💬 You: Review my paper: ../my-project/paper.docx
🤖 Bot: [Analysis and suggestions]
```

## Future Enhancements

Planned features:
- [ ] Voice input support
- [ ] Automatic citation checking
- [ ] Figure/table analysis
- [ ] Multi-paper comparison
- [ ] Export improvement reports
- [ ] Integration with reference managers
- [ ] Real-time collaboration
- [ ] Web interface option

## FAQ

**Q: Is this better than running scripts manually?**
A: Yes! The chatbot provides:
- Natural language interface
- Contextual guidance
- Step-by-step workflow
- Explanation on demand

**Q: Can I use this for any paper?**
A: Yes, the chatbot works with .docx and .txt files across all scientific domains.

**Q: How accurate are the scores?**
A: Current version uses heuristic scoring. For production, integrate with full ensemble models for research-grade accuracy.

**Q: Can I customize suggestions?**
A: Yes, ask specific questions to get tailored advice for your paper's unique needs.

**Q: Does it remember previous conversations?**
A: Within a session, yes. Across sessions, no (unless you implement persistence).

## Support

For issues or questions:
- GitHub Issues: https://github.com/Transconnectome/AI-CoScientist/issues
- Documentation: See PAPER_ENHANCEMENT_GUIDE.md for detailed methodology

## Credits

Built on the AI-CoScientist paper enhancement system.
Uses Claude AI for natural language understanding.
