#!/usr/bin/env python3
from anthropic import Anthropic
import os

# Read from .env manually
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('ANTHROPIC_API_KEY='):
            api_key = line.strip().split('=', 1)[1]
            break

with open('abstract_original.txt', 'r') as f:
    original = f.read()

prompt = f'''Scientific writing expert: Improve this abstract based on evaluation feedback.

ORIGINAL ABSTRACT (230 words, Overall Score: 6.90/10):
{original}

EVALUATION SCORES:
- Clarity: 6.50/10 (needs improvement)
- Completeness: 8.50/10 (good, but missing age range)
- Conciseness: 5.00/10 (too verbose)
- Impact: 7.50/10 (strong findings, weak presentation)

KEY IMPROVEMENTS:
1. Simplify technical language: Add brief context for "delay discounting" and other terms
2. Streamline: "GPS for cognitive performance, IQ, and educational attainment" → "cognitive polygenic scores"
3. Remove filler: Delete "Moving beyond average relationships"
4. Add specificity: Include children's age range
5. Reduce length: Target 200-210 words (remove 20-30 words)
6. Strengthen opening: Make it more compelling
7. Clarify flow: Reorganize middle section for better logic
8. Emphasize paradoxical finding: Make this surprising result more prominent

REQUIREMENTS:
- Keep all essential scientific findings
- Maintain technical accuracy
- Improve accessibility for broader audience
- Create stronger narrative arc

Provide ONLY the improved abstract text with no explanations or preamble.'''

client = Anthropic(api_key=api_key)

print('🔧 Generating improved abstract with Claude AI...\n')

response = client.messages.create(
    model='claude-sonnet-4-5-20250929',
    max_tokens=2048,
    temperature=0.3,
    messages=[{'role': 'user', 'content': prompt}]
)

improved = response.content[0].text.strip()

with open('abstract_improved.txt', 'w') as f:
    f.write(improved)

print('='*80)
print('IMPROVED ABSTRACT')
print('='*80)
print(improved)
print('='*80)
print(f'\nOriginal: 230 words')
print(f'Improved: {len(improved.split())} words')
print(f'Reduction: {230 - len(improved.split())} words\n')
print('✅ Saved to abstract_improved.txt')
