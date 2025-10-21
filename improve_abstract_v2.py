#!/usr/bin/env python3
from anthropic import Anthropic

with open('.env', 'r') as f:
    for line in f:
        if line.startswith('ANTHROPIC_API_KEY='):
            api_key = line.strip().split('=', 1)[1]
            break

with open('abstract_improved.txt', 'r') as f:
    current = f.read()

prompt = f'''Scientific writing expert: Further improve this abstract based on NEW evaluation.

CURRENT VERSION (189 words, still 6.90/10):
{current}

NEW EVALUATION FEEDBACK:
Same score BUT new issues identified:
1. Still too dense/technical despite improvements
2. Tries to cover too many findings - lose focus
3. Missing quantitative results (effect sizes, percentages)
4. Repetitive terms: "vulnerability" (4x), "neighborhood adversity/disadvantage" (4x)
5. "Behavioral poverty trap" introduced but not connected well

SPECIFIC NEW IMPROVEMENTS:
1. **Lead with paradoxical finding as hook** - most striking result first
2. Add ONE quantitative result (e.g., "X% more vulnerable")
3. Simplify IV explanation: "using policy changes as natural experiments"
4. Focus on PLEs as PRIMARY outcome (delay discounting as mechanism)
5. Remove diathesis-stress mention - focus only on differential susceptibility
6. Vary language to reduce repetition
7. Make final sentence more concrete

STRUCTURE TO FOLLOW:
1. Hook: Paradoxical finding (genetic advantage → vulnerability)
2. Context: Why this matters
3. Methods: Brief, simplified (sample, approach)
4. Results: With ONE number
5. Implication: Concrete, actionable

TARGET: ~180-190 words, 7.5+ score

Provide ONLY the improved abstract.'''

client = Anthropic(api_key=api_key)

print('🔧 Generating Version 2 with Claude AI...\n')

response = client.messages.create(
    model='claude-sonnet-4-5-20250929',
    max_tokens=2048,
    temperature=0.3,
    messages=[{'role': 'user', 'content': prompt}]
)

improved_v2 = response.content[0].text.strip()

with open('abstract_improved_v2.txt', 'w') as f:
    f.write(improved_v2)

print('='*80)
print('IMPROVED ABSTRACT V2')
print('='*80)
print(improved_v2)
print('='*80)
print(f'\nV1: 189 words')
print(f'V2: {len(improved_v2.split())} words\n')
print('✅ abstract_improved_v2.txt')
