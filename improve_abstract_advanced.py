#!/usr/bin/env python3
"""Advanced iterative improvement with multi-candidate generation."""

import sys
sys.path.insert(0, 'scripts')

from anthropic import Anthropic
from section_evaluator import SectionEvaluator, SectionType, EvaluationContext
import json

print('='*80)
print('ADVANCED IMPROVEMENT: MULTI-CANDIDATE GENERATION')
print('='*80)

# Configuration
TARGET_SCORE = 8.5
MAX_ITERATIONS = 5
N_CANDIDATES = 3  # Generate 3 candidates per iteration

# Load API key
with open('.env', 'r') as f:
    for line in f:
        if line.startswith('ANTHROPIC_API_KEY='):
            api_key = line.strip().split('=', 1)[1]
            break

# Read current best version
with open('abstract_improved_iterative.txt', 'r') as f:
    current_text = f.read()

with open('abstract_evaluation_iterative.json', 'r') as f:
    current_eval = json.load(f)

current_score = current_eval['overall']['score']

print(f'\n📄 Starting from: Iterative V1')
print(f'📊 Current Score: {current_score:.2f}/10')
print(f'📝 Word Count: {len(current_text.split())} words')
print(f'\n🎯 Target: {TARGET_SCORE}+')
print(f'🔢 Strategy: Generate {N_CANDIDATES} candidates per iteration, select best')

# Initialize
evaluator = SectionEvaluator()
client = Anthropic(api_key=api_key)

# Multi-candidate strategies
strategies = [
    {
        'name': 'Conciseness-Focused',
        'description': 'Aggressively reduce to 155-165 words while preserving key information',
        'target_words': '155-165',
        'emphasis': 'EXTREME focus on conciseness - remove ALL non-essential words'
    },
    {
        'name': 'Clarity-Focused',
        'description': 'Simplify technical language and improve accessibility',
        'target_words': '165-175',
        'emphasis': 'MAXIMUM clarity - explain technical terms, use simpler language'
    },
    {
        'name': 'Balanced-Optimization',
        'description': 'Balanced improvement across all weak dimensions',
        'target_words': '160-170',
        'emphasis': 'Balanced improvement of clarity AND conciseness equally'
    }
]

best_candidate = None
best_score = current_score

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f'\n{"="*80}')
    print(f'ITERATION {iteration}')
    print(f'{"="*80}')

    if best_score >= TARGET_SCORE:
        print(f'✅ Target {TARGET_SCORE} reached! ({best_score:.2f})')
        break

    candidates = []

    # Generate multiple candidates with different strategies
    for i, strategy in enumerate(strategies, 1):
        print(f'\n🔧 Generating Candidate {i}: {strategy["name"]}')

        prompt = f'''Scientific writing expert: Improve this abstract using the {strategy['name']} strategy.

CURRENT VERSION (score: {current_score:.2f}/10, words: {len(current_text.split())}):
{current_text}

CURRENT WEAK POINTS:
'''

        # Add weak dimensions
        for dim in ['clarity', 'completeness', 'conciseness', 'impact']:
            score = current_eval[dim]['score']
            if score < 8.0:
                prompt += f'- {dim.title()}: {score:.2f}/10 - {current_eval[dim]["justification"][:150]}...\n'

        prompt += f'''
STRATEGY: {strategy['description']}
EMPHASIS: {strategy['emphasis']}
TARGET WORDS: {strategy['target_words']}
TARGET SCORE: {TARGET_SCORE}+

SPECIFIC ACTIONS:
'''

        # Add strategy-specific actions
        if 'Conciseness' in strategy['name']:
            prompt += '''1. Remove ALL redundant phrases and words
2. Combine sentences where possible
3. Use shorter, more direct expressions
4. Cut word count by 10-15 words minimum
5. Maintain all key scientific findings'''

        elif 'Clarity' in strategy['name']:
            prompt += '''1. Simplify technical terminology
2. Add brief context for complex concepts
3. Use more accessible language
4. Improve sentence flow and readability
5. Maintain scientific precision'''

        else:  # Balanced
            prompt += '''1. Simplify language AND reduce word count
2. Remove redundant phrases
3. Clarify technical terms briefly
4. Improve overall readability
5. Maintain scientific completeness'''

        prompt += '\n\nProvide ONLY the improved abstract text.'

        # Generate candidate
        response = client.messages.create(
            model='claude-sonnet-4-5-20250929',
            max_tokens=2048,
            temperature=0.4,  # Slightly higher for diversity
            messages=[{'role': 'user', 'content': prompt}]
        )

        candidate_text = response.content[0].text.strip()

        # Evaluate candidate
        candidate_eval = evaluator.evaluate_section(
            SectionType.ABSTRACT,
            candidate_text,
            EvaluationContext.STANDALONE
        )

        candidate_score = candidate_eval['overall']['score']

        print(f'   Score: {candidate_score:.2f}/10 ({candidate_score - current_score:+.2f})')
        print(f'   Words: {len(candidate_text.split())} ({len(candidate_text.split()) - len(current_text.split()):+d})')

        candidates.append({
            'strategy': strategy['name'],
            'text': candidate_text,
            'evaluation': candidate_eval,
            'score': candidate_score,
            'word_count': len(candidate_text.split())
        })

    # Select best candidate
    best_of_iteration = max(candidates, key=lambda x: x['score'])

    print(f'\n🏆 Best Candidate: {best_of_iteration["strategy"]}')
    print(f'   Score: {best_of_iteration["score"]:.2f}/10 (improvement: {best_of_iteration["score"] - current_score:+.2f})')
    print(f'   Scores: Clarity {best_of_iteration["evaluation"]["clarity"]["score"]:.2f}, ' +
          f'Completeness {best_of_iteration["evaluation"]["completeness"]["score"]:.2f}, ' +
          f'Conciseness {best_of_iteration["evaluation"]["conciseness"]["score"]:.2f}, ' +
          f'Impact {best_of_iteration["evaluation"]["impact"]["score"]:.2f}')

    # Check if improvement
    if best_of_iteration['score'] <= current_score:
        print(f'\n⚠️  No improvement found. Stopping.')
        break

    # Update for next iteration
    current_text = best_of_iteration['text']
    current_eval = best_of_iteration['evaluation']
    current_score = best_of_iteration['score']

    if current_score > best_score:
        best_candidate = best_of_iteration
        best_score = current_score

# Save results if improved
if best_candidate:
    print(f'\n{"="*80}')
    print(f'ADVANCED IMPROVEMENT COMPLETE')
    print(f'{"="*80}')
    print(f'\n🏆 Best Strategy: {best_candidate["strategy"]}')
    print(f'📊 Final Score: {best_score:.2f}/10')
    print(f'📝 Word Count: {best_candidate["word_count"]}')

    with open('abstract_improved_advanced.txt', 'w') as f:
        f.write(best_candidate['text'])

    with open('abstract_evaluation_advanced.json', 'w') as f:
        json.dump(best_candidate['evaluation'], f, indent=2, ensure_ascii=False)

    print(f'\n✅ Files saved:')
    print(f'   • abstract_improved_advanced.txt')
    print(f'   • abstract_evaluation_advanced.json')

    print(f'\n📊 Dimensional Scores:')
    for dim in ['clarity', 'completeness', 'conciseness', 'impact']:
        print(f'   {dim.title():15} {best_candidate["evaluation"][dim]["score"]:.2f}/10')

    if best_score >= TARGET_SCORE:
        print(f'\n🎉 TARGET {TARGET_SCORE}+ ACHIEVED!')
    else:
        print(f'\n⚠️  Target not reached (gap: {TARGET_SCORE - best_score:.2f})')
else:
    print(f'\n⚠️  No improvement achieved with multi-candidate approach.')
