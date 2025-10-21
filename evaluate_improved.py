#!/usr/bin/env python3
import sys
sys.path.insert(0, 'scripts')

from section_evaluator import SectionEvaluator, SectionType, EvaluationContext, format_section_evaluation
import json

# Read improved abstract
with open('abstract_improved.txt', 'r') as f:
    improved_text = f.read()

print('🔍 Evaluating IMPROVED abstract with Claude AI...\n')

evaluator = SectionEvaluator()
result = evaluator.evaluate_section(
    SectionType.ABSTRACT,
    improved_text,
    EvaluationContext.STANDALONE
)

print(format_section_evaluation(result))

# Save result
with open('abstract_evaluation_improved.json', 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print('\n✅ Evaluation saved to abstract_evaluation_improved.json')

# Compare with original
try:
    with open('abstract_evaluation.json', 'r') as f:
        original_eval = json.load(f)

    print('\n' + '='*80)
    print('COMPARISON: Original vs Improved')
    print('='*80)
    print(f"Overall Score:   {original_eval['overall']['score']:.2f} → {result['overall']['score']:.2f} ({result['overall']['score'] - original_eval['overall']['score']:+.2f})")
    print(f"Clarity:         {original_eval['clarity']['score']:.2f} → {result['clarity']['score']:.2f} ({result['clarity']['score'] - original_eval['clarity']['score']:+.2f})")
    print(f"Completeness:    {original_eval['completeness']['score']:.2f} → {result['completeness']['score']:.2f} ({result['completeness']['score'] - original_eval['completeness']['score']:+.2f})")
    print(f"Conciseness:     {original_eval['conciseness']['score']:.2f} → {result['conciseness']['score']:.2f} ({result['conciseness']['score'] - original_eval['conciseness']['score']:+.2f})")
    print(f"Impact:          {original_eval['impact']['score']:.2f} → {result['impact']['score']:.2f} ({result['impact']['score'] - original_eval['impact']['score']:+.2f})")
    print(f"Word Count:      230 → 189 (-41 words, -17.8%)")
except:
    pass
