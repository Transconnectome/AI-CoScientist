#!/usr/bin/env python3
import sys
sys.path.insert(0, 'scripts')

from section_evaluator import SectionEvaluator, SectionType, EvaluationContext, format_section_evaluation
import json

# Read v2 abstract
with open('abstract_improved_v2.txt', 'r') as f:
    v2_text = f.read()

print('🔍 Evaluating VERSION 2 abstract with Claude AI...\n')

evaluator = SectionEvaluator()
result = evaluator.evaluate_section(
    SectionType.ABSTRACT,
    v2_text,
    EvaluationContext.STANDALONE
)

print(format_section_evaluation(result))

# Save result
with open('abstract_evaluation_v2.json', 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print('\n✅ Saved: abstract_evaluation_v2.json')

# Full comparison
print('\n' + '='*80)
print('FULL COMPARISON: Original → V1 → V2')
print('='*80)

with open('abstract_evaluation.json', 'r') as f:
    original = json.load(f)

with open('abstract_evaluation_improved.json', 'r') as f:
    v1 = json.load(f)

print(f"\n📊 Overall Score:")
print(f"   Original: {original['overall']['score']:.2f}/10")
print(f"   V1:       {v1['overall']['score']:.2f}/10 ({v1['overall']['score'] - original['overall']['score']:+.2f})")
print(f"   V2:       {result['overall']['score']:.2f}/10 ({result['overall']['score'] - original['overall']['score']:+.2f})")

print(f"\n📋 Dimensional Scores:")
for dim in ['clarity', 'completeness', 'conciseness', 'impact']:
    print(f"   {dim.title():15} {original[dim]['score']:.2f} → {v1[dim]['score']:.2f} → {result[dim]['score']:.2f}")

print(f"\n📝 Word Count:")
print(f"   Original: 230 words")
print(f"   V1:       189 words (-41, -17.8%)")
print(f"   V2:       170 words (-60, -26.1%)")

print(f"\n🎯 Key Improvements V2:")
for i, strength in enumerate(result.get('strengths', [])[:5], 1):
    print(f"   {i}. {strength}")
