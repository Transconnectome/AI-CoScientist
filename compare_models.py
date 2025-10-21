import json

# Load results
with open('improvement_history_ultimate.json', 'r') as f:
    sonnet_history = json.load(f)

with open('improvement_history_gpt5.json', 'r') as f:
    gpt5_history = json.load(f)

print('='*80)
print('MODEL COMPARISON: Claude Sonnet 4.5 vs GPT-5')
print('='*80)

# Get best versions
sonnet_best = max(sonnet_history, key=lambda x: x['weighted_score'])
gpt5_best = max(gpt5_history, key=lambda x: x['weighted_score'])

print('\n📊 FINAL SCORES')
print('-'*80)
print(f'{"Metric":<25} {"Sonnet 4.5":<15} {"GPT-5":<15} {"Winner":<15}')
print('-'*80)

# Overall scores
weighted_diff = gpt5_best['weighted_score'] - sonnet_best['weighted_score']
weighted_winner = 'GPT-5' if weighted_diff > 0 else 'Sonnet 4.5' if weighted_diff < 0 else 'TIE'
print(f'{"Weighted Score":<25} {sonnet_best["weighted_score"]:<15.2f} {gpt5_best["weighted_score"]:<15.2f} {weighted_winner:<15}')

standard_diff = gpt5_best['standard_score'] - sonnet_best['standard_score']
standard_winner = 'GPT-5' if standard_diff > 0 else 'Sonnet 4.5' if standard_diff < 0 else 'TIE'
print(f'{"Standard Score":<25} {sonnet_best["standard_score"]:<15.2f} {gpt5_best["standard_score"]:<15.2f} {standard_winner:<15}')

word_diff = gpt5_best['word_count'] - sonnet_best['word_count']
word_winner = 'Sonnet 4.5' if word_diff > 0 else 'GPT-5' if word_diff < 0 else 'TIE'  # Lower is better
print(f'{"Word Count":<25} {sonnet_best["word_count"]:<15} {gpt5_best["word_count"]:<15} {word_winner:<15}')

print('\n📈 DIMENSIONAL SCORES')
print('-'*80)
print(f'{"Dimension":<25} {"Sonnet 4.5":<15} {"GPT-5":<15} {"Winner":<15}')
print('-'*80)

for dim in ['clarity', 'completeness', 'conciseness', 'impact']:
    sonnet_score = sonnet_best['scores_by_dimension'][dim]
    gpt5_score = gpt5_best['scores_by_dimension'][dim]
    winner = 'GPT-5' if gpt5_score > sonnet_score else 'Sonnet 4.5' if gpt5_score < sonnet_score else 'TIE'
    print(f'{dim.title():<25} {sonnet_score:<15.2f} {gpt5_score:<15.2f} {winner:<15}')

print('\n🎯 IMPROVEMENT FROM ORIGINAL')
print('-'*80)
print(f'{"Metric":<25} {"Sonnet 4.5":<15} {"GPT-5":<15}')
print('-'*80)

sonnet_orig = sonnet_history[0]
gpt5_orig = gpt5_history[0]

weighted_improvement_sonnet = sonnet_best['weighted_score'] - sonnet_orig['weighted_score']
weighted_improvement_gpt5 = gpt5_best['weighted_score'] - gpt5_orig['weighted_score']
print(f'{"Weighted Improvement":<25} {weighted_improvement_sonnet:<15.2f} {weighted_improvement_gpt5:<15.2f}')

standard_improvement_sonnet = sonnet_best['standard_score'] - sonnet_orig['standard_score']
standard_improvement_gpt5 = gpt5_best['standard_score'] - gpt5_orig['standard_score']
print(f'{"Standard Improvement":<25} {standard_improvement_sonnet:<15.2f} {standard_improvement_gpt5:<15.2f}')

word_reduction_sonnet = sonnet_orig['word_count'] - sonnet_best['word_count']
word_reduction_gpt5 = gpt5_orig['word_count'] - gpt5_best['word_count']
print(f'{"Word Reduction":<25} {word_reduction_sonnet:<15} {word_reduction_gpt5:<15}')

print('\n🏆 OVERALL VERDICT')
print('-'*80)

# Count wins
wins = {'Sonnet 4.5': 0, 'GPT-5': 0, 'TIE': 0}
metrics = [weighted_winner, standard_winner, word_winner]
for w in metrics:
    wins[w] += 1

for dim in ['clarity', 'completeness', 'conciseness', 'impact']:
    sonnet_score = sonnet_best['scores_by_dimension'][dim]
    gpt5_score = gpt5_best['scores_by_dimension'][dim]
    if gpt5_score > sonnet_score:
        wins['GPT-5'] += 1
    elif sonnet_score > gpt5_score:
        wins['Sonnet 4.5'] += 1
    else:
        wins['TIE'] += 1

print(f'Sonnet 4.5 wins: {wins["Sonnet 4.5"]}/7')
print(f'GPT-5 wins: {wins["GPT-5"]}/7')
print(f'Ties: {wins["TIE"]}/7')

if wins['GPT-5'] > wins['Sonnet 4.5']:
    print('\n🏅 Winner: GPT-5')
elif wins['Sonnet 4.5'] > wins['GPT-5']:
    print('\n🏅 Winner: Sonnet 4.5')
else:
    print('\n🤝 Result: TIE')

print('\n✅ Comparison Complete!')
