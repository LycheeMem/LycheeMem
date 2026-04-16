import json
from collections import defaultdict

with open(r'd:\ML\Memory\MemOS\a_frame\reference\locomo-benchmark-eval\results\sample0_lychee_eval_scored.json', encoding='utf-8') as f:
    data = json.load(f)

cat_correct = defaultdict(int)
cat_total = defaultdict(int)
for d in data:
    c = d.get('category')
    jr = d.get('judge_result', {})
    label = jr.get('label', '')
    cat_total[c] += 1
    if label == 'CORRECT':
        cat_correct[c] += 1

cat_names = {1: 'single-hop', 2: 'temporal', 3: 'multi-hop', 4: 'adversarial', 5: 'summarization'}
print('=== Per-category accuracy ===')
for c in sorted(cat_total):
    total = cat_total[c]
    correct = cat_correct[c]
    name = cat_names.get(c, '?')
    print(f'  Cat {c} ({name}): {correct}/{total} = {correct/total:.1%}')

print()
print(f'Overall: {sum(cat_correct.values())}/{sum(cat_total.values())} = {sum(cat_correct.values())/sum(cat_total.values()):.1%}')

print()
print('=== ALL WRONG cases ===')
wrong = [d for d in data if d.get('judge_result', {}).get('label') == 'WRONG']
for i, d in enumerate(wrong):
    q = d.get('question', '')
    a = d.get('answer', '')[:150]
    gt = d.get('ground_truth', '')
    cat = d.get('category')
    ctx = d.get('retrieved_context_text', [])
    jr = d.get('judge_result', {})
    reason = jr.get('reasoning', '')[:250]
    bg = d.get('background_context', '')[:300]
    print(f'[{i+1}] Cat{cat} | Q: {q}')
    print(f'     GT:  {gt}')
    print(f'     ANS: {a}')
    ctx_short = [c[:70] for c in ctx[:3]]
    print(f'     CTX: {ctx_short}')
    print(f'     WHY: {reason}')
    print()
