import json
from collections import defaultdict

with open(r'd:\ML\Memory\MemOS\a_frame\reference\locomo-benchmark-eval\results\sample0_lychee_eval_scored.json', encoding='utf-8') as f:
    data = json.load(f)

wrong = [d for d in data if d.get('judge_result', {}).get('label') == 'WRONG']

# 分类 wrong cases
patterns = {
    'temporal_relative': [],   # 有正确内容但时间是相对引用
    'retrieval_miss': [],      # 系统说"no info"但GT有答案
    'entity_confusion': [],    # 把A的信息当B回答
    'specificity_loss': [],    # 检索到相关内容但缺少具体细节
    'multihop_inference': [],  # 需要多步推理
    'cat5_hallucination': [],  # Cat5 GT=None 但系统给出了内容
}

for d in wrong:
    cat = d.get('category')
    ans = d.get('answer', '').lower()
    gt = str(d.get('ground_truth', '')).lower()
    q = d.get('question', '').lower()
    ctx = ' '.join(d.get('retrieved_context_text', [])).lower()
    why = d.get('judge_result', {}).get('reasoning', '').lower()
    
    no_info_phrases = ['not available', 'cannot find', 'do not have', 'does not contain', 'not in', 'not something', 'cannot answer']
    relative_time_phrases = ['last saturday', 'yesterday', 'next month', 'last week', 'last weekend', 'this month']
    
    is_no_info = any(p in ans for p in no_info_phrases)
    is_relative = any(p in ans for p in relative_time_phrases)
    is_cat5 = cat == 5
    
    if is_cat5 and gt == 'none' and not is_no_info:
        patterns['cat5_hallucination'].append(d)
    elif is_relative and 'relative' in why:
        patterns['temporal_relative'].append(d)
    elif is_no_info and gt != 'none':
        patterns['retrieval_miss'].append(d)
    elif cat == 3 and 'no information' in ans:
        patterns['multihop_inference'].append(d)
    else:
        patterns['specificity_loss'].append(d)

for k, v in patterns.items():
    print(f'{k}: {len(v)}')
    for d in v[:3]:
        print(f'  Q: {d["question"]}')
        print(f'  GT: {d["ground_truth"]}')
        print(f'  A: {d["answer"][:100]}')
        print()
