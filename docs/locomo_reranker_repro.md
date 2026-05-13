# LoCoMo Reranker Reproduction Notes

This note explains what was evaluated for transformer reranker v0 and what a
researcher needs to reproduce the result. It is intentionally separate from the
runtime hook documentation: the hook explains how the system loads a model, while
this note explains where the current evidence comes from.

## Task

The evaluation task is LoCoMo memory evidence retrieval.

Given a question and a memory store built from LoCoMo conversations, the system
returns top-k memory candidates. A case is counted as a hit when at least one
gold evidence item appears in the returned top-k memories.

The main metric is:

```text
hit@10 = number of questions whose top-10 memories contain gold evidence
```

This is not a final answer-quality metric. It measures whether `/memory/search`
retrieves the right evidence for the downstream LLM.

## Model

The current v0 model is a lightweight pretrained transformer reranker:

```text
base model: prajjwal1/bert-tiny
task: query-candidate sequence classification
input: question + candidate memory text
output: evidence relevance score
merge policy: at most 1 replacement from wide candidates into baseline top-k
```

The frozen v0 checkpoint used during development was:

```text
.cache/locomo_bert_tiny_reranker_10sall_seed99
```

This path is local and should not be committed to git. The checkpoint should be
published separately if external reproduction is required.

## Data Shape

A reproducible bundle needs this information per QA case:

```text
question
answer
gold evidence ids
baseline top-k candidates
wide candidates, usually top50
candidate memory text and ids
```

Datasets without evidence ids can be useful for auxiliary training or behavior
diagnosis, but they should not be used as the main evidence-retrieval accuracy
claim.

## Reported Results

The current strongest evidence for v0 is:

```text
LoCoMo memory backend, 200 QA:
  baseline hit@10: 124/200 = 0.620
  v0 hit@10:       130/200 = 0.650
  added/lost/net:  +7/-1/+6

LoCoMo LanceDB backend, 200 QA:
  baseline hit@10: 124/200 = 0.620
  v0 hit@10:       131/200 = 0.655
  added/lost/net:  +8/-1/+7

LoCoMo full-memory cache, 5 seeds:
  held added/lost/net: +115/-7/+108
  added/lost ratio:   16.43

LoCoMo split evaluation:
  interleave held:            466/765 -> 495/765, net +29
  prefix held:                473/766 -> 501/766, net +28
  conversation-heldout held:  476/772 -> 504/772, net +28
```

Interpretation: v0 consistently improves LoCoMo evidence hit@10 in the tested
settings. The claim is scoped to LoCoMo evidence retrieval, not broad
cross-dataset memory quality.

## Current Reproduction Entry Points

In the research branch, the relevant scripts are:

```text
examples/locomo_build_bundle_cache.py
examples/locomo_transformer_reranker.py
examples/locomo_frozen_reranker_eval.py
examples/locomo_real_search_eval.py
examples/locomo_reranker_outcome_summary.py
```

Cold or failed directions should not be treated as the v0 reproduction path:

```text
tiny encoder / tiny reranker
BM25 probe
torch model trained from scratch
consolidation benchmark
utility rerank benchmark
```

## Limits

Current known limits:

- External zero-shot evidence-selection fixtures have been validated on
  LongMemEval-S, MSC-MemFuse-MC10, and HotpotQA.
- The checkpoint is distributed through Hugging Face, not bundled in the source
  branch.
- The result measures memory evidence retrieval, not final answer quality.
- The reranker is enabled by default when `lycheemem[rerank]` is installed, but
  still safely falls back to baseline search when dependencies or checkpoint
  loading fail.

## Next Reproduction Cleanup

Before presenting this as a fully reproducible benchmark package, move stable
scripts and docs into a clearer structure, for example:

```text
benchmarks/locomo_reranker/
  build_bundle_cache.py
  train_transformer_reranker.py
  eval_frozen_reranker.py
  real_search_eval.py
```

Also publish the v0 checkpoint as a separate artifact and record its checksum.
