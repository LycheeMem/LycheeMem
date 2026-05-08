# Transformer Reranker v0

This document describes the optional transformer reranker for semantic memory
search. The feature is default-off and does not change normal memory search
unless it is explicitly enabled by configuration.

## Status

This is an optional learned memory-selection feature. It is suitable as a
default-off release candidate: users must explicitly install rerank
dependencies, download a local checkpoint, and enable the experiment flag before
it changes memory search behavior.

The source branch provides:

- a default-off runtime hook in semantic memory search
- local checkpoint loading
- optional `torch` / `transformers` dependencies
- replacement diagnostics
- validation notes for the current v0 checkpoint

The source branch does not bundle:

- a model checkpoint
- LoCoMo data
- generated bundle caches
- training outputs

## What It Does

The reranker loads a local sequence-classification checkpoint and scores a wider
memory candidate pool. It may replace a small number of baseline top-k results
when an outside candidate's transformer score is higher than the weakest
baseline candidate's transformer score by at least `merge_margin`.

The current v0 policy is conservative:

- local checkpoint only; no Hugging Face network access at runtime
- disabled by default
- maximum replacements defaults to 1
- wide candidate pool defaults to 50
- merge margin defaults to 0.3

In plain terms: baseline search still retrieves memories first. The reranker
only gets a narrow chance to correct the final top-k selection when a better
evidence candidate already appears in the wider candidate pool.

## Install

The base package does not require PyTorch or Transformers. Install the optional
extra only when using the reranker:

```bash
pip install "lycheemem[rerank]"
```

Download the current v0 checkpoint from Hugging Face:

```bash
huggingface-cli download LycheeMem/reranker \
  --local-dir ./lycheemem-reranker-v0
```

## Configure

Set these environment variables, or the equivalent settings values:

```bash
EXPERIMENTAL_TRANSFORMER_RERANK=true
TRANSFORMER_RERANK_MODEL_PATH=./lycheemem-reranker-v0
TRANSFORMER_RERANK_MAX_REPLACEMENTS=1
TRANSFORMER_RERANK_MERGE_MARGIN=0.3
TRANSFORMER_RERANK_WIDE_TOP_K=50
TRANSFORMER_RERANK_DEVICE=auto
```

If the checkpoint is missing or cannot be loaded, the hook disables itself and
search continues with baseline behavior.

## Validation Summary

The current v0 checkpoint was evaluated on LoCoMo evidence retrieval. The metric
is whether the correct evidence appears in the returned top-10 memories
(`hit@10`), not final LLM answer quality.

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

External zero-shot evidence-selection fixtures, same checkpoint and same
runtime policy:

```text
dataset                         cases  baseline@10  wide     v0@10    added/lost/net
LongMemEval-S cleaned           500    469          500      484      +16/-1/+15
MSC-MemFuse-MC10 turn-level     299    142          279      152      +10/-0/+10
HotpotQA distractor sentence    7405   6957         7405     7076     +141/-22/+119
```

These numbers show a reproducible improvement on LoCoMo memory evidence
retrieval and positive zero-shot evidence-selection transfer on three external
fixture shapes. They do not prove downstream answer-quality improvement.

HotpotQA has a weaker safety profile than the memory-style fixtures
(`+141/-22`, ratio `6.41`), so deployments should keep diagnostics enabled and
avoid treating rerank as an always-safe replacement for baseline search.

## Diagnostics

`SemanticSearchResult.diagnostics` includes transformer rerank state:

```text
transformer_rerank_enabled
transformer_rerank_available
transformer_rerank_fired
transformer_rerank_reason
transformer_rerank_model_path
transformer_rerank_device
transformer_rerank_wide_count
transformer_rerank_replacements
transformer_rerank_added_ids
transformer_rerank_removed_ids
transformer_rerank_scores
transformer_rerank_latency_ms
```

Selected candidates also include `transformer_rerank_*` fields in
`score_breakdown`.

## Model Artifact

The checkpoint is distributed separately from the source repository:

```text
https://huggingface.co/LycheeMem/reranker
```

Download it locally, then point `TRANSFORMER_RERANK_MODEL_PATH` to that local
directory.

The validated v0 checkpoint used during development was based on
`prajjwal1/bert-tiny` and trained for LoCoMo memory evidence reranking.

The source tree provides the runtime hook and configuration surface. It does not
bundle a model checkpoint or enable the reranker by default.

## Reproduction Notes

Generated caches and training outputs are research artifacts and are not bundled
with the source tree.

For full reproduction, a researcher needs:

- LoCoMo-derived memory evidence bundles
- the transformer reranker training script
- the frozen checkpoint or training recipe
- the real-search evaluation script
- the validation commands and metrics

The release checkpoint is available on Hugging Face. Full training reproduction
still requires the LoCoMo-derived evidence bundles and evaluation scripts.
