# Transformer Reranker v0

This document describes the optional transformer reranker for semantic memory
search. The feature is default-off and does not change normal memory search
unless it is explicitly enabled by configuration.

## Status

This is an experimental learned memory-selection feature. It is intended for
controlled evaluation, not default production use.

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

## Configure

Set these environment variables, or the equivalent settings values:

```bash
EXPERIMENTAL_TRANSFORMER_RERANK=true
TRANSFORMER_RERANK_MODEL_PATH=/path/to/local/checkpoint
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

These numbers show a reproducible improvement on LoCoMo memory evidence
retrieval. They do not prove broad cross-dataset generalization or downstream
answer-quality improvement.

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

The checkpoint should not be committed to the source repository. Publish it as a
release artifact or store it separately, then point `TRANSFORMER_RERANK_MODEL_PATH`
to the local directory.

The validated v0 checkpoint used during development was based on
`prajjwal1/bert-tiny` and trained for LoCoMo memory evidence reranking.

This branch provides the runtime hook and configuration surface. It does not
bundle a model checkpoint or enable the reranker by default.

## Reproduction Notes

The current checkpoint and generated caches are research artifacts. They should
be distributed separately, for example as a GitHub Release artifact or a
Hugging Face model repository.

For full reproduction, a researcher needs:

- LoCoMo-derived memory evidence bundles
- the transformer reranker training script
- the frozen checkpoint or training recipe
- the real-search evaluation script
- the validation commands and metrics

The research branch currently used for those materials is
`stage-e-transformer-hook-pr`. A future cleanup should move stable reproduction
assets into a dedicated `benchmarks/` or `examples/` package.
