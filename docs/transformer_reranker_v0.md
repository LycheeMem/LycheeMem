# Transformer Reranker v0

This document describes the optional transformer reranker for semantic memory
search. The feature is default-off and does not change normal memory search
unless it is explicitly enabled by configuration.

## What It Does

The reranker loads a local sequence-classification checkpoint and scores a wider
memory candidate pool. It may replace a small number of baseline top-k results
when the model score is stronger than the weakest selected candidate.

The current v0 policy is conservative:

- local checkpoint only; no Hugging Face network access at runtime
- disabled by default
- maximum replacements defaults to 1
- wide candidate pool defaults to 50
- merge margin defaults to 0.3

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
