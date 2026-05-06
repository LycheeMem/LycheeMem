"""Runtime transformer reranker for experimental memory search.

This module is intentionally default-off and local-checkpoint only. It mirrors
the LoCoMo offline transformer probe: score a wide candidate pool, then make a
small conservative replacement in the baseline top-k.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TransformerRerankDecision:
    fired: bool
    final_ids: list[str]
    replacements: int
    added_ids: list[str]
    removed_ids: list[str]
    scores: dict[str, float]
    reason: str = ""
    latency_ms: float = 0.0


def candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("id") or candidate.get("record_id") or "").strip()


def candidate_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("semantic_text") or candidate.get("normalized_text") or "").strip()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TransformerReranker:
    """Loaded local transformer checkpoint for search-time reranking."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "auto",
        max_len: int = 192,
        score_batch_size: int = 64,
    ) -> None:
        self.model_path = str(model_path)
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"transformer reranker checkpoint not found: {path}")
        self.device_name = resolve_device(device)
        self.max_len = int(max_len)
        self.score_batch_size = max(1, int(score_batch_size))

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
        ).to(torch.device(self.device_name))
        self.model.eval()

    def score_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[tuple[float, dict[str, Any]]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        with self._torch.no_grad():
            for start in range(0, len(candidates), self.score_batch_size):
                chunk = candidates[start : start + self.score_batch_size]
                encoded = self.tokenizer(
                    [query] * len(chunk),
                    [candidate_text(candidate) for candidate in chunk],
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.model.device)
                    for key, value in encoded.items()
                }
                logits = self.model(**encoded).logits.squeeze(-1)
                scores = self._torch.sigmoid(logits).detach().cpu().tolist()
                scored.extend((float(score), candidate) for score, candidate in zip(scores, chunk, strict=False))
        return sorted(scored, key=lambda item: item[0], reverse=True)

    def decide(
        self,
        *,
        query: str,
        baseline_id_list: list[str],
        wide_candidates: list[dict[str, Any]],
        top_k: int,
        max_replacements: int = 1,
        merge_margin: float = 0.0,
    ) -> TransformerRerankDecision:
        started = time.perf_counter()
        baseline_top = [rid for rid in baseline_id_list[:top_k] if rid]
        baseline_ids = set(baseline_top)
        if not baseline_top:
            return self._decision(False, baseline_top, [], [], {}, "empty_baseline", started)

        candidate_by_id = {
            candidate_id(candidate): candidate
            for candidate in wide_candidates
            if candidate_id(candidate)
        }
        if not candidate_by_id:
            return self._decision(False, baseline_top, [], [], {}, "empty_wide_candidates", started)

        scored = self.score_candidates(query, list(candidate_by_id.values()))
        score_by_id = {
            candidate_id(candidate): score
            for score, candidate in scored
            if candidate_id(candidate)
        }
        current = list(baseline_top)
        current_set = set(current)
        replacements = 0

        for score, candidate in scored:
            cid = candidate_id(candidate)
            if not cid or cid in current_set:
                continue
            if replacements >= max(0, int(max_replacements)):
                break
            replaceable = [rid for rid in current if rid in score_by_id]
            if not replaceable:
                break
            weakest = min(replaceable, key=lambda rid: score_by_id[rid])
            weakest_score = score_by_id[weakest]
            if score <= weakest_score + max(0.0, float(merge_margin)):
                continue
            current[current.index(weakest)] = cid
            current_set.remove(weakest)
            current_set.add(cid)
            replacements += 1

        added_ids = [rid for rid in current if rid not in baseline_ids]
        removed_ids = [rid for rid in baseline_top if rid not in set(current)]
        fired = bool(added_ids)
        reason = "reranked" if fired else "no_replacement"
        selected_ids = set(current) | set(added_ids) | set(removed_ids)
        selected_scores = {
            rid: round(float(score_by_id[rid]), 6)
            for rid in selected_ids
            if rid in score_by_id
        }
        return self._decision(fired, current, added_ids, removed_ids, selected_scores, reason, started)

    @staticmethod
    def _decision(
        fired: bool,
        final_ids: list[str],
        added_ids: list[str],
        removed_ids: list[str],
        scores: dict[str, float],
        reason: str,
        started: float,
    ) -> TransformerRerankDecision:
        return TransformerRerankDecision(
            fired=fired,
            final_ids=list(final_ids),
            replacements=len(added_ids),
            added_ids=list(added_ids),
            removed_ids=list(removed_ids),
            scores=dict(scores),
            reason=reason,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
