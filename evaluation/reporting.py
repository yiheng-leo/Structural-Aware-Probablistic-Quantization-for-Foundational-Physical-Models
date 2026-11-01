"""Shared Poseidon-style JSON records, grouped by dataset name."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def get_metric_increases(
    metrics: Mapping[str, Any], origin_metrics: Mapping[str, Any],
) -> dict[str, float]:
    """Use Poseidon's percentage change rule for error metrics only."""
    excluded = {"model_size", "linear_size"}
    return {
        key: round(100.0 * (float(value) - float(origin_metrics[key])) / float(origin_metrics[key]), 6)
        for key, value in metrics.items()
        if key in origin_metrics
        and key not in excluded
        and isinstance(value, (int, float))
        and isinstance(origin_metrics[key], (int, float))
        and float(origin_metrics[key]) != 0.0
    }


def build_dataset_results(
    origin_model: Path,
    origin_metrics: Mapping[str, Mapping[str, Any]],
    *,
    compressed_model: Path | None = None,
    compressed_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    dataset_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Wrap each flat metric dictionary in the same per-dataset record.

    Metadata describes the evaluation protocol and sample counts; it never
    participates in metric increases. Baseline-only runs omit ``compressed``.
    """
    if (compressed_model is None) != (compressed_metrics is None):
        raise ValueError("compressed_model and compressed_metrics must be supplied together")
    if compressed_metrics is not None and origin_metrics.keys() != compressed_metrics.keys():
        raise ValueError("Original and compressed results must contain the same datasets")

    results = {}
    for dataset, metrics in origin_metrics.items():
        record = {**(metadata or {}), **(dataset_metadata or {}).get(dataset, {})}
        record.update({
            "dataset": dataset,
            "origin": {"model": str(origin_model), "metrics": dict(metrics)},
        })
        if compressed_metrics is not None:
            compressed = compressed_metrics[dataset]
            record["compressed"] = {
                "model": str(compressed_model),
                "metrics": dict(compressed),
                "increase": get_metric_increases(compressed, metrics),
            }
        results[dataset] = record
    return results
