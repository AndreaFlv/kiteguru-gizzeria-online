"""Metriche operative leakage-safe per previsioni di vento.

Le funzioni sono pure: nessuna dipendenza da DB, provider o dashboard. Questo
mantiene separata la politica di valutazione dagli adattatori infrastrutturali.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean, median
from typing import Iterable, Mapping, Any


def threshold_outcome(predicted: float, actual: float, threshold: float = 13.0) -> str:
    """TP/TN/FP/FN rispetto alla soglia operativa di vento medio."""
    pred_positive = float(predicted) >= threshold
    actual_positive = float(actual) >= threshold
    if pred_positive and actual_positive:
        return "TP"
    if not pred_positive and not actual_positive:
        return "TN"
    return "FP" if pred_positive else "FN"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return (numerator / denominator) if denominator else None


def _bootstrap_mean_interval(
    values: list[float], *, samples: int = 2000, seed: int = 42,
) -> list[float] | None:
    """Deterministic percentile interval for a mean, suitable for small audits."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)
    )
    lo = estimates[int(0.025 * (samples - 1))]
    hi = estimates[int(0.975 * (samples - 1))]
    return [lo, hi]


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _group_summary(
    rows: list[dict[str, Any]], prediction_key: str, actual_key: str, threshold: float,
) -> dict[str, Any]:
    signed = [float(row[prediction_key]) - float(row[actual_key]) for row in rows]
    absolute = [abs(value) for value in signed]
    outcomes = [
        threshold_outcome(row[prediction_key], row[actual_key], threshold)
        for row in rows
    ]
    return {
        "n": len(rows),
        "mae_kn": mean(absolute) if absolute else None,
        "rmse_kn": math.sqrt(mean(value * value for value in signed)) if signed else None,
        "bias_kn": mean(signed) if signed else None,
        "median_absolute_error_kn": median(absolute) if absolute else None,
        "true_positive": outcomes.count("TP"),
        "true_negative": outcomes.count("TN"),
        "false_positive": outcomes.count("FP"),
        "false_negative": outcomes.count("FN"),
    }


def operational_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    prediction_key: str = "predicted",
    actual_key: str = "actual",
    date_key: str = "date",
    hour_key: str = "hour",
    probability_key: str | None = None,
    lower_key: str | None = None,
    upper_key: str | None = None,
    expected_hours_per_day: int | None = None,
    threshold: float = 13.0,
) -> dict[str, Any]:
    """Riepilogo orario e giornaliero, inclusi falsi negativi e copertura.

    Una giornata e' positiva se almeno un'ora raggiunge la soglia. Le giornate
    senza coppie complete non entrano nel denominatore.
    """
    usable = [
        dict(row) for row in records
        if row.get(prediction_key) is not None and row.get(actual_key) is not None
    ]
    outcomes = [
        threshold_outcome(row[prediction_key], row[actual_key], threshold)
        for row in usable
    ]
    signed_errors = [
        float(row[prediction_key]) - float(row[actual_key]) for row in usable
    ]
    errors = [abs(value) for value in signed_errors]
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_day[str(row.get(date_key) or "unknown")].append(row)

    observed_days = len(by_day)
    partial_days_excluded = 0
    if expected_hours_per_day is None:
        evaluated_days = by_day
    else:
        evaluated_days = {
            day: rows for day, rows in by_day.items()
            if len({row.get(hour_key) for row in rows}) == expected_hours_per_day
        }
        partial_days_excluded = observed_days - len(evaluated_days)

    day_hits = 0
    day_false_positive = 0
    day_false_negative = 0
    for rows in evaluated_days.values():
        predicted_positive = any(float(row[prediction_key]) >= threshold for row in rows)
        actual_positive = any(float(row[actual_key]) >= threshold for row in rows)
        if predicted_positive == actual_positive:
            day_hits += 1
        elif predicted_positive:
            day_false_positive += 1
        else:
            day_false_negative += 1
    n = len(usable)
    days = len(evaluated_days)
    tp, tn = outcomes.count("TP"), outcomes.count("TN")
    fp, fn = outcomes.count("FP"), outcomes.count("FN")
    positives = tp + fn

    by_hour: dict[str, dict[str, Any]] = {}
    hour_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        if row.get(hour_key) is not None:
            hour_groups[str(row[hour_key])].append(row)
    for hour, rows in sorted(hour_groups.items(), key=lambda item: int(item[0])):
        by_hour[hour] = _group_summary(rows, prediction_key, actual_key, threshold)

    probability_rows = [
        row for row in usable
        if probability_key is not None and row.get(probability_key) is not None
    ]
    brier_score = None
    if probability_rows:
        brier_score = mean(
            (float(row[probability_key]) - (float(row[actual_key]) >= threshold)) ** 2
            for row in probability_rows
        )

    interval_rows = [
        row for row in usable
        if lower_key is not None and upper_key is not None
        and row.get(lower_key) is not None and row.get(upper_key) is not None
    ]
    interval_coverage = None
    mean_interval_width = None
    if interval_rows:
        interval_coverage = mean(
            float(row[lower_key]) <= float(row[actual_key]) <= float(row[upper_key])
            for row in interval_rows
        )
        mean_interval_width = mean(
            float(row[upper_key]) - float(row[lower_key]) for row in interval_rows
        )

    return {
        "threshold_kn": threshold,
        "hours": n,
        "hour_hits": tp + tn,
        "hour_misses": fp + fn,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "hour_accuracy": (sum(value in {"TP", "TN"} for value in outcomes) / n) if n else None,
        "mae_kn": mean(errors) if errors else None,
        "mae_95ci_kn": _bootstrap_mean_interval(errors),
        "rmse_kn": math.sqrt(mean(value * value for value in signed_errors)) if signed_errors else None,
        "bias_kn": mean(signed_errors) if signed_errors else None,
        "median_absolute_error_kn": median(errors) if errors else None,
        "median_signed_error_kn": median(signed_errors) if signed_errors else None,
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, positives),
        "recall_95ci": _wilson_interval(tp, positives),
        "false_negative_rate": _safe_ratio(fn, positives),
        "false_negative_rate_95ci": _wilson_interval(fn, positives),
        "specificity": _safe_ratio(tn, tn + fp),
        "by_hour": by_hour,
        "probability_pairs": len(probability_rows),
        "brier_score": brier_score,
        "interval_pairs": len(interval_rows),
        "interval_coverage": interval_coverage,
        "mean_interval_width_kn": mean_interval_width,
        "observed_days": observed_days,
        "complete_days": days,
        "partial_days_excluded": partial_days_excluded,
        "days": days,
        "day_hits": day_hits,
        "day_misses": day_false_positive + day_false_negative,
        "day_false_positive": day_false_positive,
        "day_false_negative": day_false_negative,
        "day_accuracy": (day_hits / days) if days else None,
    }
