"""Metriche operative leakage-safe per previsioni di vento.

Le funzioni sono pure: nessuna dipendenza da DB, provider o dashboard. Questo
mantiene separata la politica di valutazione dagli adattatori infrastrutturali.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
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


def operational_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    prediction_key: str = "predicted",
    actual_key: str = "actual",
    date_key: str = "date",
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
    errors = [abs(float(row[prediction_key]) - float(row[actual_key])) for row in usable]
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_day[str(row.get(date_key) or "unknown")].append(row)
    day_hits = 0
    day_false_positive = 0
    day_false_negative = 0
    for rows in by_day.values():
        predicted_positive = any(float(row[prediction_key]) >= threshold for row in rows)
        actual_positive = any(float(row[actual_key]) >= threshold for row in rows)
        if predicted_positive == actual_positive:
            day_hits += 1
        elif predicted_positive:
            day_false_positive += 1
        else:
            day_false_negative += 1
    n = len(usable)
    days = len(by_day)
    return {
        "threshold_kn": threshold,
        "hours": n,
        "hour_hits": sum(value in {"TP", "TN"} for value in outcomes),
        "hour_misses": sum(value in {"FP", "FN"} for value in outcomes),
        "true_positive": outcomes.count("TP"),
        "true_negative": outcomes.count("TN"),
        "false_positive": outcomes.count("FP"),
        "false_negative": outcomes.count("FN"),
        "hour_accuracy": (sum(value in {"TP", "TN"} for value in outcomes) / n) if n else None,
        "mae_kn": mean(errors) if errors else None,
        "days": days,
        "day_hits": day_hits,
        "day_misses": day_false_positive + day_false_negative,
        "day_false_positive": day_false_positive,
        "day_false_negative": day_false_negative,
        "day_accuracy": (day_hits / days) if days else None,
    }
