from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .models import ForecastHour, KiteProfile, SpotConfig
from .scoring import USEFUL_END, USEFUL_START, minimum_wind

DEFAULT_UNCERTAINTY = 1.0


def _first_onset(rows: list[tuple[int, float | None]], threshold: float) -> int | None:
    for hour, speed in sorted(rows):
        if USEFUL_START <= hour <= USEFUL_END and speed is not None and speed >= threshold:
            return int(hour)
    return None


def _confidence(scatter: float) -> str:
    if scatter < 1.0:
        return "alta"
    if scatter < 1.5:
        return "media"
    return "bassa"


def _historical_adjustment(
    historical_rows: list[dict[str, Any]],
    threshold: float,
) -> tuple[float, float, str] | None:
    usable = [
        row for row in historical_rows
        if row.get("real_speed") is not None and row.get("forecast_speed") is not None
    ]
    if len(usable) < 10:
        return None

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_day[str(row.get("obs_date"))].append(row)

    errors: list[float] = []
    for rows in by_day.values():
        predicted = _first_onset(
            [(int(row["obs_hour"]), float(row["forecast_speed"])) for row in rows],
            threshold,
        )
        real = _first_onset(
            [(int(row["obs_hour"]), float(row["real_speed"])) for row in rows],
            threshold,
        )
        if predicted is not None and real is not None:
            errors.append(float(real - predicted))

    if len(errors) < 2:
        return None

    bias = statistics.mean(errors)
    scatter = statistics.pstdev(errors)
    uncertainty = min(2.0, max(0.5, scatter))
    return bias, uncertainty, _confidence(scatter)


def estimate_onset(
    hours: list[ForecastHour],
    spot: SpotConfig,
    profile: KiteProfile,
    historical_rows: list[dict] | None = None,
) -> dict:
    """Estimate the first useful thermal hour for the day."""
    del spot  # reserved for future spot-specific onset refinements
    threshold = minimum_wind(profile)
    onset_hour = _first_onset(
        [(hour.datetime.hour, hour.wind_speed_knots) for hour in hours],
        threshold,
    )

    method = "prior"
    uncertainty = DEFAULT_UNCERTAINTY
    confidence = "bassa"

    if historical_rows:
        hist = _historical_adjustment(historical_rows, threshold)
        if hist is not None and onset_hour is not None:
            bias, uncertainty, confidence = hist
            onset_hour = round(onset_hour + bias)
            onset_hour = max(USEFUL_START, min(USEFUL_END, int(onset_hour)))
            method = "historical_bias"
        elif len([r for r in historical_rows if r.get("real_speed") is not None]) >= 10:
            method = "forecast" if onset_hour is not None else "prior"

    onset_label = f"{onset_hour:02d}:00" if onset_hour is not None else None
    return {
        "onset_hour": onset_hour,
        "onset_label": onset_label,
        "uncertainty_hours": round(float(uncertainty), 1),
        "confidence": confidence,
        "method": method,
    }
