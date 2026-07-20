from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from .models import ForecastHour, SpotConfig
from .scoring import USEFUL_END, USEFUL_START, direction_category

Regime = Literal["TERMICO", "SINOTTICO", "MISTO", "VENTURI", "ANOMALO"]

MIN_DELTA_SAMPLES = 10
THERMAL_REGIME_START = 13
THERMAL_REGIME_END = 17
MIN_THERMAL_RADIATION = 400.0

_GIZZERIA_REGIME_SPOT = SpotConfig(
    name="Gizzeria Lido",
    latitude=38.928,
    longitude=16.209,
    timezone="Europe/Rome",
    preferred_directions=["W", "WSW", "WNW"],
    acceptable_directions=["SW", "NW"],
    bad_directions=["E", "ENE", "ESE", "SE", "SSE", "NE", "NNE"],
)


@dataclass(frozen=True)
class DeltaStats:
    n_samples: int
    mean_delta: float
    p10: float
    p50: float
    p90: float
    by_hour: dict[int, float]
    by_month: dict[int, float]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def _group_mean(groups: dict[int, list[float]]) -> dict[int, float]:
    return {key: round(mean(values), 3) for key, values in groups.items() if values}


def compute_delta_stats(conn: sqlite3.Connection, spot_name: str) -> DeltaStats | None:
    rows = conn.execute(
        """
        SELECT o.obs_hour, o.real_speed, s.forecast_speed,
               strftime('%m', o.obs_date) AS month
        FROM observations o
        JOIN forecast_snapshots s
          ON s.spot = o.spot AND s.target_date = o.obs_date AND s.obs_hour = o.obs_hour
        WHERE o.spot = ? AND o.real_speed IS NOT NULL AND s.forecast_speed IS NOT NULL
        """,
        (spot_name,),
    ).fetchall()
    if len(rows) < MIN_DELTA_SAMPLES:
        return None

    deltas: list[float] = []
    by_hour: dict[int, list[float]] = defaultdict(list)
    by_month: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        delta = float(row["real_speed"]) - float(row["forecast_speed"])
        hour = int(row["obs_hour"])
        month = int(row["month"])
        deltas.append(delta)
        by_hour[hour].append(delta)
        by_month[month].append(delta)

    return DeltaStats(
        n_samples=len(deltas),
        mean_delta=round(mean(deltas), 3),
        p10=round(_percentile(deltas, 0.10), 3),
        p50=round(_percentile(deltas, 0.50), 3),
        p90=round(_percentile(deltas, 0.90), 3),
        by_hour=_group_mean(by_hour),
        by_month=_group_mean(by_month),
    )


def _is_thermal_direction(hour: ForecastHour) -> bool:
    category = direction_category(hour.wind_direction_cardinal, _GIZZERIA_REGIME_SPOT)
    return category in {"preferred", "acceptable"}


def classify_hour(
    hour: ForecastHour,
    stats: DeltaStats | None,
    regional: dict,
) -> Regime:
    if stats is None:
        return "ANOMALO"

    hour_value = hour.datetime.hour
    delta_expected = stats.by_hour.get(hour_value, stats.p50)
    radiation = hour.radiation or 0.0
    in_thermal_window = THERMAL_REGIME_START <= hour_value <= THERMAL_REGIME_END
    thermal_candidate = (
        in_thermal_window
        and _is_thermal_direction(hour)
        and radiation >= MIN_THERMAL_RADIATION
    )
    anomalous_candidate = delta_expected > stats.p90
    venturi_candidate = (
        stats.p50 <= delta_expected <= stats.p90
        and not in_thermal_window
    )
    cross_isthmus = regional.get("cross_isthmus") if regional else None
    mixed_regional_candidate = (
        thermal_candidate
        and cross_isthmus is not None
        and float(cross_isthmus) > 0.0
        and delta_expected >= stats.p50
    )

    if anomalous_candidate:
        return "ANOMALO"
    if delta_expected < stats.p10:
        return "SINOTTICO"
    if thermal_candidate and (venturi_candidate or mixed_regional_candidate):
        return "MISTO"
    if thermal_candidate:
        return "TERMICO"
    if venturi_candidate:
        return "VENTURI"
    return "MISTO"


def classify_day(
    hours: list[ForecastHour],
    stats: DeltaStats | None,
    regional_by_hour: dict[int, dict],
) -> dict[int, Regime]:
    regimes: dict[int, Regime] = {}
    for hour in hours:
        hour_value = hour.datetime.hour
        if USEFUL_START <= hour_value <= USEFUL_END:
            regimes[hour_value] = classify_hour(
                hour,
                stats,
                regional_by_hour.get(hour_value, {}),
            )
    return regimes
