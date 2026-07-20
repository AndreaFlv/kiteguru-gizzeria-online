from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from .models import ForecastHour
from .utils import degrees_to_cardinal

MIN_DIRECTION_SAMPLES = 20


@dataclass(frozen=True)
class DirectionModel:
    circular_bias_deg: float
    scatter_deg: float
    n_samples: int
    trained: bool


def _signed_angle(deg: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return ((deg + 180.0) % 360.0) - 180.0


def _circular_mean_deg(angles: list[float]) -> float:
    s = sum(math.sin(math.radians(a)) for a in angles) / len(angles)
    c = sum(math.cos(math.radians(a)) for a in angles) / len(angles)
    return _signed_angle(math.degrees(math.atan2(s, c)))


def _circular_std_deg(angles: list[float]) -> float:
    s = sum(math.sin(math.radians(a)) for a in angles) / len(angles)
    c = sum(math.cos(math.radians(a)) for a in angles) / len(angles)
    r = min(1.0, max(0.0, math.hypot(s, c)))
    if r >= 1.0:
        return 0.0
    if r <= 0.0:
        return 180.0
    return math.degrees(math.sqrt(-2.0 * math.log(r)))


def train_direction(conn: sqlite3.Connection, spot_name: str) -> DirectionModel:
    rows = conn.execute(
        """
        SELECT s.forecast_dir_deg, o.real_dir_deg
        FROM observations o
        JOIN forecast_snapshots s
          ON s.spot = o.spot AND s.target_date = o.obs_date AND s.obs_hour = o.obs_hour
        WHERE o.spot = ? AND s.forecast_dir_deg IS NOT NULL AND o.real_dir_deg IS NOT NULL
        """,
        (spot_name,),
    ).fetchall()
    n = len(rows)
    if n < MIN_DIRECTION_SAMPLES:
        return DirectionModel(
            circular_bias_deg=0.0,
            scatter_deg=999.0,
            n_samples=n,
            trained=False,
        )

    errors = [
        _signed_angle(float(row["real_dir_deg"]) - float(row["forecast_dir_deg"]))
        for row in rows
    ]
    bias = _circular_mean_deg(errors)
    scatter = _circular_std_deg(errors)
    return DirectionModel(
        circular_bias_deg=round(bias, 3),
        scatter_deg=round(scatter, 3),
        n_samples=n,
        trained=True,
    )


def apply_direction_correction(
    hours: list[ForecastHour], model: DirectionModel
) -> list[ForecastHour]:
    if not model.trained:
        return hours
    corrected: list[ForecastHour] = []
    for hour in hours:
        direction = (hour.wind_direction_degrees + model.circular_bias_deg) % 360.0
        corrected.append(
            hour.model_copy(
                update={
                    "wind_direction_degrees": direction,
                    "wind_direction_cardinal": degrees_to_cardinal(direction),
                }
            )
        )
    return corrected
