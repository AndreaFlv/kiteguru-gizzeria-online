"""Gradient boosting gated-by-skill per la previsione del vento reale.

Il modello e' opzionale: senza extra `ml` installato, o senza abbastanza dati,
il chiamante riceve `None` e la pipeline resta su analog/fisico.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .analog import Forecast
from .scoring import USEFUL_END, USEFUL_START
from .storage import feature_rows

FEATURE_COLUMNS = (
    "hour",
    "forecast_speed",
    "f_radiation",
    "f_wind_speed",
    "f_wind_dir_deg",
    "dT_land_sea",
    "dT_land_sst",
    "boundary_layer_height_m",
    "cross_isthmus",
    "synoptic_kn",
    "month",
)
MIN_PAIRS = 60
MIN_DISTINCT_DAYS = 6
MIN_RELATIVE_IMPROVEMENT = 0.02


def gb_model_path() -> Path:
    """Percorso del modello serializzato, sovrascrivibile nei test."""
    import os

    override = os.getenv("KITEGURU_GB_MODEL")
    if override:
        return Path(override)
    return Path.home() / ".kiteguru" / "gb_model.pkl"


def _optional_ml():
    try:
        import joblib  # type: ignore
        import lightgbm as lgb  # type: ignore
    except Exception:
        return None, None
    return joblib, lgb


def _month_from(value: Any) -> int:
    if isinstance(value, date):
        return value.month
    if value:
        return int(str(value)[5:7])
    return 6


def _row_to_features(row: dict[str, Any]) -> list[float]:
    month = row.get("month")
    if month is None:
        month = _month_from(row.get("obs_date") or row.get("target_date"))
    return [
        float(row.get("hour") or row.get("obs_hour") or 0),
        float(row.get("forecast_speed") or 0.0),
        float(row.get("f_radiation") or 0.0),
        float(row.get("f_wind_speed") or row.get("forecast_speed") or 0.0),
        float(row.get("f_wind_dir_deg") or 0.0),
        float(row.get("dT_land_sea") or 0.0),
        float(row.get("dT_land_sst") or 0.0),
        float(row.get("boundary_layer_height_m") or 0.0),
        float(row.get("cross_isthmus") or 0.0),
        float(row.get("synoptic_kn") or 0.0),
        float(month),
    ]


@dataclass
class GBModel:
    median_model: Any
    lo_model: Any
    hi_model: Any
    mae_oos: float
    benchmark_mae: float
    n_samples: int
    training_start: str
    training_end: str
    holdout_start: str

    def predict(self, feature_row: dict[str, Any]) -> tuple[float, float, float]:
        x = np.array([_row_to_features(feature_row)], dtype=float)
        median = float(self.median_model.predict(x)[0])
        lo = float(self.lo_model.predict(x)[0])
        hi = float(self.hi_model.predict(x)[0])
        lo, hi = sorted((lo, hi))
        median = min(max(median, lo), hi)
        return round(median, 1), round(lo, 1), round(hi, 1)


def _mae(actual: list[float], pred: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)


def _train_regressor(lgb, X, y, *, objective: str = "regression", alpha: float | None = None):
    params = {
        "objective": objective,
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_child_samples": 5,
        "seed": 42,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    dataset = lgb.Dataset(np.array(X, dtype=float), label=np.array(y, dtype=float), free_raw_data=False)
    return lgb.train(params, dataset, num_boost_round=90)


def train_gb(conn, spot) -> GBModel | None:
    """Allena LightGBM solo se batte il benchmark analog/fisico fuori campione."""
    joblib, lgb = _optional_ml()
    if joblib is None or lgb is None:
        return None

    spot_name = getattr(spot, "name", spot)
    records = [dict(r) for r in feature_rows(conn, spot_name)]
    usable = [
        r for r in records
        if r.get("real_speed") is not None and r.get("forecast_speed") is not None
    ]
    usable.sort(key=lambda r: (str(r.get("obs_date") or ""), int(r.get("hour") or 0)))
    if len(usable) < MIN_PAIRS:
        return None

    days = sorted({str(r.get("obs_date")) for r in usable if r.get("obs_date")})
    if len(days) < MIN_DISTINCT_DAYS:
        return None
    holdout_day_count = max(1, math.ceil(len(days) * 0.2))
    holdout_start = days[-holdout_day_count]
    train_rows = [r for r in usable if str(r.get("obs_date")) < holdout_start]
    holdout_rows = [r for r in usable if str(r.get("obs_date")) >= holdout_start]
    if len(train_rows) < 40 or len(holdout_rows) < 10:
        return None

    X_train = [_row_to_features(r) for r in train_rows]
    y_train = [float(r["real_speed"]) for r in train_rows]
    X_holdout = [_row_to_features(r) for r in holdout_rows]
    y_holdout = [float(r["real_speed"]) for r in holdout_rows]
    raw_holdout = [float(r["forecast_speed"]) for r in holdout_rows]
    gate_model = _train_regressor(lgb, X_train, y_train)
    oos_pred = [float(v) for v in gate_model.predict(X_holdout)]
    mae_oos = _mae(y_holdout, oos_pred)
    benchmark = _mae(y_holdout, raw_holdout)
    if not math.isfinite(benchmark) or mae_oos >= benchmark * (1.0 - MIN_RELATIVE_IMPROVEMENT):
        return None

    X = [_row_to_features(r) for r in usable]
    y = [float(r["real_speed"]) for r in usable]
    median_model = _train_regressor(lgb, X, y)
    lo_model = _train_regressor(lgb, X, y, objective="quantile", alpha=0.1)
    hi_model = _train_regressor(lgb, X, y, objective="quantile", alpha=0.9)
    model = GBModel(
        median_model=median_model,
        lo_model=lo_model,
        hi_model=hi_model,
        mae_oos=float(mae_oos),
        benchmark_mae=float(benchmark),
        n_samples=len(usable),
        training_start=days[0],
        training_end=days[-1],
        holdout_start=holdout_start,
    )
    path = gb_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)
    metadata = {
        "training_start": model.training_start,
        "training_end": model.training_end,
        "holdout_start": model.holdout_start,
        "n_samples": model.n_samples,
        "mae_oos": model.mae_oos,
        "raw_holdout_mae": model.benchmark_mae,
        "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
        "features": list(FEATURE_COLUMNS),
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8",
    )
    return model


def predict_day_gb(
    conn, spot, target: date, threshold: float = 13.0, hours=None, regional=None,
) -> dict[int, Forecast] | None:
    """Previsione giorno-dopo via GB, oppure None se il gating non passa."""
    from .providers.open_meteo import OpenMeteoProvider
    from .providers.regional import fetch_regional_features

    model = train_gb(conn, spot)
    if model is None:
        return None

    if hours is None:
        hours = OpenMeteoProvider().fetch(spot, target).hours
    if regional is None:
        regional = fetch_regional_features(spot, target)
    out: dict[int, Forecast] = {}
    for h in hours:
        hr = h.datetime.hour
        if not (USEFUL_START <= hr <= USEFUL_END):
            continue
        reg = regional.get(hr, {})
        median, lo, hi = model.predict({
            "target_date": target.isoformat(),
            "hour": hr,
            "forecast_speed": h.wind_speed_knots,
            "f_radiation": h.radiation,
            "f_wind_speed": h.wind_speed_knots,
            "f_wind_dir_deg": h.wind_direction_degrees,
            "dT_land_sea": reg.get("dT_land_sea"),
            "dT_land_sst": reg.get("dT_land_sst"),
            "boundary_layer_height_m": h.boundary_layer_height_m,
            "cross_isthmus": reg.get("cross_isthmus"),
            "synoptic_kn": reg.get("synoptic_kn"),
        })
        if hi <= lo:
            p = 1.0 if median >= threshold else 0.0
        elif threshold <= lo:
            p = 1.0
        elif threshold >= hi:
            p = 0.0
        else:
            p = (hi - threshold) / (hi - lo)
        out[hr] = Forecast(median, lo, hi, model.n_samples, round(max(0.0, min(1.0, p)), 2), "lgbm")
    return out
