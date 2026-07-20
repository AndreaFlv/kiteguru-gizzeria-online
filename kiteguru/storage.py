from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from .models import ForecastHour, RealObservation

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at_utc TEXT NOT NULL,
    spot TEXT NOT NULL,
    obs_date TEXT NOT NULL,
    obs_hour INTEGER NOT NULL,
    forecast_speed REAL,
    forecast_gust REAL,
    forecast_dir_deg REAL,
    forecast_dir_card TEXT,
    real_speed REAL,
    real_gust REAL,
    real_dir_deg REAL,
    real_dir_card TEXT,
    real_source TEXT,
    UNIQUE (spot, obs_date, obs_hour)
);

CREATE TABLE IF NOT EXISTS model_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at_utc TEXT NOT NULL,
    spot TEXT NOT NULL,
    obs_date TEXT NOT NULL,
    obs_hour INTEGER NOT NULL,
    model TEXT NOT NULL,
    forecast_speed REAL,
    origin TEXT NOT NULL DEFAULT 'legacy_same_day',
    UNIQUE (spot, obs_date, obs_hour, model)
);

CREATE TABLE IF NOT EXISTS forecast_snapshots (
    spot TEXT NOT NULL,
    target_date TEXT NOT NULL,
    obs_hour INTEGER NOT NULL,
    made_at_utc TEXT NOT NULL,
    forecast_speed REAL NOT NULL,
    forecast_gust REAL,
    forecast_dir_deg REAL,
    forecast_dir_card TEXT,
    f_temp REAL,
    f_cloud REAL,
    f_radiation REAL,
    f_pressure REAL,
    boundary_layer_height_m REAL,
    dT_land_sea REAL,
    sea_surface_temp_c REAL,
    dT_land_sst REAL,
    dP_ionio_mare REAL,
    cross_isthmus REAL,
    synoptic_kn REAL,
    UNIQUE (spot, target_date, obs_hour)
);

CREATE TABLE IF NOT EXISTS real_raw (
    spot TEXT NOT NULL,
    ts TEXT NOT NULL,
    wind_speed REAL,
    wind_gust REAL,
    wind_dir_deg REAL,
    temp_c REAL,
    UNIQUE (spot, ts)
);

CREATE TABLE IF NOT EXISTS regional_features (
    spot TEXT NOT NULL,
    obs_date TEXT NOT NULL,
    obs_hour INTEGER NOT NULL,
    dT_land_sea REAL,
    dP_ionio_mare REAL,
    cross_isthmus REAL,
    synoptic_kn REAL,
    sea_surface_temp_c REAL,
    dT_land_sst REAL,
    UNIQUE (spot, obs_date, obs_hour)
);

CREATE TABLE IF NOT EXISTS predictions (
    spot TEXT NOT NULL,
    target_date TEXT NOT NULL,
    obs_hour INTEGER NOT NULL,
    made_at_utc TEXT NOT NULL,
    method TEXT,
    pred_median REAL,
    pred_lo REAL,
    pred_hi REAL,
    p_kiteable REAL,
    UNIQUE (spot, target_date, obs_hour)
);

CREATE TABLE IF NOT EXISTS regime_log (
    spot TEXT NOT NULL,
    obs_date TEXT NOT NULL,
    obs_hour INTEGER NOT NULL,
    regime TEXT NOT NULL,
    delta_actual REAL,
    logged_at_utc TEXT NOT NULL,
    UNIQUE (spot, obs_date, obs_hour)
);

CREATE TABLE IF NOT EXISTS alternative_real_raw (
    source TEXT NOT NULL,
    spot TEXT NOT NULL,
    ts TEXT NOT NULL,
    wind_speed REAL,
    wind_gust REAL,
    wind_dir_deg REAL,
    source_url TEXT,
    software TEXT,
    logged_at_utc TEXT NOT NULL,
    UNIQUE (source, spot, ts)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT NOT NULL,
    spot TEXT NOT NULL,
    status TEXT NOT NULL,
    observations_logged INTEGER NOT NULL DEFAULT 0,
    predictions_logged INTEGER NOT NULL DEFAULT 0,
    details TEXT
);
"""

# Colonne-feature aggiunte in seguito (alimentano il modello termico). La
# migrazione le aggiunge se mancano, cosi' i DB esistenti restano validi.
FEATURE_COLUMNS = {
    "f_temp": "REAL",
    "f_cloud": "REAL",
    "f_radiation": "REAL",
    "f_pressure": "REAL",
    "f_wind_speed": "REAL",
    "f_wind_dir_deg": "REAL",
    "real_temp": "REAL",
}


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(observations)")}
    for col, col_type in FEATURE_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE observations ADD COLUMN {col} {col_type}")
    model_existing = {row[1] for row in conn.execute("PRAGMA table_info(model_forecasts)")}
    if "origin" not in model_existing:
        conn.execute(
            "ALTER TABLE model_forecasts ADD COLUMN origin TEXT "
            "NOT NULL DEFAULT 'legacy_same_day'"
        )
    snapshot_columns = {
        "boundary_layer_height_m": "REAL", "sea_surface_temp_c": "REAL",
        "dT_land_sst": "REAL",
    }
    snapshot_existing = {
        row[1] for row in conn.execute("PRAGMA table_info(forecast_snapshots)")
    }
    for col, col_type in snapshot_columns.items():
        if col not in snapshot_existing:
            conn.execute(f"ALTER TABLE forecast_snapshots ADD COLUMN {col} {col_type}")
    regional_columns = {"sea_surface_temp_c": "REAL", "dT_land_sst": "REAL"}
    regional_existing = {
        row[1] for row in conn.execute("PRAGMA table_info(regional_features)")
    }
    for col, col_type in regional_columns.items():
        if col not in regional_existing:
            conn.execute(f"ALTER TABLE regional_features ADD COLUMN {col} {col_type}")


def get_db_path() -> Path:
    override = os.getenv("KITEGURU_DB")
    if override:
        return Path(override)
    return Path.home() / ".kiteguru" / "kiteguru.db"


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    db_path = path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect_readonly(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the KiteGuru database without creating, migrating or committing.

    Use this for dashboards, audits and reports: those consumers must not touch
    delicate stored observations while merely reading them.
    """
    db_path = path or get_db_path()
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    uri = db_path.resolve().as_posix()
    conn = sqlite3.connect(f"file:{uri}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def log_pair(
    conn: sqlite3.Connection,
    *,
    spot: str,
    obs_date: str,
    obs_hour: int,
    forecast: ForecastHour | None,
    real: RealObservation | None,
) -> None:
    """Inserisce/aggiorna la coppia previsione-misura per una data+ora.

    L'UNIQUE su (spot, obs_date, obs_hour) garantisce una riga per ora: se la
    misura reale arriva in un secondo momento (es. key Holfuy aggiunta dopo)
    aggiorna la riga senza duplicarla, conservando il valore non nullo.
    """
    conn.execute(
        """
        INSERT INTO observations (
            logged_at_utc, spot, obs_date, obs_hour,
            forecast_speed, forecast_gust, forecast_dir_deg, forecast_dir_card,
            real_speed, real_gust, real_dir_deg, real_dir_card, real_source,
            f_temp, f_cloud, f_radiation, f_pressure, f_wind_speed, f_wind_dir_deg, real_temp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (spot, obs_date, obs_hour) DO UPDATE SET
            logged_at_utc = excluded.logged_at_utc,
            forecast_speed = COALESCE(excluded.forecast_speed, observations.forecast_speed),
            forecast_gust = COALESCE(excluded.forecast_gust, observations.forecast_gust),
            forecast_dir_deg = COALESCE(excluded.forecast_dir_deg, observations.forecast_dir_deg),
            forecast_dir_card = COALESCE(excluded.forecast_dir_card, observations.forecast_dir_card),
            real_speed = COALESCE(excluded.real_speed, observations.real_speed),
            real_gust = COALESCE(excluded.real_gust, observations.real_gust),
            real_dir_deg = COALESCE(excluded.real_dir_deg, observations.real_dir_deg),
            real_dir_card = COALESCE(excluded.real_dir_card, observations.real_dir_card),
            real_source = COALESCE(excluded.real_source, observations.real_source),
            f_temp = COALESCE(excluded.f_temp, observations.f_temp),
            f_cloud = COALESCE(excluded.f_cloud, observations.f_cloud),
            f_radiation = COALESCE(excluded.f_radiation, observations.f_radiation),
            f_pressure = COALESCE(excluded.f_pressure, observations.f_pressure),
            f_wind_speed = COALESCE(excluded.f_wind_speed, observations.f_wind_speed),
            f_wind_dir_deg = COALESCE(excluded.f_wind_dir_deg, observations.f_wind_dir_deg),
            real_temp = COALESCE(excluded.real_temp, observations.real_temp)
        """,
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            spot,
            obs_date,
            obs_hour,
            forecast.wind_speed_knots if forecast else None,
            forecast.wind_gusts_knots if forecast else None,
            forecast.wind_direction_degrees if forecast else None,
            forecast.wind_direction_cardinal if forecast else None,
            real.wind_speed_knots if real else None,
            real.wind_gusts_knots if real else None,
            real.wind_direction_degrees if real else None,
            real.wind_direction_cardinal if real else None,
            real.source if real else None,
            forecast.temp_c if forecast else None,
            forecast.cloud_pct if forecast else None,
            forecast.radiation if forecast else None,
            forecast.pressure_hpa if forecast else None,
            forecast.wind_speed_knots if forecast else None,
            forecast.wind_direction_degrees if forecast else None,
            real.temp_c if real else None,
        ),
    )


def paired_rows(conn: sqlite3.Connection, spot: str) -> list[sqlite3.Row]:
    """Day-ahead snapshots paired with later real data: leakage-safe training."""
    cursor = conn.execute(
        """
        SELECT o.obs_date, o.obs_hour,
               s.forecast_speed, s.forecast_dir_card,
               o.real_speed, o.real_dir_card,
               s.f_radiation, s.forecast_speed AS f_wind_speed,
               s.forecast_dir_deg AS f_wind_dir_deg, s.f_cloud, s.f_temp
        FROM observations o
        JOIN forecast_snapshots s
          ON s.spot = o.spot AND s.target_date = o.obs_date AND s.obs_hour = o.obs_hour
        WHERE o.spot = ? AND s.forecast_speed IS NOT NULL AND o.real_speed IS NOT NULL
        """,
        (spot,),
    )
    return cursor.fetchall()


def log_raw_points(conn: sqlite3.Connection, *, spot: str, observations) -> int:
    """Salva i punti grezzi (~2-15 min) della stazione. Upsert per timestamp:
    rieseguire reimporta l'intera finestra Holfuy e ricuce eventuali buchi."""
    n = 0
    for obs in observations:
        conn.execute(
            """
            INSERT INTO real_raw (spot, ts, wind_speed, wind_gust, wind_dir_deg, temp_c)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (spot, ts) DO UPDATE SET
                wind_speed = excluded.wind_speed, wind_gust = excluded.wind_gust,
                wind_dir_deg = excluded.wind_dir_deg, temp_c = excluded.temp_c
            """,
            (spot, obs.datetime.isoformat(), obs.wind_speed_knots, obs.wind_gusts_knots,
             obs.wind_direction_degrees, obs.temp_c),
        )
        n += 1
    return n


def log_alternative_reading(conn: sqlite3.Connection, *, spot: str, reading) -> None:
    """Persist one reading from an independent station without merging sources."""
    conn.execute(
        """
        INSERT INTO alternative_real_raw
            (source, spot, ts, wind_speed, wind_gust, wind_dir_deg,
             source_url, software, logged_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (source, spot, ts) DO UPDATE SET
            wind_speed = excluded.wind_speed,
            wind_gust = excluded.wind_gust,
            wind_dir_deg = excluded.wind_dir_deg,
            source_url = excluded.source_url,
            software = excluded.software,
            logged_at_utc = excluded.logged_at_utc
        """,
        (
            "gizzeriakite_meteotemplate",
            spot,
            reading.observed_at.isoformat(),
            reading.wind_speed_knots,
            reading.wind_gust_knots,
            reading.wind_direction_degrees,
            reading.source_url,
            reading.software,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def log_pipeline_run(
    conn: sqlite3.Connection, *, started_at_utc: str, spot: str, status: str,
    observations_logged: int, predictions_logged: int, details: str | None,
) -> None:
    """Append an inspectable run result; never rewrites prior run history."""
    conn.execute(
        """
        INSERT INTO pipeline_runs
            (started_at_utc, finished_at_utc, spot, status,
             observations_logged, predictions_logged, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            started_at_utc,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            spot,
            status,
            observations_logged,
            predictions_logged,
            details,
        ),
    )


def alternative_source_comparison(
    conn: sqlite3.Connection, spot: str, threshold: float = 13.0,
    max_minutes: int = 20,
) -> sqlite3.Row:
    """Compare each alternative reading with the nearest Holfuy raw sample."""
    return conn.execute(
        """
        WITH candidates AS (
            SELECT a.ts, a.wind_speed AS alt_speed, r.wind_speed AS primary_speed,
                   ABS((julianday(r.ts) - julianday(a.ts)) * 1440.0) AS minutes_apart,
                   ROW_NUMBER() OVER (
                       PARTITION BY a.source, a.spot, a.ts
                       ORDER BY ABS(julianday(r.ts) - julianday(a.ts))
                   ) AS rn
            FROM alternative_real_raw a
            JOIN real_raw r ON r.spot = a.spot
             AND ABS((julianday(r.ts) - julianday(a.ts)) * 1440.0) <= ?
            WHERE a.spot = ? AND a.wind_speed IS NOT NULL AND r.wind_speed IS NOT NULL
        ), paired AS (
            SELECT alt_speed, primary_speed
            FROM candidates WHERE rn = 1
        )
        SELECT COUNT(*) AS n,
               AVG(alt_speed - primary_speed) AS bias,
               AVG(ABS(alt_speed - primary_speed)) AS mae_between_sources,
               AVG((alt_speed >= ?) = (primary_speed >= ?)) AS threshold_agreement,
               CASE WHEN COUNT(*) > 1 THEN
                   (AVG(alt_speed * primary_speed) - AVG(alt_speed) * AVG(primary_speed)) /
                   NULLIF(
                       SQRT(
                           (AVG(alt_speed * alt_speed) - AVG(alt_speed) * AVG(alt_speed)) *
                           (AVG(primary_speed * primary_speed) - AVG(primary_speed) * AVG(primary_speed))
                       ), 0
                   )
               END AS correlation
        FROM paired
        """,
        (max_minutes, spot, threshold, threshold),
    ).fetchone()


def log_regional_features(
    conn: sqlite3.Connection, *, spot: str, obs_date: str, obs_hour: int, feats: dict,
) -> None:
    """Salva le feature di contesto regionale per una data+ora (upsert)."""
    conn.execute(
        """
        INSERT INTO regional_features
            (spot, obs_date, obs_hour, dT_land_sea, dP_ionio_mare, cross_isthmus,
             synoptic_kn, sea_surface_temp_c, dT_land_sst)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (spot, obs_date, obs_hour) DO UPDATE SET
            dT_land_sea = COALESCE(excluded.dT_land_sea, regional_features.dT_land_sea),
            dP_ionio_mare = COALESCE(excluded.dP_ionio_mare, regional_features.dP_ionio_mare),
            cross_isthmus = COALESCE(excluded.cross_isthmus, regional_features.cross_isthmus),
            synoptic_kn = COALESCE(excluded.synoptic_kn, regional_features.synoptic_kn),
            sea_surface_temp_c = COALESCE(excluded.sea_surface_temp_c, regional_features.sea_surface_temp_c),
            dT_land_sst = COALESCE(excluded.dT_land_sst, regional_features.dT_land_sst)
        """,
        (spot, obs_date, obs_hour, feats.get("dT_land_sea"), feats.get("dP_ionio_mare"),
         feats.get("cross_isthmus"), feats.get("synoptic_kn"),
         feats.get("sea_surface_temp_c"), feats.get("dT_land_sst")),
    )


def log_model_forecast(
    conn: sqlite3.Connection, *, spot: str, obs_date: str, obs_hour: int, model: str,
    speed: float, origin: str = "day_ahead",
) -> None:
    """Congela la prima previsione day-ahead di un modello per data+ora."""
    conn.execute(
        """
        INSERT INTO model_forecasts
            (logged_at_utc, spot, obs_date, obs_hour, model, forecast_speed, origin)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (spot, obs_date, obs_hour, model) DO NOTHING
        """,
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), spot, obs_date,
         obs_hour, model, speed, origin),
    )


def log_forecast_snapshot(
    conn: sqlite3.Connection, *, spot: str, target_date: str, forecast: ForecastHour,
    regional: dict | None = None,
) -> None:
    """Freeze tomorrow's raw NWP and features before any observation exists."""
    regional = regional or {}
    conn.execute(
        """
        INSERT INTO forecast_snapshots (
            spot, target_date, obs_hour, made_at_utc,
            forecast_speed, forecast_gust, forecast_dir_deg, forecast_dir_card,
            f_temp, f_cloud, f_radiation, f_pressure, boundary_layer_height_m,
            dT_land_sea, sea_surface_temp_c, dT_land_sst,
            dP_ionio_mare, cross_isthmus, synoptic_kn
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (spot, target_date, obs_hour) DO NOTHING
        """,
        (
            spot, target_date, forecast.datetime.hour,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            forecast.wind_speed_knots, forecast.wind_gusts_knots,
            forecast.wind_direction_degrees, forecast.wind_direction_cardinal,
            forecast.temp_c, forecast.cloud_pct, forecast.radiation,
            forecast.pressure_hpa, forecast.boundary_layer_height_m,
            regional.get("dT_land_sea"), regional.get("sea_surface_temp_c"),
            regional.get("dT_land_sst"),
            regional.get("dP_ionio_mare"), regional.get("cross_isthmus"),
            regional.get("synoptic_kn"),
        ),
    )


def model_skill(conn: sqlite3.Connection, spot: str) -> list[sqlite3.Row]:
    """Classifica dei modelli per accuratezza vs stazione reale (MAE crescente).

    Confronta la previsione di ogni modello con `real_speed` della stessa ora.
    """
    cursor = conn.execute(
        """
        SELECT m.model AS model,
               COUNT(*) AS n,
               AVG(ABS(m.forecast_speed - o.real_speed)) AS mae,
               AVG(o.real_speed - m.forecast_speed) AS bias
        FROM model_forecasts m
        JOIN observations o
          ON o.spot = m.spot AND o.obs_date = m.obs_date AND o.obs_hour = m.obs_hour
        WHERE m.spot = ? AND m.origin = 'day_ahead'
              AND o.real_speed IS NOT NULL AND m.forecast_speed IS NOT NULL
        GROUP BY m.model
        ORDER BY mae ASC
        """,
        (spot,),
    )
    return cursor.fetchall()


def recommended_model(conn: sqlite3.Connection, spot: str, min_samples: int = 10) -> str | None:
    """Modello piu' affidabile se ha abbastanza campioni, altrimenti None."""
    rows = model_skill(conn, spot)
    if rows and rows[0]["n"] >= min_samples:
        return rows[0]["model"]
    return None


def feature_rows(conn: sqlite3.Connection, spot: str) -> list[sqlite3.Row]:
    """Righe con feature di forecast + regionali + misura reale: base per gli analoghi."""
    cursor = conn.execute(
        """
        SELECT o.obs_date AS obs_date, o.obs_hour AS hour,
               o.real_speed AS real_speed, s.forecast_speed AS forecast_speed,
               s.f_radiation AS f_radiation, s.forecast_speed AS f_wind_speed,
               s.forecast_dir_deg AS f_wind_dir_deg,
               s.dT_land_sea AS dT_land_sea, s.cross_isthmus AS cross_isthmus,
               s.synoptic_kn AS synoptic_kn,
               s.boundary_layer_height_m AS boundary_layer_height_m,
               s.sea_surface_temp_c AS sea_surface_temp_c,
               s.dT_land_sst AS dT_land_sst
        FROM observations o
        JOIN forecast_snapshots s
          ON s.spot = o.spot AND s.target_date = o.obs_date AND s.obs_hour = o.obs_hour
        WHERE o.spot = ? AND o.real_speed IS NOT NULL AND s.forecast_speed IS NOT NULL
        """,
        (spot,),
    )
    return cursor.fetchall()


def frozen_predictions(conn: sqlite3.Connection, spot: str, target_date: str) -> list[sqlite3.Row]:
    """Previsioni congelate per una data, ordinate per ora."""
    return conn.execute(
        """
        SELECT target_date, obs_hour, method, pred_median, pred_lo, pred_hi, p_kiteable
        FROM predictions
        WHERE spot = ? AND target_date = ?
        ORDER BY obs_hour
        """,
        (spot, target_date),
    ).fetchall()


def log_prediction(
    conn: sqlite3.Connection, *, spot: str, target_date: str, obs_hour: int, method: str,
    median: float, lo: float, hi: float, p_kiteable: float,
) -> None:
    """Congela la previsione del giorno dopo (upsert) per la verifica successiva."""
    conn.execute(
        """
        INSERT INTO predictions
            (spot, target_date, obs_hour, made_at_utc, method, pred_median, pred_lo, pred_hi, p_kiteable)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (spot, target_date, obs_hour) DO NOTHING
        """,
        (spot, target_date, obs_hour, datetime.now(timezone.utc).isoformat(timespec="seconds"),
         method, median, lo, hi, p_kiteable),
    )


def prediction_skill(conn: sqlite3.Connection, spot: str) -> sqlite3.Row:
    """Skill della previsione del giorno dopo vs reale, e confronto col NWP grezzo."""
    return conn.execute(
        """
        SELECT COUNT(*) AS n,
               AVG(ABS(p.pred_median - o.real_speed)) AS mae_pred,
               AVG(ABS(s.forecast_speed - o.real_speed)) AS mae_raw,
               AVG(CASE WHEN o.real_speed BETWEEN p.pred_lo AND p.pred_hi THEN 1.0 ELSE 0.0 END) AS coverage
        FROM predictions p
        JOIN observations o
          ON o.spot = p.spot AND o.obs_date = p.target_date AND o.obs_hour = p.obs_hour
        JOIN forecast_snapshots s
          ON s.spot = p.spot AND s.target_date = p.target_date AND s.obs_hour = p.obs_hour
        WHERE p.spot = ? AND o.real_speed IS NOT NULL AND p.pred_median IS NOT NULL
              AND s.forecast_speed IS NOT NULL
        """,
        (spot,),
    ).fetchone()


def prediction_skill_by_method(conn: sqlite3.Connection, spot: str) -> list[sqlite3.Row]:
    """Skill della previsione congelata separata per metodo."""
    return conn.execute(
        """
        SELECT p.method AS method,
               COUNT(*) AS n,
               AVG(ABS(p.pred_median - o.real_speed)) AS mae,
               AVG(o.real_speed - p.pred_median) AS bias,
               AVG(CASE WHEN o.real_speed BETWEEN p.pred_lo AND p.pred_hi THEN 1.0 ELSE 0.0 END) AS coverage
        FROM predictions p
        JOIN observations o
          ON o.spot = p.spot AND o.obs_date = p.target_date AND o.obs_hour = p.obs_hour
        WHERE p.spot = ? AND o.real_speed IS NOT NULL AND p.pred_median IS NOT NULL
        GROUP BY p.method
        ORDER BY mae ASC
        """,
        (spot,),
    ).fetchall()


def rolling_prediction_mae(conn: sqlite3.Connection, spot: str, window_days: int = 14) -> list[sqlite3.Row]:
    """MAE rolling per metodo, con finestra temporale in giorni."""
    return conn.execute(
        """
        SELECT p1.method AS method,
               p1.target_date AS target_date,
               AVG(ABS(p2.pred_median - o2.real_speed)) AS mae
        FROM predictions p1
        JOIN predictions p2
          ON p2.spot = p1.spot
         AND p2.method = p1.method
         AND julianday(p2.target_date) BETWEEN julianday(p1.target_date) - ? AND julianday(p1.target_date)
        JOIN observations o2
          ON o2.spot = p2.spot AND o2.obs_date = p2.target_date AND o2.obs_hour = p2.obs_hour
        WHERE p1.spot = ? AND o2.real_speed IS NOT NULL AND p2.pred_median IS NOT NULL
        GROUP BY p1.method, p1.target_date
        ORDER BY p1.target_date, p1.method
        """,
        (window_days - 1, spot),
    ).fetchall()


def onset_history_rows(conn: sqlite3.Connection, spot: str, lookback_days: int = 60) -> list[dict]:
    """Righe recenti per calcolare lo storico dell'orario di ingresso termico."""
    since = (date.today() - timedelta(days=lookback_days)).isoformat()
    rows = conn.execute(
        """
        SELECT o.obs_date, o.obs_hour, o.real_speed, s.forecast_speed
        FROM observations o
        JOIN forecast_snapshots s
          ON s.spot = o.spot AND s.target_date = o.obs_date AND s.obs_hour = o.obs_hour
        WHERE o.spot = ? AND o.obs_date >= ? AND o.real_speed IS NOT NULL
        ORDER BY obs_date, obs_hour
        """,
        (spot, since),
    ).fetchall()
    return [dict(row) for row in rows]


def log_regime(
    conn: sqlite3.Connection,
    *,
    spot: str,
    obs_date: str,
    obs_hour: int,
    regime: str,
    delta_actual: float | None,
) -> None:
    conn.execute(
        """
        INSERT INTO regime_log (
            spot, obs_date, obs_hour, regime, delta_actual, logged_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (spot, obs_date, obs_hour) DO UPDATE SET
            regime = excluded.regime,
            delta_actual = excluded.delta_actual,
            logged_at_utc = excluded.logged_at_utc
        """,
        (
            spot,
            obs_date,
            obs_hour,
            regime,
            delta_actual,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def counts(conn: sqlite3.Connection, spot: str) -> tuple[int, int]:
    """(righe totali, righe con misura reale) per uno spot."""
    total = conn.execute(
        "SELECT COUNT(*) FROM observations WHERE spot = ?", (spot,)
    ).fetchone()[0]
    real = conn.execute(
        "SELECT COUNT(*) FROM observations WHERE spot = ? AND real_speed IS NOT NULL",
        (spot,),
    ).fetchone()[0]
    return int(total), int(real)
