"""Letture pubbliche, verificabili e non ricostruite dei confronti day-ahead."""
from __future__ import annotations

import json
from pathlib import Path


def load_verified_day_comparison(data_root: Path, target_iso: str) -> dict | None:
    """Combina solo uno snapshot day-ahead con il reale Holfuy della stessa data."""
    snapshot_path = data_root / "snapshots" / f"{target_iso}.json"
    actual_path = data_root / "actual" / f"{target_iso}.json"
    if not snapshot_path.exists() or not actual_path.exists():
        return None
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if snapshot.get("target_date") != target_iso or actual.get("date") != target_iso:
        return None

    actual_by_hour = {
        int(row["hour"]): row for row in actual.get("hours", [])
        if row.get("hour") is not None
    }
    rows = []
    for forecast in snapshot.get("hours", []):
        hour = forecast.get("hour")
        observed = actual_by_hour.get(int(hour)) if hour is not None else None
        if observed is None:
            continue
        base = forecast.get("raw_speed_kn")
        gust = forecast.get("raw_gust_kn")
        real_base = observed.get("wind_speed_kn")
        real_gust = observed.get("wind_gust_kn")
        if None in (base, gust, real_base, real_gust):
            continue
        rows.append({
            "Ora": f"{int(hour):02d}:00",
            "Base prevista": round(float(base), 1),
            "Raffica prevista": round(float(gust), 1),
            "Atteso corretto": round(float(forecast.get("scenario_speed_kn", base)), 1),
            "Reale base": round(float(real_base), 1),
            "Reale raffica": round(float(real_gust), 1),
            "Scostamento base": round(float(real_base) - float(base), 1),
            "Scostamento raffica": round(float(real_gust) - float(gust), 1),
        })
    if not rows:
        return None
    return {"rows": sorted(rows, key=lambda row: row["Ora"])}
