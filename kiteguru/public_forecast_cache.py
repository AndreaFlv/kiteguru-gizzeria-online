"""Persistent operational forecast fallback for the public dashboard.

These files are deliberately separate from ``data/snapshots``: they may be
refreshed and must never enter the leakage-safe day-ahead evaluation.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from kiteguru.models import ProviderResult, SpotConfig
from kiteguru.providers.open_meteo import OpenMeteoProvider


def cache_path(root: Path, target: date) -> Path:
    return root / "data" / "public_forecasts" / f"{target.isoformat()}.json"


def write_public_forecast(root: Path, spot: SpotConfig, target: date) -> Path:
    """Fetch and atomically persist one operational forecast."""
    result = OpenMeteoProvider().fetch(spot, target)
    if not result.is_real or not result.hours:
        raise RuntimeError(result.error or f"forecast vuoto per {target}")
    path = cache_path(root, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "public_dashboard_operational_fallback",
        "target_date": target.isoformat(),
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "forecast": result.model_dump(mode="json"),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_public_forecast(root: Path, target: date) -> tuple[dict, datetime] | None:
    """Load a valid cache artifact for exactly ``target``."""
    path = cache_path(root, target)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("target_date") != target.isoformat():
            return None
        forecast = ProviderResult.model_validate(payload.get("forecast"))
        if not forecast.is_real or not forecast.hours:
            return None
        generated_at = datetime.fromisoformat(payload["generated_at_utc"])
        return forecast.model_dump(mode="python"), generated_at
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
