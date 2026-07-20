"""Cloud collector: immutable day-ahead forecast, Holfuy truth and metrics JSON."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from kiteguru.config import get_spot
from kiteguru.correction import apply_correction
from kiteguru.evaluation import operational_summary
from kiteguru.providers.holfuy_chart import HolfuyChartProvider, aggregate_hourly
from kiteguru.providers.open_meteo import OpenMeteoProvider
from kiteguru.providers.regional import fetch_regional_features
from kiteguru.thermal_model import train as train_thermal_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_atomic_if_changed(
    path: Path, payload: dict, *, ignored_top_level_keys: tuple[str, ...] = (),
) -> bool:
    """Write only when semantic content changed; return whether a write occurred."""
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if isinstance(current, dict):
            current_semantic = {
                key: value for key, value in current.items()
                if key not in ignored_top_level_keys
            }
            new_semantic = {
                key: value for key, value in payload.items()
                if key not in ignored_top_level_keys
            }
            if current_semantic == new_semantic:
                return False
    _write_json_atomic(path, payload)
    return True


def _write_actual_if_improved(path: Path, payload: dict, *, local_today: date) -> bool:
    """Keep completed historical truth stable as the rolling source window moves."""
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        payload_day = date.fromisoformat(str(payload["date"]))
        if isinstance(current, dict) and payload_day < local_today:
            current_hours = {
                int(row["hour"]) for row in current.get("hours", [])
                if row.get("hour") is not None
            }
            new_hours = {
                int(row["hour"]) for row in payload.get("hours", [])
                if row.get("hour") is not None
            }
            if current.get("complete_useful_hours"):
                return False
            if len(new_hours) <= len(current_hours):
                return False
    return _write_json_atomic_if_changed(
        path, payload, ignored_top_level_keys=("collected_at_utc",),
    )


def freeze_forecast(target: date) -> tuple[Path, bool]:
    """Create the first snapshot for target; never overwrite an existing one."""
    path = DATA / "snapshots" / f"{target.isoformat()}.json"
    if path.exists():
        return path, False
    spot = get_spot("gizzeria")
    result = OpenMeteoProvider().fetch(spot, target)
    if not result.hours:
        raise RuntimeError(result.error or f"forecast vuoto per {target}")
    regional = fetch_regional_features(spot, target)
    prior = train_thermal_model(spot, [])
    corrected, _ = apply_correction(result.hours, prior, spot)
    corrected_by_hour = {hour.datetime.hour: hour for hour in corrected}
    now = datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds")
    hours = []
    for raw in result.hours:
        hour = raw.datetime.hour
        if not 10 <= hour <= 19:
            continue
        scenario = corrected_by_hour[hour]
        hours.append({
            "hour": hour,
            "raw_speed_kn": raw.wind_speed_knots,
            "scenario_speed_kn": scenario.wind_speed_knots,
            "raw_gust_kn": raw.wind_gusts_knots,
            "direction_deg": raw.wind_direction_degrees,
            "direction_cardinal": raw.wind_direction_cardinal,
            "radiation": raw.radiation,
            "boundary_layer_height_m": raw.boundary_layer_height_m,
            "precipitation_probability_pct": raw.precipitation_probability_pct,
            "precipitation_mm": raw.precipitation_mm,
            "weather_code": raw.weather_code,
            "cape_jkg": raw.cape_jkg,
            "regional": regional.get(hour, {}),
        })
    if len(hours) != 10:
        raise RuntimeError(f"snapshot incompleto: {len(hours)}/10 ore")
    _write_json_atomic(path, {
        "schema_version": 1,
        "spot": spot.name,
        "target_date": target.isoformat(),
        "made_at_utc": now,
        "source": result.source,
        "scenario_status": "uncalibrated_physical_prior",
        "hours": hours,
    })
    return path, True


def collect_actuals() -> list[Path]:
    """Materialize all dates currently exposed by the public Holfuy series."""
    spot = get_spot("gizzeria")
    series = HolfuyChartProvider().fetch_series(spot)
    if not series:
        raise RuntimeError("Holfuy non ha restituito letture")
    hourly = aggregate_hourly(series)
    by_date: dict[str, list[dict]] = {}
    for (day, hour), obs in sorted(hourly.items()):
        if 10 <= hour <= 19:
            by_date.setdefault(day, []).append({
                "hour": hour,
                "wind_speed_kn": obs.wind_speed_knots,
                "wind_gust_kn": obs.wind_gusts_knots,
                "direction_deg": obs.wind_direction_degrees,
                "direction_cardinal": obs.wind_direction_cardinal,
            })
    written = []
    collected_at = datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds")
    local_today = datetime.now(ZoneInfo(spot.timezone)).date()
    for day, hours in by_date.items():
        path = DATA / "actual" / f"{day}.json"
        changed = _write_actual_if_improved(path, {
            "schema_version": 1,
            "spot": spot.name,
            "date": day,
            "collected_at_utc": collected_at,
            "source": HolfuyChartProvider.source,
            "complete_useful_hours": len(hours) == 10,
            "hours": hours,
        }, local_today=local_today)
        if changed:
            written.append(path)
    return written


def evaluate() -> dict:
    raw_records, scenario_records = [], []
    paired_dates = []
    for snapshot_path in sorted((DATA / "snapshots").glob("*.json")):
        actual_path = DATA / "actual" / snapshot_path.name
        if not actual_path.exists():
            continue
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        actual_by_hour = {int(row["hour"]): row for row in actual.get("hours", [])}
        paired = 0
        for forecast in snapshot.get("hours", []):
            observed = actual_by_hour.get(int(forecast["hour"]))
            if observed is None:
                continue
            base = {
                "date": snapshot["target_date"],
                "hour": forecast["hour"],
                "actual": observed["wind_speed_kn"],
            }
            raw_records.append({**base, "predicted": forecast["raw_speed_kn"]})
            scenario_records.append({**base, "predicted": forecast["scenario_speed_kn"]})
            paired += 1
        if paired:
            paired_dates.append(snapshot["target_date"])
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "paired_dates": paired_dates,
        "raw_forecast": operational_summary(
            raw_records, expected_hours_per_day=10,
        ),
        "thermal_scenario_research_only": operational_summary(
            scenario_records, expected_hours_per_day=10,
        ),
    }
    _write_json_atomic_if_changed(
        DATA / "metrics" / "latest.json",
        payload,
        ignored_top_level_keys=("generated_at_utc",),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", type=date.fromisoformat)
    args = parser.parse_args()
    spot = get_spot("gizzeria")
    local_today = datetime.now(ZoneInfo(spot.timezone)).date()
    target = args.target_date or (local_today + timedelta(days=1))
    snapshot, created = freeze_forecast(target)
    actuals = collect_actuals()
    metrics = evaluate()
    print(json.dumps({
        "snapshot": str(snapshot.relative_to(ROOT)),
        "snapshot_created": created,
        "actual_files_changed": len(actuals),
        "paired_dates": len(metrics["paired_dates"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
