from datetime import date, datetime, timezone
import json

from kiteguru.public_forecast_cache import load_public_forecast


def test_load_public_forecast_validates_target_and_payload(tmp_path):
    target = date(2026, 8, 7)
    path = tmp_path / "data" / "public_forecasts" / "2026-08-07.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "purpose": "public_dashboard_operational_fallback",
        "target_date": target.isoformat(),
        "generated_at_utc": datetime(2026, 8, 7, 10, tzinfo=timezone.utc).isoformat(),
        "forecast": {
            "source": "Open-Meteo Forecast API",
            "is_real": True,
            "error": None,
            "hours": [{
                "datetime": "2026-08-07T10:00:00",
                "wind_speed_knots": 12.0,
                "wind_gusts_knots": 16.0,
                "wind_direction_degrees": 280.0,
                "wind_direction_cardinal": "W",
                "source": "Open-Meteo Forecast API",
            }],
        },
    }), encoding="utf-8")

    loaded = load_public_forecast(tmp_path, target)

    assert loaded is not None
    forecast, generated_at = loaded
    assert forecast["is_real"] is True
    assert len(forecast["hours"]) == 1
    assert generated_at == datetime(2026, 8, 7, 10, tzinfo=timezone.utc)


def test_load_public_forecast_rejects_wrong_target(tmp_path):
    target = date(2026, 8, 7)
    path = tmp_path / "data" / "public_forecasts" / "2026-08-07.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "target_date": "2026-08-08",
        "generated_at_utc": "2026-08-07T10:00:00+00:00",
        "forecast": {},
    }), encoding="utf-8")

    assert load_public_forecast(tmp_path, target) is None
