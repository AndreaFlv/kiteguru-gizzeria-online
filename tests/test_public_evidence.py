import json

from kiteguru.public_evidence import load_verified_day_comparison


def test_public_comparison_uses_only_matching_snapshot_and_actual(tmp_path) -> None:
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "actual").mkdir()
    (tmp_path / "snapshots" / "2026-08-08.json").write_text(json.dumps({
        "target_date": "2026-08-08",
        "hours": [{
            "hour": 10,
            "raw_speed_kn": 8.0,
            "raw_gust_kn": 14.0,
            "scenario_speed_kn": 9.5,
        }],
    }), encoding="utf-8")
    (tmp_path / "actual" / "2026-08-08.json").write_text(json.dumps({
        "date": "2026-08-08",
        "hours": [{"hour": 10, "wind_speed_kn": 9.0, "wind_gust_kn": 16.0}],
    }), encoding="utf-8")

    result = load_verified_day_comparison(tmp_path, "2026-08-08")

    assert result == {"rows": [{
        "Ora": "10:00",
        "Base prevista": 8.0,
        "Raffica prevista": 14.0,
        "Atteso corretto": 9.5,
        "Reale base": 9.0,
        "Reale raffica": 16.0,
        "Scostamento base": 1.0,
        "Scostamento raffica": 2.0,
    }]}
