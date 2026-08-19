from __future__ import annotations

import json

from scripts.check_streamlit_availability import health_url, write_report


def test_health_url_targets_the_deployed_app_backend() -> None:
    assert health_url("https://example.streamlit.app/") == (
        "https://example.streamlit.app/~/+/_stcore/health"
    )


def test_write_report_is_persistent_and_parseable(tmp_path) -> None:
    output = tmp_path / "evidence" / "availability.json"
    write_report(output, {"schema_version": 1, "status": "READY"})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "READY",
    }
