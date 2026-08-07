from datetime import date
from unittest.mock import Mock, patch

from kiteguru.config import get_spot
from kiteguru.providers.open_meteo import OpenMeteoProvider


def test_rate_limit_is_reported_without_leaking_request_url() -> None:
    response = Mock(status_code=429, headers={"Retry-After": "0"})
    response.raise_for_status.side_effect = __import__("requests").HTTPError("429 request URL")

    with patch("kiteguru.providers.open_meteo.requests.get", return_value=response) as get:
        with patch("kiteguru.providers.open_meteo.time.sleep"):
            result = OpenMeteoProvider(attempts=2).fetch(
                get_spot("gizzeria"), date(2026, 8, 7)
            )

    assert not result.is_real
    assert result.hours == []
    assert result.error == "Open-Meteo temporaneamente limitato (HTTP 429)"
    assert get.call_count == 2
    assert get.call_args.kwargs["headers"]["User-Agent"].startswith("KiteGuru-Gizzeria")
