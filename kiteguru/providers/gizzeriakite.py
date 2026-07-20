"""Read-only adapter for the GizzeriaKite Meteotemplate station.

The public WordPress page embeds a separate Meteotemplate installation.  This
adapter discovers the iframe (or accepts a direct live URL) and reads only the
public ``meteotemplateLive.txt`` snapshot.  It never posts data to the station.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse

import requests

KMH_TO_KNOTS = 0.539956803
DEFAULT_PAGE = "https://www.gizzeriakite.it/meteo/"


@dataclass(frozen=True)
class GizzeriaKiteReading:
    observed_at: datetime
    wind_speed_knots: float
    wind_gust_knots: float | None
    wind_direction_degrees: float | None
    source_url: str
    software: str | None = None


class _IframeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "iframe":
            return
        values = dict(attrs)
        if values.get("src"):
            self.sources.append(values["src"])


def parse_meteotemplate_live(payload: str, source_url: str) -> GizzeriaKiteReading:
    """Parse Meteotemplate's public live JSON (W/G are always km/h)."""
    data = json.loads(payload.lstrip("\ufeff").strip())
    if not isinstance(data, dict) or data.get("W") is None:
        raise ValueError("Meteotemplate live file without wind field W")
    stamp = data.get("UTime") or data.get("U") or data.get("WTime")
    if stamp is None:
        raise ValueError("Meteotemplate live file without update timestamp")
    observed = datetime.fromtimestamp(float(stamp), tz=timezone.utc)
    gust = data.get("G")
    bearing = data.get("B")
    return GizzeriaKiteReading(
        observed_at=observed,
        wind_speed_knots=round(float(data["W"]) * KMH_TO_KNOTS, 3),
        wind_gust_knots=round(float(gust) * KMH_TO_KNOTS, 3) if gust is not None else None,
        wind_direction_degrees=float(bearing) if bearing is not None else None,
        source_url=source_url,
        software=str(data["SW"]) if data.get("SW") is not None else None,
    )


def _candidate_live_urls(page_url: str, html: str) -> list[str]:
    parser = _IframeParser()
    parser.feed(html)
    candidates: list[str] = []
    for raw_src in parser.sources:
        src = urljoin(page_url, raw_src)
        parsed = urlparse(src)
        directory = parsed.path.rsplit("/", 1)[0] + "/"
        parts = [part for part in directory.split("/") if part]
        # MeteotemplateLive is in the template root.  Try the iframe directory
        # and a few parents, never unrelated hosts or broad URL scans.
        paths = ["/" + "/".join(parts[:i]) + "/" for i in range(len(parts), max(-1, len(parts) - 4), -1)]
        for path in paths:
            candidate = urlunparse((parsed.scheme, parsed.netloc, path + "meteotemplateLive.txt", "", "", ""))
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


class GizzeriaKiteProvider:
    def __init__(self, page_url: str = DEFAULT_PAGE, timeout: float = 12.0) -> None:
        self.page_url = page_url
        self.timeout = timeout

    def fetch(self) -> GizzeriaKiteReading:
        direct = os.getenv("KITEGURU_GIZZERIAKITE_LIVE_URL")
        if direct:
            candidates = [direct]
        else:
            response = requests.get(self.page_url, timeout=self.timeout)
            response.raise_for_status()
            candidates = _candidate_live_urls(self.page_url, response.text)
        if not candidates:
            raise RuntimeError("No Meteotemplate iframe/live endpoint discovered")

        errors: list[str] = []
        for url in candidates:
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                return parse_meteotemplate_live(response.text, url)
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{url}: {exc}")
        raise RuntimeError("GizzeriaKite live endpoint unavailable; " + " | ".join(errors))
