from __future__ import annotations

import ast
import math
import re
from datetime import datetime
from statistics import mean
from zoneinfo import ZoneInfo

import requests

from kiteguru.models import RealObservation, SpotConfig
from kiteguru.utils import degrees_to_cardinal

# Il grafico pubblico Holfuy memorizza il vento in km/h; noi lavoriamo in nodi.
KMH_TO_KN = 0.539957
_ARRAY = r"{name}\s*=\s*(\[[^\]]*\])"


def _array(text: str, name: str) -> list:
    match = re.search(_ARRAY.format(name=re.escape(name)), text)
    if not match:
        return []
    try:
        return list(ast.literal_eval(match.group(1)))
    except (ValueError, SyntaxError):
        return []


def parse_series(text: str, timezone: str) -> list[RealObservation]:
    """Parsa il file `tdarr<station>.js` in osservazioni reali (vento in nodi).

    Nessuna eccezione propagata: dati mancanti/non parsabili vengono saltati.
    """
    tz = ZoneInfo(timezone)
    times = _array(text, "unt")
    speeds = _array(text, "gd_speed")
    gusts = _array(text, "gd_gust")
    directions = _array(text, "gd_direction")
    temps = _array(text, "gd_temp")
    n = min(len(times), len(speeds), len(gusts), len(directions))
    out: list[RealObservation] = []
    for i in range(n):
        try:
            dt = datetime.strptime(str(times[i]), "%Y/%m/%d %H:%M:%S").replace(tzinfo=tz)
            speed = float(speeds[i]) * KMH_TO_KN
            gust = float(gusts[i]) * KMH_TO_KN
            direction = float(directions[i])
        except (ValueError, TypeError):
            continue
        try:
            temp = float(temps[i]) if i < len(temps) else None
        except (ValueError, TypeError):
            temp = None
        out.append(
            RealObservation(
                datetime=dt,
                wind_speed_knots=round(speed, 1),
                wind_gusts_knots=round(gust, 1),
                wind_direction_degrees=direction,
                wind_direction_cardinal=degrees_to_cardinal(direction),
                source=HolfuyChartProvider.source,
                temp_c=temp,
            )
        )
    return out


def aggregate_hourly(series: list[RealObservation]) -> dict[tuple[str, int], RealObservation]:
    """Media oraria delle osservazioni a 15 min, indicizzata per (data, ora)."""
    buckets: dict[tuple[str, int], list[RealObservation]] = {}
    for obs in series:
        buckets.setdefault((obs.datetime.date().isoformat(), obs.datetime.hour), []).append(obs)
    hourly: dict[tuple[str, int], RealObservation] = {}
    for key, group in buckets.items():
        sin_mean = mean(math.sin(math.radians(o.wind_direction_degrees)) for o in group)
        cos_mean = mean(math.cos(math.radians(o.wind_direction_degrees)) for o in group)
        avg_dir = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
        ref = group[-1]
        temps = [o.temp_c for o in group if o.temp_c is not None]
        hourly[key] = RealObservation(
            datetime=ref.datetime.replace(minute=0, second=0, microsecond=0),
            wind_speed_knots=round(mean(o.wind_speed_knots for o in group), 1),
            # A gust is an extreme within the hour, not an average of extremes.
            wind_gusts_knots=round(max(o.wind_gusts_knots for o in group), 1),
            wind_direction_degrees=avg_dir,
            wind_direction_cardinal=degrees_to_cardinal(avg_dir),
            source=ref.source,
            temp_c=round(mean(temps), 1) if temps else None,
        )
    return hourly


class HolfuyChartProvider:
    """Dato reale dal grafico pubblico Holfuy (nessuna API key richiesta).

    Usa l'endpoint che la pagina della stazione consuma per disegnare il grafico:
    e' una sorgente best-effort, piu' stabile dello scraping HTML ma non
    ufficialmente documentata: puo' cambiare senza preavviso.
    """

    source = "Holfuy (grafico pubblico)"
    endpoint = "https://holfuy.com/dynamic/graphs/tdarr{station}.js"

    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout

    def fetch_series(self, spot: SpotConfig) -> list[RealObservation]:
        if not spot.holfuy_station:
            return []
        url = self.endpoint.format(station=spot.holfuy_station)
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": f"https://holfuy.com/it/weather/{spot.holfuy_station}",
                },
            )
            response.raise_for_status()
        except requests.RequestException:
            return []
        return parse_series(response.text, spot.timezone)

    def fetch_current(self, spot: SpotConfig) -> RealObservation | None:
        series = self.fetch_series(spot)
        return series[-1] if series else None
