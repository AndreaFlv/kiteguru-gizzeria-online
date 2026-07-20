from __future__ import annotations

import os

import requests
from dateutil.parser import isoparse

from kiteguru.models import RealObservation, SpotConfig
from kiteguru.utils import degrees_to_cardinal


class HolfuyProvider:
    """Dato reale misurato da una stazione fisica Holfuy.

    Usa solo l'API ufficiale (nessuno scraping). La stazione di Hang Loose Beach
    (id 1178) e' privata: richiede una API key rilasciata dal gestore Holfuy.
    La key si passa via parametro o variabile d'ambiente HOLFUY_API_KEY.

    Senza key, o se la stazione nega l'accesso, fetch_current restituisce None:
    il chiamante degrada in modo pulito (nessun dato reale, niente crash).
    """

    source = "Holfuy (stazione reale)"
    endpoint = "https://api.holfuy.com/live/"

    def __init__(self, api_key: str | None = None, timeout: float = 10.0) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("HOLFUY_API_KEY")
        self.timeout = timeout

    def fetch_current(self, spot: SpotConfig) -> RealObservation | None:
        if not spot.holfuy_station:
            return None
        params = {
            "s": spot.holfuy_station,
            "m": "JSON",
            "su": "knots",
            "tu": "C",
        }
        if self.api_key:
            params["pw"] = self.api_key
        try:
            response = requests.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error") or payload.get("errorCode"):
                return None
            return self._parse(payload)
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return None

    def _parse(self, payload: dict) -> RealObservation | None:
        wind = payload.get("wind")
        if isinstance(wind, dict):
            speed = wind.get("speed")
            gust = wind.get("gust", wind.get("max"))
            direction = wind.get("direction")
        else:
            speed = payload.get("speed")
            gust = payload.get("gust")
            direction = payload.get("direction")
        if speed is None or direction is None:
            return None
        if gust is None:
            gust = speed
        when = payload.get("dateTime") or payload.get("date")
        dt = isoparse(when) if when else None
        if dt is None:
            return None
        direction = float(direction)
        return RealObservation(
            datetime=dt,
            wind_speed_knots=float(speed),
            wind_gusts_knots=float(gust),
            wind_direction_degrees=direction,
            wind_direction_cardinal=degrees_to_cardinal(direction),
            source=self.source,
        )
