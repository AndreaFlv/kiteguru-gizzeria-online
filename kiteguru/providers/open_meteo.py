from __future__ import annotations

from datetime import date
import time

import requests
from dateutil.parser import isoparse

from kiteguru.models import ForecastHour, ProviderResult, SpotConfig
from kiteguru.utils import degrees_to_cardinal


class OpenMeteoProvider:
    source = "Open-Meteo Forecast API"
    endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: float = 10.0, attempts: int = 3) -> None:
        self.timeout = timeout
        self.attempts = max(1, attempts)

    def fetch(self, spot: SpotConfig, target: date) -> ProviderResult:
        params = {
            "latitude": spot.latitude,
            "longitude": spot.longitude,
            "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                      "temperature_2m,cloud_cover,shortwave_radiation,surface_pressure,"
                      "boundary_layer_height,precipitation_probability,precipitation,"
                      "weather_code,cape",
            "wind_speed_unit": "kn",
            "timezone": spot.timezone,
            # Chiedere esplicitamente l'intera giornata e' essenziale nei run
            # serali: senza start/end Open-Meteo puo' omettere le ore passate.
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
        }
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                response = requests.get(self.endpoint, params=params, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(0.5 * (attempt + 1))
        else:
            return ProviderResult(source=self.source, is_real=False, hours=[], error=str(last_error))

        try:
            payload = response.json()
            hourly = payload.get("hourly")
            if not isinstance(hourly, dict):
                raise ValueError("Risposta API senza blocco hourly")
            times = hourly.get("time") or []
            speeds = hourly.get("wind_speed_10m") or []
            gusts = hourly.get("wind_gusts_10m") or []
            directions = hourly.get("wind_direction_10m") or []
            if not (len(times) == len(speeds) == len(gusts) == len(directions)):
                raise ValueError("Serie orarie Open-Meteo non allineate")
            temps = hourly.get("temperature_2m") or [None] * len(times)
            clouds = hourly.get("cloud_cover") or [None] * len(times)
            radiation = hourly.get("shortwave_radiation") or [None] * len(times)
            pressure = hourly.get("surface_pressure") or [None] * len(times)
            boundary_layer = hourly.get("boundary_layer_height") or [None] * len(times)
            precipitation_probability = (
                hourly.get("precipitation_probability") or [None] * len(times)
            )
            precipitation = hourly.get("precipitation") or [None] * len(times)
            weather_code = hourly.get("weather_code") or [None] * len(times)
            cape = hourly.get("cape") or [None] * len(times)

            def _opt(seq, i):
                value = seq[i] if i < len(seq) else None
                return float(value) if value is not None else None

            hours: list[ForecastHour] = []
            for i, (time_value, speed, gust, direction) in enumerate(zip(times, speeds, gusts, directions)):
                dt = isoparse(time_value)
                if dt.date() != target:
                    continue
                hours.append(
                    ForecastHour(
                        datetime=dt,
                        wind_speed_knots=float(speed),
                        wind_gusts_knots=float(gust),
                        wind_direction_degrees=float(direction),
                        wind_direction_cardinal=degrees_to_cardinal(float(direction)),
                        source=self.source,
                        temp_c=_opt(temps, i),
                        cloud_pct=_opt(clouds, i),
                        radiation=_opt(radiation, i),
                        pressure_hpa=_opt(pressure, i),
                        boundary_layer_height_m=_opt(boundary_layer, i),
                        precipitation_probability_pct=_opt(precipitation_probability, i),
                        precipitation_mm=_opt(precipitation, i),
                        weather_code=(
                            int(weather_code[i])
                            if i < len(weather_code) and weather_code[i] is not None
                            else None
                        ),
                        cape_jkg=_opt(cape, i),
                    )
                )
            if not hours:
                raise ValueError("Nessun dato orario per la data richiesta")
            return ProviderResult(source=self.source, is_real=True, hours=hours)
        except (ValueError, TypeError) as exc:
            return ProviderResult(source=self.source, is_real=False, hours=[], error=str(exc))
