from __future__ import annotations

from datetime import date

import requests

from kiteguru.models import SpotConfig

# Modelli meteo confrontati (label leggibile -> id Open-Meteo). ICON-2I e' il
# modello italiano ad alta risoluzione (2.2 km) che copre anche il Sud Italia.
COMPARISON_MODELS = {
    "ICON-2I (IT 2km)": "italia_meteo_arpae_icon_2i",
    "ICON (EU 7km)": "icon_seamless",
    "GFS": "gfs_seamless",
    "ECMWF": "ecmwf_ifs025",
}

ENDPOINT = "https://api.open-meteo.com/v1/forecast"


def fetch_model_winds(
    spot: SpotConfig, target: date, timeout: float = 15.0
) -> dict[str, dict[int, float]]:
    """Vento orario (nodi) di ogni modello per il giorno `target`.

    Ritorna {label: {ora: velocita'}}. Mai solleva: in caso di errore ritorna
    dizionari vuoti per ogni modello.
    """
    out: dict[str, dict[int, float]] = {label: {} for label in COMPARISON_MODELS}
    target_iso = target.isoformat()
    try:
        response = requests.get(
            ENDPOINT,
            params={
                "latitude": spot.latitude,
                "longitude": spot.longitude,
                "hourly": "wind_speed_10m",
                "wind_speed_unit": "kn",
                "timezone": spot.timezone,
                "start_date": target_iso,
                "end_date": target_iso,
                "models": ",".join(COMPARISON_MODELS.values()),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
    except (requests.RequestException, ValueError):
        return out
    times = hourly.get("time", [])
    for label, model in COMPARISON_MODELS.items():
        series = hourly.get(f"wind_speed_10m_{model}", [])
        for time_value, value in zip(times, series):
            day, hh = time_value.split("T")
            if day == target_iso and value is not None:
                out[label][int(hh[:2])] = float(value)
    return out
