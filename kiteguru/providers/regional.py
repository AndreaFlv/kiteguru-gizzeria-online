"""Feature di contesto regionale per la brezza di Gizzeria.

Gizzeria sta sul punto piu' stretto della Calabria (istmo di Catanzaro): il vento
accelera attraverso l'istmo dallo Ionio al Tirreno (Venturi). Le condizioni a
Ionio/Tirreno/entroterra spiegano gran parte della brezza locale. Qui le
campioniamo in una sola chiamata multi-localita' Open-Meteo e ne ricaviamo:

- dT_land_sea   : T(entroterra) - T(mare)   -> forzante termica (terra calda vs mare)
- dP_ionio_mare : P(ionio) - P(mare)        -> gradiente di pressione sull'istmo
- cross_isthmus : componente del vento allo Ionio diretta VERSO Gizzeria
                  (positiva = rinforza la brezza trans-istmo; negativa = contraria)
- synoptic_kn   : intensita' del vento al largo (Tirreno) -> forzante sinottica
- sea_surface_temp_c: temperatura superficiale marina al largo
- dT_land_sst   : T(entroterra) - SST -> contrasto termico terra/mare piu' fisico
"""
from __future__ import annotations

import math

import requests

from kiteguru.models import SpotConfig

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
MARINE_ENDPOINT = "https://marine-api.open-meteo.com/v1/marine"
FEATURE_KEYS = (
    "dT_land_sea", "dP_ionio_mare", "cross_isthmus", "synoptic_kn",
    "sea_surface_temp_c", "dT_land_sst",
)


def _fetch_sst(region: dict, target, timezone: str, timeout: float) -> dict[int, float]:
    """SST oraria al punto mare; best effort e indipendente dal forecast meteo."""
    if "mare" not in region:
        return {}
    try:
        response = requests.get(
            MARINE_ENDPOINT,
            params={
                "latitude": region["mare"][0], "longitude": region["mare"][1],
                "hourly": "sea_surface_temperature", "timezone": timezone,
                "start_date": target.isoformat(), "end_date": target.isoformat(),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
    except (requests.RequestException, ValueError, AttributeError):
        return {}
    out: dict[int, float] = {}
    for timestamp, value in zip(
        hourly.get("time", []), hourly.get("sea_surface_temperature", []),
    ):
        if timestamp.startswith(target.isoformat()) and value is not None:
            out[int(timestamp[11:13])] = float(value)
    return out


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Rotta iniziale (gradi bussola) dal punto 1 al punto 2."""
    d_lon = math.radians(lon2 - lon1)
    y = math.sin(d_lon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(d_lon)
    return math.degrees(math.atan2(y, x)) % 360


def cross_isthmus_component(speed: float, wind_from_deg: float, bearing_to_spot: float) -> float:
    """Componente del flusso (dove VA il vento) lungo la rotta verso lo spot."""
    flow_dir = (wind_from_deg + 180.0) % 360.0
    return speed * math.cos(math.radians(flow_dir - bearing_to_spot))


def fetch_regional_features(
    spot: SpotConfig, target, timeout: float = 15.0
) -> dict[int, dict[str, float]]:
    """{ora: {feature: valore}} per il giorno `target`. Mai solleva."""
    region = spot.region or {}
    roles = [r for r in ("mare", "entroterra", "ionio") if r in region]
    if not roles:
        return {}
    lats = ",".join(str(region[r][0]) for r in roles)
    lons = ",".join(str(region[r][1]) for r in roles)
    try:
        response = requests.get(
            ENDPOINT,
            params={
                "latitude": lats,
                "longitude": lons,
                "hourly": "temperature_2m,pressure_msl,wind_speed_10m,wind_direction_10m",
                "wind_speed_unit": "kn",
                "timezone": spot.timezone,
                "start_date": target.isoformat(),
                "end_date": target.isoformat(),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return {}
    if isinstance(data, dict):
        data = [data]
    by_role = {role: data[i].get("hourly", {}) for i, role in enumerate(roles) if i < len(data)}
    sst_by_hour = _fetch_sst(region, target, spot.timezone, timeout)

    bearing = None
    if "ionio" in region:
        bearing = _bearing(region["ionio"][0], region["ionio"][1], spot.latitude, spot.longitude)

    ref = by_role.get(roles[0], {})
    times = ref.get("time", [])
    target_iso = target.isoformat()
    out: dict[int, dict[str, float]] = {}
    for i, t in enumerate(times):
        day, hh = t.split("T")
        if day != target_iso:
            continue

        def val(role: str, key: str):
            arr = by_role.get(role, {}).get(key, [])
            return arr[i] if i < len(arr) and arr[i] is not None else None

        feats: dict[str, float] = {}
        t_land, t_sea = val("entroterra", "temperature_2m"), val("mare", "temperature_2m")
        if t_land is not None and t_sea is not None:
            feats["dT_land_sea"] = round(float(t_land) - float(t_sea), 1)
        p_ionio, p_mare = val("ionio", "pressure_msl"), val("mare", "pressure_msl")
        if p_ionio is not None and p_mare is not None:
            feats["dP_ionio_mare"] = round(float(p_ionio) - float(p_mare), 2)
        if bearing is not None:
            s, d = val("ionio", "wind_speed_10m"), val("ionio", "wind_direction_10m")
            if s is not None and d is not None:
                feats["cross_isthmus"] = round(cross_isthmus_component(float(s), float(d), bearing), 1)
        syn = val("mare", "wind_speed_10m")
        if syn is not None:
            feats["synoptic_kn"] = round(float(syn), 1)
        sst = sst_by_hour.get(int(hh[:2]))
        if sst is not None:
            feats["sea_surface_temp_c"] = round(sst, 1)
            if t_land is not None:
                feats["dT_land_sst"] = round(float(t_land) - sst, 1)
        if feats:
            out[int(hh[:2])] = feats
    return out
