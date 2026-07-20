from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


BoardType = Literal["twintip", "foil"]
Decision = Literal["LASCIA PERDERE", "CONTROLLA 14-16", "MARGINALE", "VAI", "VAI FORTE"]


class SpotConfig(BaseModel):
    name: str
    latitude: float
    longitude: float
    timezone: str
    preferred_directions: list[str]
    acceptable_directions: list[str]
    bad_directions: list[str]
    holfuy_station: int | None = None
    thermal_seed: dict[str, float] = Field(default_factory=dict)
    # punti regionali per le feature di contesto: ruolo -> [lat, lon]
    # ruoli attesi: "mare" (Tirreno al largo), "entroterra", "ionio" (Golfo di Squillace)
    region: dict[str, list[float]] = Field(default_factory=dict)


class ForecastHour(BaseModel):
    datetime: datetime
    wind_speed_knots: float
    wind_gusts_knots: float
    wind_direction_degrees: float
    wind_direction_cardinal: str
    source: str
    # variabili meteo che alimentano il modello termico (None se non disponibili)
    temp_c: float | None = None
    cloud_pct: float | None = None
    radiation: float | None = None
    pressure_hpa: float | None = None
    boundary_layer_height_m: float | None = None


class ForecastDay(BaseModel):
    date: date
    hours: list[ForecastHour]
    source: str
    is_real: bool = True


class KiteProfile(BaseModel):
    weight_kg: float = 75
    kite_size_m2: float = 10
    kite_model: str = "North Orbit"
    board: BoardType = "twintip"


class BestWindow(BaseModel):
    available: bool
    start: str | None = None
    end: str | None = None
    hours: list[ForecastHour] = Field(default_factory=list, exclude=True)


class KiteAssessment(BaseModel):
    spot: str
    date_label: str
    date: date
    source: str
    wind_reference: str = "10m_standard_meteorological"
    best_window: BestWindow
    wind_avg_min_knots: int | None
    wind_avg_max_knots: int | None
    gust_min_knots: int | None
    gust_max_knots: int | None
    dominant_direction: str | None
    stability: str
    reliability: str
    score: int
    decision: Decision
    profile: KiteProfile
    notes: list[str]
    confidence_interval: tuple[float, float] | None = None
    prediction_method: str | None = None
    prediction_p_kiteable: float | None = None
    thermal_onset: dict | None = None
    debug: dict[str, object] = Field(default_factory=dict)


class ProviderResult(BaseModel):
    source: str
    is_real: bool
    hours: list[ForecastHour]
    error: str | None = None


class RealObservation(BaseModel):
    """Singola lettura misurata da una stazione fisica (es. Holfuy)."""

    datetime: datetime
    wind_speed_knots: float
    wind_gusts_knots: float
    wind_direction_degrees: float
    wind_direction_cardinal: str
    source: str
    temp_c: float | None = None
