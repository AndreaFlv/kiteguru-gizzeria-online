"""Applicazione della correzione termica alle ore di forecast.

La logica di apprendimento vive in `thermal_model.ThermalModel`; qui restano la
costruzione del modello dal DB e l'applicazione alle ore (solo direzioni utili).
"""
from __future__ import annotations

import sqlite3

from .models import ForecastHour, SpotConfig
from .storage import paired_rows
from .thermal_model import ThermalModel, train
from .direction_model import DirectionModel, train_direction


def _correctable(card: str, spot: SpotConfig) -> bool:
    # Il termico rinforza la brezza onshore (W); niente bonus sulle altre direzioni.
    return card in spot.preferred_directions or card in spot.acceptable_directions


def build_correction(conn: sqlite3.Connection, spot: SpotConfig) -> ThermalModel:
    rows = [dict(r) for r in paired_rows(conn, spot.name)]
    return train(spot, rows)


def build_direction_correction(conn: sqlite3.Connection, spot: SpotConfig) -> DirectionModel:
    return train_direction(conn, spot.name)


def apply_correction(
    hours: list[ForecastHour], model: ThermalModel, spot: SpotConfig
) -> tuple[list[ForecastHour], float]:
    """Ore con vento corretto dal modello termico; restituisce anche il delta massimo.

    NOTA sulle raffiche: il boost viene sommato anche alla raffica *internamente*,
    solo per preservare il divario raffica-medio (altrimenti la raffica corretta
    potrebbe risultare < del vento medio corretto, valore non fisico) e mantenere
    coerente il punteggio sulle raffiche. NON e' una previsione di raffica nostra:
    sulla raffica non abbiamo una correzione propria. Per questo la dashboard
    MOSTRA sempre la raffica grezza di Open-Meteo, non questa.
    """
    corrected: list[ForecastHour] = []
    max_delta = 0.0
    for hour in hours:
        if not _correctable(hour.wind_direction_cardinal, spot):
            corrected.append(hour)
            continue
        delta = model.boost_for_hour(hour)
        if abs(delta) >= 0.5:
            max_delta = max(max_delta, delta)
            corrected.append(
                hour.model_copy(
                    update={
                        "wind_speed_knots": max(0.0, hour.wind_speed_knots + delta),
                        "wind_gusts_knots": max(0.0, hour.wind_gusts_knots + delta),
                    }
                )
            )
        else:
            corrected.append(hour)
    return corrected, max_delta
