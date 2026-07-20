"""Predittore ad analoghi per la previsione del giorno dopo (probabilistica).

Idea (analog ensemble): per prevedere una certa ora di domani, cerca nel DB i
casi passati con una situazione *simile* (stessa fascia oraria + feature di
forecast/regionali vicine) e guarda **cosa ha fatto davvero il vento** in quei
casi. La distribuzione dei reali analoghi da' mediana + banda (incertezza) +
probabilita' di vento navigabile.

Con pochi analoghi degrada in modo pulito sul modello fisico (grey-box).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

ONSHORE_REF = 270.0  # W, asse onshore di Gizzeria
FEATS = (
    "radiation", "fwind", "onshore", "synoptic", "cross", "dT",
    "dT_sst", "boundary_layer",
)
MIN_ANALOGS = 5
K_NEIGHBORS = 10


def _features(rec: dict) -> dict[str, float]:
    rad = rec.get("f_radiation") or 0.0
    fw = rec.get("f_wind_speed") or 0.0
    fdir = rec.get("f_wind_dir_deg")
    onshore = fw * math.cos(math.radians(fdir - ONSHORE_REF)) if fdir is not None else 0.0
    return {
        "radiation": float(rad),
        "fwind": float(fw),
        "onshore": float(onshore),
        "synoptic": float(rec.get("synoptic_kn") or 0.0),
        "cross": float(rec.get("cross_isthmus") or 0.0),
        "dT": float(rec.get("dT_land_sea") or 0.0),
        "dT_sst": float(rec.get("dT_land_sst") or 0.0),
        "boundary_layer": float(rec.get("boundary_layer_height_m") or 0.0),
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac
    return sorted_vals[lo]


@dataclass
class Forecast:
    median: float
    lo: float
    hi: float
    n_analogs: int
    p_kiteable: float
    method: str


class AnalogPredictor:
    def __init__(self, rows) -> None:
        recs = [dict(r) for r in rows]  # accetta sia sqlite3.Row sia dict
        self.items = [(int(r["hour"]), _features(r), float(r["real_speed"])) for r in recs]
        self.std = {}
        for key in FEATS:
            vals = [f[key] for _, f, _ in self.items]
            self.std[key] = (statistics.pstdev(vals) if len(vals) > 1 else 0.0) or 1.0

    def predict(self, target: dict[str, float], hour: int, threshold: float, window: int = 1) -> Forecast | None:
        cand = [(f, r) for (h, f, r) in self.items if abs(h - hour) <= window]
        if len(cand) < MIN_ANALOGS:
            return None

        def dist(f: dict[str, float]) -> float:
            return sum(((f[k] - target[k]) / self.std[k]) ** 2 for k in FEATS)

        cand.sort(key=lambda fr: dist(fr[0]))
        reals = [r for _, r in cand[:K_NEIGHBORS]]
        ordered = sorted(reals)
        return Forecast(
            median=round(statistics.median(reals), 1),
            lo=round(_quantile(ordered, 0.25), 1),
            hi=round(_quantile(ordered, 0.75), 1),
            n_analogs=len(reals),
            p_kiteable=round(sum(1 for r in reals if r >= threshold) / len(reals), 2),
            method="analog",
        )


def predict_day(
    conn, spot, target, threshold: float = 13.0, hours=None, regional=None,
) -> dict[int, Forecast]:
    """Previsione oraria probabilistica per `target` (di norma domani).

    Analoghi dove ci sono abbastanza casi simili, altrimenti modello fisico.
    """
    from .correction import build_correction
    from .providers.open_meteo import OpenMeteoProvider
    from .providers.regional import fetch_regional_features
    from .scoring import USEFUL_END, USEFUL_START, direction_category
    from .storage import feature_rows

    if hours is None:
        hours = OpenMeteoProvider().fetch(spot, target).hours
    if regional is None:
        regional = fetch_regional_features(spot, target)
    predictor = AnalogPredictor(feature_rows(conn, spot.name))
    model = build_correction(conn, spot)

    out: dict[int, Forecast] = {}
    for h in hours:
        hr = h.datetime.hour
        if not (USEFUL_START <= hr <= USEFUL_END):
            continue
        reg = regional.get(hr, {})
        rec = {
            "f_radiation": h.radiation, "f_wind_speed": h.wind_speed_knots,
            "f_wind_dir_deg": h.wind_direction_degrees,
            "synoptic_kn": reg.get("synoptic_kn"), "cross_isthmus": reg.get("cross_isthmus"),
            "dT_land_sea": reg.get("dT_land_sea"),
            "dT_land_sst": reg.get("dT_land_sst"),
            "boundary_layer_height_m": h.boundary_layer_height_m,
        }
        forecast = predictor.predict(_features(rec), hr, threshold)
        if forecast is None:
            boost = model.boost_for_hour(h) if direction_category(h.wind_direction_cardinal, spot) != "bad" else 0.0
            med = round(max(0.0, h.wind_speed_knots + boost), 1)
            forecast = Forecast(med, max(0.0, round(med - 3, 1)), round(med + 3, 1), 0,
                                1.0 if med >= threshold else 0.0, "fisico")
        out[hr] = forecast
    return out
