"""Modello adattivo di correzione termica.

Obiettivo: predire il *boost* termico `b = vento_reale - vento_previsto` per una
data ora futura, usando SOLO variabili disponibili nel forecast (cosi' la
correzione e' applicabile a domani/dopodomani).

Filosofia "grey-box" (scatola grigia):
- La FISICA della brezza di mare fornisce le feature (insolazione × finestra
  diurna, componente di vento onshore/offshore). Questo da' un forte bias
  induttivo che permette di imparare da pochissimi dati.
- Un piccolo modello lineare regolarizzato (ridge) fitta i coefficienti.
- Con pochi campioni il modello viene "shrinkato" verso un prior fisico (il
  seed per banda oraria); man mano che i dati crescono, il peso passa ai dati.

Il modello e' onesto: espone n. campioni, errore (MAE) e coefficienti, e quando
i dati non bastano lo dichiara invece di fingere precisione.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .models import ForecastHour, SpotConfig
from .utils import CARDINAL_16

# Quante coppie reali/previste servono prima di fidarsi del modello fittato.
MIN_TRAIN = 8
# Costante di shrinkage: peso_dati = n / (n + SHRINK_K). Con n=SHRINK_K il
# modello pesa il 50% e il prior fisico il 50%.
SHRINK_K = 15.0
RIDGE_LAMBDA = 1.0
BOOST_MIN, BOOST_MAX = -5.0, 16.0
RADIATION_NORM = 800.0  # W/m^2, insolazione di riferimento per normalizzare
FEATURES = ("thermal", "onshore", "offshore")


def thermal_shape(hour: int) -> float:
    """Finestra diurna del termico: campana centrata ~15:00, ~0 fuori 9-20."""
    return math.exp(-((hour - 15.0) ** 2) / (2 * 3.0 ** 2)) if 8 <= hour <= 21 else 0.0


def _onshore_reference_deg(spot: SpotConfig) -> float:
    """Direzione 'onshore' (verso terra) come media circolare delle preferite."""
    degs = [CARDINAL_16.index(c) * 22.5 for c in spot.preferred_directions if c in CARDINAL_16]
    if not degs:
        return 270.0
    s = sum(math.sin(math.radians(d)) for d in degs)
    c = sum(math.cos(math.radians(d)) for d in degs)
    return math.degrees(math.atan2(s, c)) % 360


def hour_band(hour: int) -> str:
    if hour <= 11:
        return "pre"
    if hour <= 16:
        return "termico"
    return "post"


def features_from_values(
    hour: int, radiation: float | None, wind_speed: float | None, wind_dir_deg: float | None,
    onshore_ref: float,
) -> dict[str, float]:
    """Estrae il vettore di feature fisiche (uguale in training e predizione)."""
    diurnal = thermal_shape(hour)
    solar = (radiation if radiation is not None else 0.0) / RADIATION_NORM
    speed = wind_speed or 0.0
    if wind_dir_deg is None:
        proj = 0.0
    else:
        # componente lungo l'asse onshore: >0 se il vento sinottico spinge verso terra
        proj = speed * math.cos(math.radians(wind_dir_deg - onshore_ref))
    return {
        "thermal": solar * diurnal,            # riscaldamento durante il giorno
        "onshore": max(0.0, proj),             # vento sinottico che rinforza la brezza
        "offshore": max(0.0, -proj),           # vento sinottico che la contrasta
    }


def _features_from_hour(hour: ForecastHour, onshore_ref: float) -> dict[str, float]:
    return features_from_values(
        hour.datetime.hour, hour.radiation, hour.wind_speed_knots,
        hour.wind_direction_degrees, onshore_ref,
    )


@dataclass
class ThermalModel:
    spot: str
    onshore_ref: float
    seed: dict[str, float] = field(default_factory=dict)
    # ridge fittato (None finche' i dati non bastano)
    weights: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    intercept: float = 0.0
    n_samples: int = 0
    mae: float | None = None
    cv_mae: float | None = None

    @property
    def trained(self) -> bool:
        return self.weights is not None

    @property
    def blend_alpha(self) -> float:
        """Peso del modello-dati rispetto al prior fisico (0..1)."""
        return self.n_samples / (self.n_samples + SHRINK_K) if self.trained else 0.0

    def _model_boost(self, feats: dict[str, float]) -> float:
        x = np.array([feats[k] for k in FEATURES], dtype=float)
        xs = (x - self.mean) / self.std
        return float(self.intercept + xs @ self.weights)

    def _seed_boost(self, hour: int, feats: dict[str, float], has_radiation: bool) -> float:
        """Prior fisico (modello non ancora tarato).

        Se l'insolazione e' nota, il boost scala con l'indice termico
        (sole × finestra diurna): cosi' un'ora nuvolosa o fuori picco riceve
        poco, evitando il +7 piatto su ogni giorno. L'ampiezza di picco e' il
        seed della fascia termica. Senza insolazione, ricade sul seed per banda.
        """
        if has_radiation:
            peak = float(self.seed.get("termico", 0.0))
            return peak * feats["thermal"]
        return float(self.seed.get(hour_band(hour), 0.0))

    def boost_for_values(
        self, hour: int, radiation: float | None, wind_speed: float | None, wind_dir_deg: float | None,
    ) -> float:
        feats = features_from_values(hour, radiation, wind_speed, wind_dir_deg, self.onshore_ref)
        seed = self._seed_boost(hour, feats, radiation is not None)
        if self.trained:
            a = self.blend_alpha
            boost = a * self._model_boost(feats) + (1 - a) * seed
        else:
            boost = seed
        return float(max(BOOST_MIN, min(BOOST_MAX, boost)))

    def boost_for_hour(self, hour: ForecastHour) -> float:
        return self.boost_for_values(
            hour.datetime.hour, hour.radiation, hour.wind_speed_knots, hour.wind_direction_degrees,
        )

    def summary(self) -> dict:
        return {
            "trained": self.trained,
            "n_samples": self.n_samples,
            "blend_alpha": round(self.blend_alpha, 2),
            "mae": None if self.mae is None else round(self.mae, 2),
            "cv_mae": None if self.cv_mae is None else round(self.cv_mae, 2),
            "coef": None if not self.trained else {
                k: round(float(w), 2) for k, w in zip(FEATURES, self.weights)
            },
            "seed": self.seed,
        }


def _ridge_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    k = Xs.shape[1]
    A = Xs.T @ Xs + RIDGE_LAMBDA * np.eye(k)
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    intercept = float(y.mean())
    return w, intercept, mean, std


def train(
    spot: SpotConfig,
    rows: Iterable[dict],
) -> ThermalModel:
    """Allena il modello dalle righe DB con feature di forecast + misura reale.

    Ogni riga deve avere: obs_hour, f_radiation, f_wind_speed, f_wind_dir_deg,
    forecast_speed, real_speed, forecast_dir_card.
    """
    onshore_ref = _onshore_reference_deg(spot)
    seed = {"pre": 0.0, "termico": 0.0, "post": 0.0, **spot.thermal_seed}
    model = ThermalModel(spot=spot.name, onshore_ref=onshore_ref, seed=seed)

    X, y, groups = [], [], []
    for r in rows:
        if r.get("real_speed") is None or r.get("forecast_speed") is None:
            continue
        card = r.get("forecast_dir_card")
        if card not in spot.preferred_directions and card not in spot.acceptable_directions:
            continue  # impariamo il termico solo sulle direzioni utili (brezza onshore)
        feats = features_from_values(
            int(r["obs_hour"]), r.get("f_radiation"), r.get("f_wind_speed"), r.get("f_wind_dir_deg"),
            onshore_ref,
        )
        X.append([feats[k] for k in FEATURES])
        y.append(float(r["real_speed"]) - float(r["forecast_speed"]))
        groups.append(str(r.get("obs_date") or "unknown"))

    model.n_samples = len(y)
    if len(y) < MIN_TRAIN:
        return model  # resta sul prior fisico (seed)

    Xa, ya = np.array(X, dtype=float), np.array(y, dtype=float)
    w, intercept, mean, std = _ridge_fit(Xa, ya)
    model.weights, model.intercept, model.mean, model.std = w, intercept, mean, std

    pred = intercept + ((Xa - mean) / std) @ w
    model.mae = float(np.mean(np.abs(pred - ya)))
    model.cv_mae = _leave_day_out_cv_mae(Xa, ya, groups)
    return model


def _leave_day_out_cv_mae(
    X: np.ndarray, y: np.ndarray, groups: list[str],
) -> float | None:
    """Leave-one-day-out MAE; hours from one day never leak across folds."""
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        return None
    errs = []
    group_array = np.array(groups)
    for group in unique_groups:
        train_mask = group_array != group
        test_mask = group_array == group
        if int(train_mask.sum()) < MIN_TRAIN or not test_mask.any():
            continue
        w, intercept, mean, std = _ridge_fit(X[train_mask], y[train_mask])
        pred = intercept + ((X[test_mask] - mean) / std) @ w
        errs.extend(np.abs(pred - y[test_mask]).tolist())
    if not errs:
        return None
    return float(np.mean(errs))
