from __future__ import annotations

import json
from importlib.resources import files

from .models import SpotConfig


class UnknownSpotError(ValueError):
    pass


def load_spots() -> dict[str, SpotConfig]:
    path = files("kiteguru").joinpath("config/spots.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: SpotConfig.model_validate(value) for key, value in raw.items()}


def get_spot(slug: str) -> SpotConfig:
    spots = load_spots()
    key = slug.lower().strip()
    if key not in spots:
        raise UnknownSpotError(f"Spot sconosciuto: {slug}")
    return spots[key]
