from __future__ import annotations

from datetime import date
from typing import Protocol

from kiteguru.models import ProviderResult, SpotConfig


class ForecastProvider(Protocol):
    source: str

    def fetch(self, spot: SpotConfig, target: date) -> ProviderResult:
        ...
