from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from kiteguru.models import ForecastHour, ProviderResult, SpotConfig
from kiteguru.utils import degrees_to_cardinal


class StaticMockProvider:
    source = "MOCK/FALLBACK - dati non reali"

    def fetch(self, spot: SpotConfig, target: date) -> ProviderResult:
        tz = ZoneInfo(spot.timezone)
        pattern = {
            0: (5, 8, 90),
            1: (5, 8, 90),
            2: (5, 8, 90),
            3: (5, 8, 90),
            4: (6, 9, 100),
            5: (6, 9, 100),
            6: (7, 10, 120),
            7: (8, 11, 190),
            8: (9, 13, 220),
            9: (11, 15, 240),
            10: (13, 17, 250),
            11: (14, 18, 255),
            12: (16, 21, 260),
            13: (17, 22, 260),
            14: (18, 23, 255),
            15: (17, 24, 250),
            16: (16, 22, 250),
            17: (14, 20, 260),
            18: (12, 18, 270),
            19: (10, 15, 280),
            20: (8, 12, 300),
            21: (7, 10, 320),
            22: (6, 9, 20),
            23: (5, 8, 70),
        }
        hours = []
        for hour in range(24):
            speed, gust, direction = pattern[hour]
            dt = datetime.combine(target, time(hour=hour), tzinfo=tz)
            hours.append(
                ForecastHour(
                    datetime=dt,
                    wind_speed_knots=float(speed),
                    wind_gusts_knots=float(gust),
                    wind_direction_degrees=float(direction),
                    wind_direction_cardinal=degrees_to_cardinal(direction),
                    source=self.source,
                )
            )
        return ProviderResult(source=self.source, is_real=False, hours=hours)
