from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


CARDINAL_16 = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]


def degrees_to_cardinal(deg: float) -> str:
    normalized = deg % 360
    index = int((normalized + 11.25) // 22.5) % 16
    return CARDINAL_16[index]


def target_date(label: str | None, explicit_date: date | None, timezone: str) -> tuple[str, date]:
    today = datetime.now(ZoneInfo(timezone)).date()
    if explicit_date:
        return "date", explicit_date
    if label == "tomorrow":
        return "tomorrow", today + timedelta(days=1)
    return "today", today


def dominant_cardinal(directions: list[str]) -> str | None:
    if not directions:
        return None
    counts = Counter(directions)
    top_count = counts.most_common(1)[0][1]
    top = [direction for direction, count in counts.items() if count == top_count]
    return "/".join(top[:2])


def direction_span_degrees(degrees: list[float]) -> float:
    if len(degrees) < 2:
        return 0.0
    normalized = sorted(deg % 360 for deg in degrees)
    gaps = [
        normalized[i + 1] - normalized[i]
        for i in range(len(normalized) - 1)
    ]
    gaps.append((normalized[0] + 360) - normalized[-1])
    return 360 - max(gaps)
