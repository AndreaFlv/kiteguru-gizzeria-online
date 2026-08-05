"""Send the two-day KiteGuru summary to one private Telegram chat.

The command is intentionally stateless: it fetches fresh public forecast data,
builds an inspectable text summary, and sends it only when both Telegram
credentials are present. Use ``--dry-run`` to print the exact message without
contacting Telegram.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import os
import sys
from zoneinfo import ZoneInfo

import requests

from kiteguru.config import get_spot
from kiteguru.correction import apply_correction
from kiteguru.models import ForecastHour, KiteProfile
from kiteguru.providers.open_meteo import OpenMeteoProvider
from kiteguru.scoring import assess_day, minimum_wind, weather_risk_summary
from kiteguru.thermal_model import train as train_thermal_model


DASHBOARD_URL = "https://kiteguru-gizzeria.streamlit.app/"
WEEKDAYS_IT = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")


@dataclass(frozen=True)
class DaySummary:
    label: str
    target: date
    decision: str
    window: str
    raw_wind: str
    scenario_wind: str
    direction: str
    rain: str
    thunderstorm: str


def _range_text(minimum: float | None, maximum: float | None) -> str:
    if minimum is None or maximum is None:
        return "n/d"
    return f"{minimum:.0f}-{maximum:.0f} kn"


def summarize_day(label: str, target: date) -> DaySummary:
    spot = get_spot("gizzeria")
    profile = KiteProfile(board="twintip", kite_size_m2=10.0, weight_kg=75.0)
    result = OpenMeteoProvider().fetch(spot, target)
    if not result.is_real or not result.hours:
        raise RuntimeError(result.error or f"Forecast non disponibile per {target.isoformat()}")

    raw_hours = [ForecastHour.model_validate(hour) for hour in result.hours]
    physical_prior = train_thermal_model(spot, [])
    scenario_hours, _ = apply_correction(raw_hours, physical_prior, spot)
    raw = assess_day(
        spot=spot,
        date_label=label.lower(),
        target=target,
        hours=raw_hours,
        source=result.source,
        source_is_real=True,
        profile=profile,
        historical_rows=[],
    )
    scenario = assess_day(
        spot=spot,
        date_label=label.lower(),
        target=target,
        hours=scenario_hours,
        source=result.source,
        source_is_real=True,
        profile=profile,
        historical_rows=[],
    )
    decision = raw.decision
    if raw.decision not in {"VAI", "VAI FORTE"} and scenario.decision in {"VAI", "VAI FORTE"}:
        decision = "CONTROLLA 14-16"

    risk = weather_risk_summary(raw_hours)
    rain_value = risk.get("max_precipitation_probability_pct")
    thunderstorm_hours = risk.get("thunderstorm_hours") or []
    return DaySummary(
        label=label,
        target=target,
        decision=decision,
        window=(
            f"{raw.best_window.start}-{raw.best_window.end}"
            if raw.best_window.available else "nessuna"
        ),
        raw_wind=_range_text(raw.wind_avg_min_knots, raw.wind_avg_max_knots),
        scenario_wind=_range_text(scenario.wind_avg_min_knots, scenario.wind_avg_max_knots),
        direction=raw.dominant_direction or "n/d",
        rain=f"{rain_value:.0f}%" if rain_value is not None else "NON_VALUTATA",
        thunderstorm=(
            ", ".join(f"{hour:02d}:00" for hour in thunderstorm_hours)
            if thunderstorm_hours else "non indicato"
        ),
    )


def build_message(today: date | None = None) -> str:
    spot = get_spot("gizzeria")
    local_today = today or datetime.now(ZoneInfo(spot.timezone)).date()
    summaries = [
        summarize_day("Domani", local_today + timedelta(days=1)),
        summarize_day("Dopodomani", local_today + timedelta(days=2)),
    ]
    blocks = ["<b>🌬️ KiteGuru · Gizzeria</b>"]
    for item in summaries:
        blocks.append(
            "\n".join(
                [
                    f"<b>{item.label} · {WEEKDAYS_IT[item.target.weekday()]} {item.target:%d/%m}</b>",
                    f"{item.decision}",
                    f"Finestra grezza: {item.window}",
                    f"Open-Meteo: {item.raw_wind} · {item.direction}",
                    f"Scenario termico non calibrato: {item.scenario_wind}",
                    f"Pioggia max: {item.rain} · temporali: {item.thunderstorm}",
                ]
            )
        )
    blocks.append(
        "Lo scenario termico è indicativo e non può da solo generare un VAI. "
        "Controlla sempre stazione, radar e condizioni reali."
    )
    return "\n\n".join(blocks)


def send_message(text: str, token: str, chat_id: str) -> None:
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
            "reply_markup": {
                "inline_keyboard": [[{"text": "Apri previsione completa", "url": DASHBOARD_URL}]]
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("description") or "Telegram ha rifiutato il messaggio")


def main() -> int:
    parser = argparse.ArgumentParser(description="Invia la previsione KiteGuru a Telegram")
    parser.add_argument("--dry-run", action="store_true", help="stampa il messaggio senza inviarlo")
    args = parser.parse_args()
    text = build_message()
    if args.dry_run:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(text)
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Servono TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID")
    send_message(text, token, chat_id)
    print("Messaggio Telegram inviato")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
