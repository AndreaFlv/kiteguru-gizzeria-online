from __future__ import annotations

from collections import Counter
import json
from datetime import date as Date, datetime
from typing import Optional

import typer
from rich.console import Console

from .config import UnknownSpotError, get_spot
from .correction import apply_correction, build_correction, build_direction_correction
from .direction_model import DirectionModel, apply_direction_correction
from .models import KiteProfile
from .providers.holfuy import HolfuyProvider
from .providers.holfuy_chart import HolfuyChartProvider
from .providers.open_meteo import OpenMeteoProvider
from .providers.regional import fetch_regional_features
from .regime_classifier import classify_day, compute_delta_stats
from .providers.static_mock import StaticMockProvider
from .scoring import assess_day
from .storage import connect_readonly, frozen_predictions, onset_history_rows
from .thermal_model import train as train_thermal_model
from .utils import target_date

app = typer.Typer(help="Consulente operativo kitesurf per Gizzeria Lido.")


def _display_spot(name: str) -> str:
    if "Gizzeria" in name:
        return "Gizzeria Lido"
    return name


def _json_payload(assessment) -> dict:
    payload = {
        "spot": _display_spot(assessment.spot),
        "date_label": assessment.date_label,
        "date": assessment.date.isoformat(),
        "source": assessment.source,
        "wind_reference": assessment.wind_reference,
        "best_window": {
            "start": assessment.best_window.start,
            "end": assessment.best_window.end,
            "available": assessment.best_window.available,
        },
        "wind": {
            "avg_min_knots": assessment.wind_avg_min_knots,
            "avg_max_knots": assessment.wind_avg_max_knots,
            "gust_min_knots": assessment.gust_min_knots,
            "gust_max_knots": assessment.gust_max_knots,
            "dominant_direction": assessment.dominant_direction,
        },
        "stability": assessment.stability,
        "reliability": assessment.reliability,
        "score": assessment.score,
        "decision": assessment.decision,
        "profile": {
            "weight_kg": assessment.profile.weight_kg,
            "kite_size_m2": assessment.profile.kite_size_m2,
            "kite_model": assessment.profile.kite_model,
            "board": assessment.profile.board,
        },
        "notes": assessment.notes,
    }
    if assessment.confidence_interval is not None or assessment.prediction_method is not None:
        lo_hi = assessment.confidence_interval
        payload["prediction"] = {
            "lo": None if lo_hi is None else lo_hi[0],
            "hi": None if lo_hi is None else lo_hi[1],
            "p_kiteable": assessment.prediction_p_kiteable,
            "method": assessment.prediction_method,
        }
    if assessment.thermal_onset is not None:
        payload["thermal_onset"] = assessment.thermal_onset
    return payload


def _attach_frozen_prediction(assessment, spot_name: str) -> None:
    try:
        with connect_readonly() as conn:
            rows = frozen_predictions(conn, spot_name, assessment.date.isoformat())
    except FileNotFoundError:
        return
    if not rows:
        return
    if assessment.best_window.available:
        window_hours = {h.datetime.hour for h in assessment.best_window.hours}
        selected = [r for r in rows if int(r["obs_hour"]) in window_hours]
    else:
        selected = rows
    if not selected:
        selected = rows
    los = [float(r["pred_lo"]) for r in selected if r["pred_lo"] is not None]
    his = [float(r["pred_hi"]) for r in selected if r["pred_hi"] is not None]
    probs = [float(r["p_kiteable"]) for r in selected if r["p_kiteable"] is not None]
    methods = {r["method"] for r in selected if r["method"]}
    if los and his:
        assessment.confidence_interval = (round(min(los), 1), round(max(his), 1))
    if probs:
        assessment.prediction_p_kiteable = round(max(probs), 2)
    if methods:
        assessment.prediction_method = methods.pop() if len(methods) == 1 else "mixed"


def _regime_note(regime: str) -> str | None:
    if regime in {"TERMICO", "MISTO"}:
        return "Regime atteso: TERMICO - incremento tipico +2/+3 kn rispetto al previsto."
    if regime == "VENTURI":
        return "Regime atteso: VENTURI - possibile incremento fuori fascia termica, verifica stazione."
    if regime == "ANOMALO":
        return "Regime inclassificabile: dati insufficienti o condizione fuori distribuzione storica."
    return None


def _attach_regime(assessment, spot_config, hours, target: Date) -> None:
    try:
        with connect_readonly() as conn:
            stats = compute_delta_stats(conn, spot_config.name)
    except FileNotFoundError:
        stats = None

    try:
        regional_by_hour = fetch_regional_features(spot_config, target)
    except Exception:
        regional_by_hour = {}

    regimes = classify_day(hours, stats, regional_by_hour)
    if not regimes:
        return
    dominant = Counter(regimes.values()).most_common(1)[0][0]
    assessment.debug["regime_dominante"] = dominant
    assessment.debug["regime_ore"] = regimes
    note = _regime_note(dominant)
    if note:
        assessment.notes.append(note)


def _print_console(assessment, date_display: str, debug: bool) -> None:
    console = Console()
    window_text = (
        f"{assessment.best_window.start}-{assessment.best_window.end}"
        if assessment.best_window.available
        else "nessuna finestra stabile"
    )
    wind_text = (
        f"{assessment.wind_avg_min_knots}-{assessment.wind_avg_max_knots} kn"
        if assessment.wind_avg_min_knots is not None
        else "n/d"
    )
    gust_text = (
        f"{assessment.gust_min_knots}-{assessment.gust_max_knots} kn"
        if assessment.gust_min_knots is not None
        else "n/d"
    )
    direction_text = assessment.dominant_direction or "n/d"

    console.print(f"Spot: {_display_spot(assessment.spot)}")
    console.print(f"Giorno: {date_display}")
    console.print(f"Fonte dati: {assessment.source}")
    console.print("Dati vento: 10 m standard meteorologici")
    console.print()
    console.print(f"Finestra migliore: {window_text}")
    console.print(f"Vento medio: {wind_text}")
    console.print(f"Raffiche: {gust_text}")
    console.print(f"Direzione: {direction_text}")
    console.print(f"Stabilità: {assessment.stability}")
    console.print(f"Affidabilità: {assessment.reliability}")
    console.print(f"Punteggio: {assessment.score}/100")
    console.print()
    console.print(f"Valutazione: {assessment.decision}")
    if assessment.confidence_interval is not None:
        lo, hi = assessment.confidence_interval
        p_text = (
            f"  (P80: {assessment.prediction_p_kiteable * 100:.0f}%)"
            if assessment.prediction_p_kiteable is not None
            else ""
        )
        console.print(f"Intervallo previsione: {lo:.0f}-{hi:.0f} kn{p_text}")
    if assessment.prediction_method is not None:
        console.print(f"Metodo: {assessment.prediction_method}")
    console.print()
    console.print("Assetto Andrea:")
    console.print(f"Peso: {assessment.profile.weight_kg:g} kg")
    console.print(f"Kite: {assessment.profile.kite_model} {assessment.profile.kite_size_m2:g} m²")
    console.print(f"Tavola: {assessment.profile.board}")
    console.print()
    console.print("Note:")
    for note in assessment.notes:
        console.print(f"- {note}")
    if debug:
        console.print()
        console.print("Debug:")
        for key, value in assessment.debug.items():
            console.print(f"- {key}: {value}")


def _fetch_with_provider(provider_name: str, spot, target: Date):
    open_meteo = OpenMeteoProvider()
    mock = StaticMockProvider()
    if provider_name == "mock":
        return mock.fetch(spot, target), None
    if provider_name == "open-meteo":
        return open_meteo.fetch(spot, target), None
    result = open_meteo.fetch(spot, target)
    if result.hours:
        return result, None
    fallback = mock.fetch(spot, target)
    return fallback, result.error or "errore Open-Meteo"


@app.command()
def main(
    spot: str = typer.Argument(..., help="Spot da analizzare, es. gizzeria."),
    today: bool = typer.Option(False, "--today", help="Analizza oggi."),
    tomorrow: bool = typer.Option(False, "--tomorrow", help="Analizza domani."),
    date: Optional[str] = typer.Option(None, "--date", help="Analizza una data YYYY-MM-DD."),
    json_output: bool = typer.Option(False, "--json", help="Stampa solo JSON valido."),
    kite: float = typer.Option(10, "--kite", min=1, help="Misura kite in m²."),
    weight: float = typer.Option(75, "--weight", min=1, help="Peso rider in kg."),
    board: str = typer.Option("twintip", "--board", help="Tavola: twintip o foil."),
    debug: bool = typer.Option(False, "--debug", help="Mostra dettagli di scoring."),
    provider: str = typer.Option("auto", "--provider", help="Provider: auto, open-meteo o mock."),
    correggi: bool = typer.Option(False, "--correggi", help="Applica la correzione termica locale calibrata sui dati reali."),
    correggi_dir: bool = typer.Option(False, "--correggi-dir/--no-correggi-dir", help="Applica la correzione statistica circolare della direzione."),
    reale: bool = typer.Option(False, "--reale", help="Aggiunge la lettura attuale della stazione fisica (Holfuy)."),
) -> None:
    """Consulente operativo kitesurf per Gizzeria Lido."""
    try:
        if board not in {"twintip", "foil"}:
            raise typer.BadParameter("--board deve essere twintip o foil")
        if provider not in {"auto", "open-meteo", "mock"}:
            raise typer.BadParameter("--provider deve essere auto, open-meteo o mock")
        if sum(bool(x) for x in [today, tomorrow, date is not None]) > 1:
            raise typer.BadParameter("usa solo una tra --today, --tomorrow e --date")
        explicit_date = None
        if date is not None:
            try:
                explicit_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError as exc:
                raise typer.BadParameter("--date deve avere formato YYYY-MM-DD") from exc

        spot_config = get_spot(spot)
        label_hint = "tomorrow" if tomorrow else "today"
        date_label, target = target_date(label_hint, explicit_date, spot_config.timezone)
        result, real_error = _fetch_with_provider(provider, spot_config, target)
        if not result.hours:
            message = "Errore: impossibile recuperare dati reali da Open-Meteo."
            if json_output:
                print(json.dumps({"error": message, "details": result.error}, ensure_ascii=False))
            else:
                Console().print(message)
            raise typer.Exit(code=1)

        hours = result.hours
        source_label = result.source
        correction_note: str | None = None
        historical_rows: list[dict] = []
        if correggi:
            try:
                with connect_readonly() as conn:
                    correction = build_correction(conn, spot_config)
                    historical_rows = onset_history_rows(conn, spot_config.name)
            except FileNotFoundError:
                correction = train_thermal_model(spot_config, [])
            hours, max_delta = apply_correction(hours, correction, spot_config)
            if max_delta > 0:
                if correction.trained:
                    err = f", errore ~{correction.cv_mae or correction.mae:.0f} kn" if (correction.cv_mae or correction.mae) else ""
                    origin = f"modello tarato su {correction.n_samples} dati reali{err}"
                else:
                    origin = "stima fisica iniziale, non ancora tarata sui dati"
                correction_note = (
                    f"Valori corretti con modello termico locale (fino a +{max_delta:.0f} kn, {origin})."
                )
                source_label = f"{source_label} + correzione termica"
            else:
                correction_note = "Correzione termica richiesta ma non applicabile a questa direzione/fascia."
        else:
            try:
                with connect_readonly() as conn:
                    historical_rows = onset_history_rows(conn, spot_config.name)
            except FileNotFoundError:
                historical_rows = []

        direction_note: str | None = None
        if correggi_dir:
            try:
                with connect_readonly() as conn:
                    direction_model = build_direction_correction(conn, spot_config)
            except FileNotFoundError:
                direction_model = DirectionModel(
                    circular_bias_deg=0.0,
                    scatter_deg=999.0,
                    n_samples=0,
                    trained=False,
                )
            hours = apply_direction_correction(hours, direction_model)
            if direction_model.trained:
                direction_note = (
                    "Direzione corretta con bias circolare locale "
                    f"({direction_model.circular_bias_deg:+.0f}°, su {direction_model.n_samples} osservazioni, "
                    f"scatter ±{direction_model.scatter_deg:.0f}°)."
                )
                source_label = f"{source_label} + correzione direzione"
            else:
                direction_note = "Correzione direzione richiesta ma dati insufficienti (< 20 coppie disponibili)."

        profile = KiteProfile(weight_kg=weight, kite_size_m2=kite, board=board)
        assessment = assess_day(
            spot=spot_config,
            date_label=date_label,
            target=target,
            hours=hours,
            source=source_label,
            source_is_real=result.is_real,
            profile=profile,
            historical_rows=historical_rows,
        )
        if correction_note:
            assessment.notes.insert(0, correction_note)
        if direction_note:
            assessment.notes.insert(0, direction_note)
        if reale:
            observation = HolfuyProvider().fetch_current(spot_config)
            if observation is None:
                observation = HolfuyChartProvider().fetch_current(spot_config)
            if observation is not None:
                assessment.notes.insert(
                    0,
                    f"Stazione reale ora ({observation.datetime:%H:%M}): "
                    f"{observation.wind_speed_knots:.0f} kn raff. {observation.wind_gusts_knots:.0f} kn "
                    f"{observation.wind_direction_cardinal}.",
                )
            else:
                assessment.notes.insert(
                    0,
                    "Stazione reale non disponibile (serve HOLFUY_API_KEY per la stazione di questo spot).",
                )
        if real_error:
            assessment.notes.insert(0, "Errore: impossibile recuperare dati reali da Open-Meteo.")
        _attach_frozen_prediction(assessment, spot_config.name)
        _attach_regime(assessment, spot_config, hours, target)

        if json_output:
            print(json.dumps(_json_payload(assessment), ensure_ascii=False, separators=(",", ":")))
            return
        _print_console(assessment, "domani" if date_label == "tomorrow" else "oggi" if date_label == "today" else target.isoformat(), debug)
    except UnknownSpotError as exc:
        if json_output:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            Console().print(f"Errore: {exc}")
        raise typer.Exit(code=1)
    except typer.BadParameter:
        raise
    except Exception as exc:
        if json_output:
            print(json.dumps({"error": "Errore non gestibile dalla CLI", "details": str(exc)}, ensure_ascii=False))
        else:
            Console().print(f"Errore: {exc}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
