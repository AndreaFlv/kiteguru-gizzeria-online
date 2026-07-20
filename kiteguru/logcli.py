from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

from .config import UnknownSpotError, get_spot
from .correction import apply_correction, build_correction
from .models import KiteProfile
from .providers.holfuy_chart import HolfuyChartProvider, aggregate_hourly
from .providers.open_meteo import OpenMeteoProvider
from .providers.open_meteo_models import fetch_model_winds
from .providers.regional import fetch_regional_features
from .regime_classifier import classify_hour, compute_delta_stats
from .scoring import USEFUL_END, USEFUL_START
from .storage import (
    connect,
    counts,
    log_model_forecast,
    log_alternative_reading,
    log_forecast_snapshot,
    log_pair,
    log_pipeline_run,
    log_prediction,
    log_raw_points,
    log_regime,
    log_regional_features,
    onset_history_rows,
)

app = typer.Typer(help="Registra previsione e dato reale nel database KiteGuru.")


@app.command()
def log(
    spot: str = typer.Argument("gizzeria", help="Spot da registrare."),
    quiet: bool = typer.Option(False, "--quiet", help="Nessun output, solo scrittura DB."),
    backfill: bool = typer.Option(False, "--backfill", help="Importa tutto lo storico reale disponibile dal grafico pubblico."),
) -> None:
    """Registra le coppie previsione/reale per le ore utili di oggi gia' trascorse.

    A ogni esecuzione prende dal grafico pubblico Holfuy TUTTE le ore del giorno
    disponibili (non solo l'ora corrente) e le accoppia con la previsione
    Open-Meteo e con tutti i modelli confrontati. L'upsert evita duplicati, quindi
    rieseguirlo piu' volte al giorno arricchisce e aggiorna le stesse righe.
    """
    console = Console()
    try:
        spot_config = get_spot(spot)
    except UnknownSpotError as exc:
        console.print(f"Errore: {exc}")
        raise typer.Exit(code=1)

    now = datetime.now(ZoneInfo(spot_config.timezone))
    started_at_utc = datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds")
    today = now.date()
    today_iso = today.isoformat()

    series = HolfuyChartProvider().fetch_series(spot_config)
    real_hourly = aggregate_hourly(series)

    if backfill:
        with connect() as conn:
            raw = log_raw_points(conn, spot=spot_config.name, observations=series)
            for (date_iso, hour), obs in sorted(real_hourly.items()):
                log_pair(conn, spot=spot_config.name, obs_date=date_iso, obs_hour=hour,
                         forecast=None, real=obs)
            total, real_count = counts(conn, spot_config.name)
        if not quiet:
            console.print(f"Backfill: {raw} punti grezzi + {len(real_hourly)} ore reali dal grafico pubblico.")
            console.print(f"DB: {total} righe ({real_count} con misura reale)")
        return

    provider = OpenMeteoProvider()
    today_result = provider.fetch(spot_config, today)
    forecast_by_hour = {h.datetime.hour: h for h in today_result.hours}
    failures: list[str] = []
    warnings: list[str] = []
    if not series:
        failures.append("centralina Holfuy primaria: nessun dato restituito")
    else:
        latest = max(obs.datetime for obs in series)
        age_minutes = (now - latest.astimezone(now.tzinfo)).total_seconds() / 60.0
        if age_minutes > 45:
            warnings.append(
                f"centralina Holfuy primaria: ultimo dato vecchio di {age_minutes:.0f} minuti"
            )
    if not forecast_by_hour:
        failures.append(f"forecast {today_iso}: {today_result.error or 'nessuna ora restituita'}")
    regional = fetch_regional_features(spot_config, today)
    if not regional:
        warnings.append(f"feature regionali {today_iso}: nessun dato restituito")
    alternative = None
    try:
        from .providers.gizzeriakite import GizzeriaKiteProvider

        alternative = GizzeriaKiteProvider().fetch()
    except Exception as exc:
        # The secondary station is deliberately non-critical until enough
        # overlap proves its calibration and reliability.
        warnings.append(f"centralina alternativa: {exc}")

    logged, last = 0, None
    thermal_onset = None
    regime_counts: Counter[str] = Counter()
    with connect() as conn:
        log_raw_points(conn, spot=spot_config.name, observations=series)
        if alternative is not None:
            log_alternative_reading(conn, spot=spot_config.name, reading=alternative)
        for hour in range(USEFUL_START, min(USEFUL_END, now.hour) + 1):
            real = real_hourly.get((today_iso, hour))
            forecast = forecast_by_hour.get(hour)
            if real is None and forecast is None:
                continue
            log_pair(conn, spot=spot_config.name, obs_date=today_iso, obs_hour=hour,
                     forecast=forecast, real=real)
            if hour in regional:
                log_regional_features(conn, spot=spot_config.name, obs_date=today_iso,
                                      obs_hour=hour, feats=regional[hour])
            if real is not None:
                logged += 1
                last = (hour, forecast, real)

        stats = compute_delta_stats(conn, spot_config.name)
        for hour in range(USEFUL_START, min(USEFUL_END, now.hour) + 1):
            forecast = forecast_by_hour.get(hour)
            real = real_hourly.get((today_iso, hour))
            if forecast is None or real is None:
                continue
            regime = classify_hour(forecast, stats, regional.get(hour, {}))
            delta_actual = real.wind_speed_knots - forecast.wind_speed_knots
            log_regime(
                conn,
                spot=spot_config.name,
                obs_date=today_iso,
                obs_hour=hour,
                regime=regime,
                delta_actual=round(delta_actual, 3),
            )
            regime_counts[regime] += 1

        # Run serale: congela la previsione probabilistica del giorno dopo.
        predicted = 0
        if now.hour >= 19:
            from .analog import predict_day
            from .gb_model import predict_day_gb

            tomorrow = today + timedelta(days=1)
            tomorrow_result = provider.fetch(spot_config, tomorrow)
            if not tomorrow_result.hours:
                failures.append(
                    f"forecast {tomorrow.isoformat()}: "
                    f"{tomorrow_result.error or 'nessuna ora restituita'}"
                )
                forecasts = {}
                method_used = "non disponibile"
            else:
                tomorrow_regional = fetch_regional_features(spot_config, tomorrow)
                tomorrow_models = fetch_model_winds(spot_config, tomorrow)
                if not tomorrow_regional:
                    warnings.append(
                        f"feature regionali {tomorrow.isoformat()}: nessun dato restituito"
                    )
                if not any(tomorrow_models.values()):
                    warnings.append(
                        f"modelli comparativi {tomorrow.isoformat()}: nessun dato restituito"
                    )
                for hour_forecast in tomorrow_result.hours:
                    hour_value = hour_forecast.datetime.hour
                    if USEFUL_START <= hour_value <= USEFUL_END:
                        log_forecast_snapshot(
                            conn,
                            spot=spot_config.name,
                            target_date=tomorrow.isoformat(),
                            forecast=hour_forecast,
                            regional=tomorrow_regional.get(hour_value, {}),
                        )
                for model, hours_map in tomorrow_models.items():
                    for hour_value, speed in hours_map.items():
                        if USEFUL_START <= hour_value <= USEFUL_END:
                            log_model_forecast(
                                conn, spot=spot_config.name,
                                obs_date=tomorrow.isoformat(), obs_hour=hour_value,
                                model=model, speed=speed, origin="day_ahead",
                            )
                forecasts = predict_day_gb(
                    conn, spot_config, tomorrow, hours=tomorrow_result.hours,
                    regional=tomorrow_regional,
                )
                method_used = "lgbm" if forecasts else "analog/fisico"
                if not forecasts:
                    forecasts = predict_day(
                        conn, spot_config, tomorrow, hours=tomorrow_result.hours,
                        regional=tomorrow_regional,
                    )
            for hr, fc in forecasts.items():
                log_prediction(conn, spot=spot_config.name, target_date=tomorrow.isoformat(),
                               obs_hour=hr, method=fc.method, median=fc.median, lo=fc.lo,
                               hi=fc.hi, p_kiteable=fc.p_kiteable)
                predicted += 1
        correction = build_correction(conn, spot_config)
        corrected_hours = apply_correction(list(forecast_by_hour.values()), correction, spot_config)[0]
        from .thermal_onset import estimate_onset

        thermal_onset = estimate_onset(
            corrected_hours,
            spot_config,
            KiteProfile(),
            onset_history_rows(conn, spot_config.name),
        )
        total, real_count = counts(conn, spot_config.name)
        run_status = "failed" if failures else ("degraded" if warnings else "ok")
        log_pipeline_run(
            conn,
            started_at_utc=started_at_utc,
            spot=spot_config.name,
            status=run_status,
            observations_logged=logged,
            predictions_logged=predicted,
            details=json.dumps(
                {"errors": failures, "warnings": warnings}, ensure_ascii=False,
            ),
        )

    if quiet and failures:
        raise typer.Exit(code=1)
    if quiet:
        return
    console.print(f"[{today_iso}] {spot_config.name}: registrate {logged} ore utili di oggi (10-{min(USEFUL_END, now.hour)}).")
    if last:
        hour, forecast, real = last
        f_txt = f"{forecast.wind_speed_knots:.0f} kn {forecast.wind_direction_cardinal}" if forecast else "n/d"
        console.print(f"  ultima ({hour:02d}:00)  previsto: {f_txt}  |  reale: {real.wind_speed_knots:.0f} kn {real.wind_direction_cardinal}")
    if predicted:
        console.print(f"  previsione di domani congelata: {predicted} ore (metodo: {method_used}).")
    if thermal_onset and thermal_onset.get("onset_hour") is not None:
        console.print(
            "  Termico atteso: "
            f"{thermal_onset['onset_label']} ±{thermal_onset['uncertainty_hours']:g}h "
            f"(confidenza: {thermal_onset['confidence']})"
        )
    if regime_counts:
        regime, count = regime_counts.most_common(1)[0]
        console.print(f"  Regime dominante: {regime} ({count}/{sum(regime_counts.values())} ore)")
    console.print(f"  DB: {total} righe ({real_count} con misura reale)")
    if failures:
        for failure in failures:
            console.print(f"[red]Errore pipeline: {failure}[/red]")
        raise typer.Exit(code=1)
    if warnings:
        for warning in warnings:
            console.print(f"[yellow]Avviso pipeline: {warning}[/yellow]")


if __name__ == "__main__":
    app()
