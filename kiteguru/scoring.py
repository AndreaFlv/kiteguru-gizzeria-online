from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .models import BestWindow, ForecastHour, KiteAssessment, KiteProfile, SpotConfig
from .utils import direction_span_degrees, dominant_cardinal

USEFUL_START = 10
USEFUL_END = 19
THERMAL_START = 12
THERMAL_END = 17
THERMAL_CHECK_LABEL = "CONTROLLA 14-16"


@dataclass(frozen=True)
class HourScore:
    hour: ForecastHour
    acceptable: bool
    score: float


def direction_category(direction: str, spot: SpotConfig) -> str:
    if direction in spot.preferred_directions:
        return "preferred"
    if direction in spot.acceptable_directions:
        return "acceptable"
    if direction in {"SSW", "NNW"}:
        return "caution"
    if direction in spot.bad_directions:
        return "bad"
    return "neutral"


def shore_orientation(direction: str, spot: SpotConfig) -> str:
    """Operational coast-relative label derived from the spot direction contract."""
    category = direction_category(direction, spot)
    return {
        "preferred": "on-shore / side-on-shore",
        "acceptable": "side-shore",
        "caution": "side-shore con cautela",
        "bad": "off-shore / side-off-shore",
        "neutral": "orientamento non classificato",
    }[category]


def weather_risk_summary(hours: list[ForecastHour]) -> dict[str, object]:
    """Expose model hazard indicators without turning them into a safety guarantee."""
    usable = useful_hours(hours)
    probabilities = [
        hour.precipitation_probability_pct for hour in usable
        if hour.precipitation_probability_pct is not None
    ]
    precipitation = [
        hour.precipitation_mm for hour in usable if hour.precipitation_mm is not None
    ]
    cape = [hour.cape_jkg for hour in usable if hour.cape_jkg is not None]
    thunderstorm_hours = [
        hour.datetime.hour for hour in usable
        if hour.weather_code is not None and 95 <= hour.weather_code <= 99
    ]
    available = bool(probabilities or precipitation or cape or any(
        hour.weather_code is not None for hour in usable
    ))
    return {
        "status": "VALUTATO" if available else "NON_VALUTATO",
        "max_precipitation_probability_pct": max(probabilities) if probabilities else None,
        "precipitation_sum_mm": round(sum(precipitation), 2) if precipitation else None,
        "max_cape_jkg": max(cape) if cape else None,
        "thunderstorm_hours": thunderstorm_hours,
    }


def minimum_wind(profile: KiteProfile) -> float:
    return 9.0 if profile.board == "foil" else 13.0


def wind_component(speed: float, profile: KiteProfile) -> float:
    if profile.board == "foil":
        if speed < 9:
            return 8
        if speed < 12:
            return 30
        if speed <= 18:
            return 43
        if speed <= 23:
            return 32
        return 20

    if profile.kite_size_m2 == 10 and profile.weight_kg == 75:
        if speed < 13:
            return 8
        if speed < 14:
            return 18
        if speed < 16:
            return 27
        if speed <= 22:
            return 43
        if speed <= 25:
            return 36
        return 22

    if speed < 10:
        return 8
    if speed < 12:
        return 18
    if speed < 15:
        return 27
    if speed <= 20:
        return 42
    if speed <= 25:
        return 36
    return 22


def direction_component(category: str) -> float:
    return {
        "preferred": 25,
        "acceptable": 19,
        "caution": 11,
        "neutral": 14,
        "bad": 2,
    }[category]


def gust_component(delta: float) -> float:
    if delta <= 5:
        return 15
    if delta <= 8:
        return 11
    if delta <= 12:
        return 6
    return 2


def hour_acceptability(hour: ForecastHour, spot: SpotConfig, profile: KiteProfile) -> bool:
    if hour.wind_speed_knots < minimum_wind(profile):
        return False
    return direction_category(hour.wind_direction_cardinal, spot) != "bad"


def score_hour(hour: ForecastHour, spot: SpotConfig, profile: KiteProfile) -> HourScore:
    category = direction_category(hour.wind_direction_cardinal, spot)
    gust_delta = hour.wind_gusts_knots - hour.wind_speed_knots
    score = (
        wind_component(hour.wind_speed_knots, profile)
        + direction_component(category)
        + gust_component(gust_delta)
    )
    if THERMAL_START <= hour.datetime.hour <= THERMAL_END:
        score += 5
    return HourScore(hour=hour, acceptable=hour_acceptability(hour, spot, profile), score=score)


def useful_hours(hours: list[ForecastHour]) -> list[ForecastHour]:
    return [hour for hour in hours if USEFUL_START <= hour.datetime.hour <= USEFUL_END]


def find_best_window(hours: list[ForecastHour], spot: SpotConfig, profile: KiteProfile) -> BestWindow:
    scored = [score_hour(hour, spot, profile) for hour in useful_hours(hours)]
    runs: list[list[HourScore]] = []
    current: list[HourScore] = []
    previous_hour: int | None = None
    for item in scored:
        hour_value = item.hour.datetime.hour
        if item.acceptable and (previous_hour is None or hour_value == previous_hour + 1):
            current.append(item)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [item] if item.acceptable else []
        previous_hour = hour_value
    if len(current) >= 2:
        runs.append(current)
    if not runs:
        return BestWindow(available=False, hours=[])

    def run_key(run: list[HourScore]) -> tuple[float, int, float]:
        thermal_count = sum(THERMAL_START <= item.hour.datetime.hour <= THERMAL_END for item in run)
        return (thermal_count / len(run), len(run), mean(item.score for item in run))

    best = max(runs, key=run_key)
    start = best[0].hour.datetime.strftime("%H:%M")
    end_hour = best[-1].hour.datetime.hour + 1
    end = f"{end_hour:02d}:00"
    return BestWindow(available=True, start=start, end=end, hours=[item.hour for item in best])


def classify_stability(hours: list[ForecastHour], threshold: float) -> tuple[str, int]:
    if not hours:
        return "molto instabile", 0
    wind_range = max(hour.wind_speed_knots for hour in hours) - min(hour.wind_speed_knots for hour in hours)
    gust_range = max(hour.wind_gusts_knots for hour in hours) - min(hour.wind_gusts_knots for hour in hours)
    direction_range = direction_span_degrees([hour.wind_direction_degrees for hour in hours])
    holes = sum(hour.wind_speed_knots < threshold for hour in hours)
    penalty = 0
    penalty += 0 if wind_range <= 4 else 1 if wind_range <= 7 else 2
    penalty += 0 if gust_range <= 6 else 1 if gust_range <= 10 else 2
    penalty += 0 if direction_range <= 35 else 1 if direction_range <= 70 else 2
    penalty += min(2, holes)
    if penalty <= 1:
        return "buona", 10
    if penalty <= 3:
        return "discreta", 7
    if penalty <= 5:
        return "instabile", 4
    return "molto instabile", 1


def reliability(source_is_real: bool, hours: list[ForecastHour]) -> tuple[str, int]:
    if not source_is_real:
        return "bassa", 1
    if len(hours) >= 8:
        return "media", 4
    return "bassa", 2


def decision_from_score(score: int) -> str:
    if score < 40:
        return "LASCIA PERDERE"
    if score < 60:
        return "MARGINALE"
    if score < 80:
        return "VAI"
    return "VAI FORTE"


def thermal_watch(hours: list[ForecastHour], spot: SpotConfig, profile: KiteProfile) -> bool:
    """Detect days that are not a go, but deserve a same-day real check.

    This catches Gizzeria-style thermal misses: the model wind is just below the
    twintip threshold, direction is useful, and the afternoon has enough sun or
    gust signal that a local boost can make the session work.
    """
    threshold = minimum_wind(profile)
    candidates = [
        hour for hour in useful_hours(hours)
        if 14 <= hour.datetime.hour <= 16
        and direction_category(hour.wind_direction_cardinal, spot) in {"preferred", "acceptable"}
    ]
    if not candidates:
        return False
    peak = max(hour.wind_speed_knots for hour in candidates)
    peak_gust = max(hour.wind_gusts_knots for hour in candidates)
    sunny_hours = sum((hour.radiation or 0) >= 600 for hour in candidates)
    close_to_threshold = peak >= threshold - 2
    gust_hint = peak_gust >= threshold
    sun_hint = sunny_hours >= 2
    return close_to_threshold and (sun_hint or gust_hint)


def build_notes(
    profile: KiteProfile,
    wind_min: int | None,
    wind_max: int | None,
    max_gust_delta: float,
    decision: str,
) -> list[str]:
    notes: list[str] = []
    if profile.board == "foil":
        if wind_min is not None and wind_min >= 9:
            notes.append("Foil utilizzabile con controllo della reale intensità sullo spot.")
        if wind_max is not None and wind_max > 18:
            notes.append("Con foil vento sostenuto: valutare prudenza e misura più piccola.")
    else:
        if wind_max is None or wind_max < 13:
            notes.append("Vento medio insufficiente per Orbit 10 m² con twintip.")
            notes.append("Possibile solo con foil o kite più grande.")
        elif wind_min is not None and wind_min < 16:
            notes.append("Orbit 10 m² utilizzabile ma marginale se il vento resta sotto 16 kn.")
            notes.append("Porta tavola più grande se il vento resta sotto 16 kn.")
        else:
            notes.append("Orbit 10 m² nel range utilizzabile.")
    if max_gust_delta > 8:
        notes.append("Vento rafficato: controllare anemometro e osservazione sul posto prima di entrare.")
    if decision == THERMAL_CHECK_LABEL:
        notes.append("Termico possibile: non partire alla cieca, ma ricontrolla stazione, webcam o un contatto locale tra le 14 e le 16.")
        notes.append("Se il reale sale oltre soglia con direzione W/WNW, la giornata puo' diventare buona anche se il forecast live resta basso.")
    if decision == "LASCIA PERDERE":
        notes.append("Non fare strada apposta salvo conferma anemometro locale.")
    elif decision != THERMAL_CHECK_LABEL:
        notes.append("Controlla anemometro/webcam locali prima di entrare: termico e Venturi possono cambiare il vento reale.")
    return notes


def assess_day(
    *,
    spot: SpotConfig,
    date_label: str,
    target,
    hours: list[ForecastHour],
    source: str,
    source_is_real: bool,
    profile: KiteProfile,
    historical_rows: list[dict] | None = None,
) -> KiteAssessment:
    usable = useful_hours(hours)
    window = find_best_window(usable, spot, profile)
    basis = window.hours if window.available else usable
    threshold = minimum_wind(profile)
    stability_label, stability_score = classify_stability(basis, threshold)
    reliability_label, confidence_score = reliability(source_is_real, usable)

    if basis:
        avg_min = round(min(hour.wind_speed_knots for hour in basis))
        avg_max = round(max(hour.wind_speed_knots for hour in basis))
        gust_min = round(min(hour.wind_gusts_knots for hour in basis))
        gust_max = round(max(hour.wind_gusts_knots for hour in basis))
        dominant_direction = dominant_cardinal([hour.wind_direction_cardinal for hour in basis])
        avg_wind = mean(hour.wind_speed_knots for hour in basis)
        avg_gust_delta = mean(hour.wind_gusts_knots - hour.wind_speed_knots for hour in basis)
        max_gust_delta = max(hour.wind_gusts_knots - hour.wind_speed_knots for hour in basis)
        categories = [direction_category(hour.wind_direction_cardinal, spot) for hour in basis]
        worst_bad = "bad" in categories
        primary_category = max(set(categories), key=categories.count)
    else:
        avg_min = avg_max = gust_min = gust_max = None
        dominant_direction = None
        avg_wind = 0
        avg_gust_delta = 99
        max_gust_delta = 99
        worst_bad = True
        primary_category = "bad"

    wind_score = wind_component(avg_wind, profile)
    direction_score = direction_component(primary_category)
    gust_score = gust_component(avg_gust_delta)
    score = round(wind_score + direction_score + gust_score + stability_score + confidence_score)
    if not window.available:
        score = min(score, 39)
    if worst_bad:
        score = min(score, 49 if avg_wind >= threshold + 5 else 39)
    if max_gust_delta > 12 and not (avg_wind >= 18 and primary_category == "preferred"):
        score = min(score, 59)
    score = max(0, min(100, score))
    decision = decision_from_score(score)
    if decision == "LASCIA PERDERE" and not window.available and thermal_watch(usable, spot, profile):
        decision = THERMAL_CHECK_LABEL
        score = max(score, 45)
    notes = build_notes(profile, avg_min, avg_max, max_gust_delta, decision)
    from .thermal_onset import estimate_onset

    thermal_onset = estimate_onset(usable, spot, profile, historical_rows)
    if thermal_onset.get("onset_hour") is not None:
        notes.insert(
            0,
            "Termico atteso: "
            f"{thermal_onset['onset_label']} ±{thermal_onset['uncertainty_hours']:g}h "
            f"(confidenza: {thermal_onset['confidence']})",
        )

    return KiteAssessment(
        spot=spot.name,
        date_label=date_label,
        date=target,
        source=source,
        best_window=window,
        wind_avg_min_knots=avg_min,
        wind_avg_max_knots=avg_max,
        gust_min_knots=gust_min,
        gust_max_knots=gust_max,
        dominant_direction=dominant_direction,
        stability=stability_label,
        reliability=reliability_label,
        score=score,
        decision=decision,
        profile=profile,
        notes=notes,
        thermal_onset=thermal_onset,
        debug={
            "wind_score": wind_score,
            "direction_score": direction_score,
            "gust_score": gust_score,
            "stability_score": stability_score,
            "confidence_score": confidence_score,
            "max_gust_delta": max_gust_delta,
        },
    )
