"""KiteGuru pubblico: previsione stateless di domani per Gizzeria.

Entrypoint pensato per Streamlit Community Cloud. Non legge DB, file utente,
task Windows o segreti locali: ogni visita ricava il forecast dalle fonti web.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kiteguru.config import get_spot
from kiteguru.correction import apply_correction
from kiteguru.models import KiteProfile
from kiteguru.providers.holfuy_chart import HolfuyChartProvider
from kiteguru.providers.open_meteo import OpenMeteoProvider
from kiteguru.providers.open_meteo_models import fetch_model_winds
from kiteguru.providers.regional import fetch_regional_features
from kiteguru.scoring import (
    assess_day,
    minimum_wind,
    shore_orientation,
    weather_risk_summary,
)
from kiteguru.thermal_model import train as train_thermal_model
from kiteguru.thermal_onset import estimate_onset


st.set_page_config(
    page_title="KiteGuru Gizzeria - domani",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1080px; padding-top: 1.4rem;}
    .kg-hero {padding: 1.2rem 1.4rem; border-radius: 18px;
      background: linear-gradient(135deg,#082f49,#0369a1); color:white; margin-bottom:1rem;}
    .kg-hero h1 {margin:0; font-size:2.05rem;} .kg-hero p {margin:.4rem 0 0; opacity:.9;}
    .kg-verdict {font-size:1.8rem; font-weight:800;}
    @media (max-width: 700px) {.kg-hero h1 {font-size:1.6rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

spot = get_spot("gizzeria")
today = datetime.now(ZoneInfo(spot.timezone)).date()
target = today + timedelta(days=1)
WEEKDAYS_IT = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")


@st.cache_data(ttl=900, show_spinner=False)
def load_forecast(target_iso: str):
    target_date = datetime.fromisoformat(target_iso).date()
    result = OpenMeteoProvider().fetch(spot, target_date)
    return result.model_dump(mode="python")


@st.cache_data(ttl=900, show_spinner=False)
def load_context(target_iso: str):
    target_date = datetime.fromisoformat(target_iso).date()
    return fetch_regional_features(spot, target_date)


@st.cache_data(ttl=900, show_spinner=False)
def load_models(target_iso: str):
    target_date = datetime.fromisoformat(target_iso).date()
    return fetch_model_winds(spot, target_date)


@st.cache_data(ttl=300, show_spinner=False)
def load_station():
    obs = HolfuyChartProvider().fetch_current(spot)
    return obs.model_dump(mode="python") if obs else None


@st.cache_data(ttl=300, show_spinner=False)
def load_measured_skill():
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "metrics" / "latest.json",
        Path(__file__).resolve().parent / "data" / "metrics" / "latest.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


with st.sidebar:
    st.header("Il tuo assetto")
    board = st.radio("Tavola", ["twintip", "foil"], horizontal=True)
    kite = st.number_input("Kite (m²)", 3.0, 19.0, 10.0, 0.5)
    weight = st.number_input("Peso (kg)", 40.0, 130.0, 75.0, 1.0)
    st.caption("La soglia operativa cambia con la tavola; la misura del kite e il peso affinano il verdetto.")

profile = KiteProfile(board=board, kite_size_m2=kite, weight_kg=weight)
threshold = minimum_wind(profile)

st.markdown(
    f"""
    <div class="kg-hero">
      <h1>🌬️ KiteGuru · Gizzeria</h1>
      <p>Previsione di domani · {WEEKDAYS_IT[target.weekday()]} {target:%d/%m/%Y} · aggiornata automaticamente</p>
    </div>
    """,
    unsafe_allow_html=True,
)

payload = load_forecast(target.isoformat())
if not payload.get("is_real") or not payload.get("hours"):
    st.error("Previsione momentaneamente non disponibile. Riprova tra qualche minuto.")
    if payload.get("error"):
        st.caption(payload["error"])
    st.stop()

from kiteguru.models import ForecastHour

raw_hours = [ForecastHour.model_validate(item) for item in payload["hours"]]
context = load_context(target.isoformat())

# In assenza del DB privato online si usa soltanto il prior fisico conservativo.
# Non viene presentato come modello gia' calibrato sui dati locali.
physical_prior = train_thermal_model(spot, [])
corrected_hours, max_boost = apply_correction(raw_hours, physical_prior, spot)
assessment = assess_day(
    spot=spot,
    date_label="domani",
    target=target,
    hours=corrected_hours,
    source=payload["source"],
    source_is_real=True,
    profile=profile,
    historical_rows=[],
)
raw_assessment = assess_day(
    spot=spot,
    date_label="domani",
    target=target,
    hours=raw_hours,
    source=payload["source"],
    source_is_real=True,
    profile=profile,
    historical_rows=[],
)
onset = estimate_onset(corrected_hours, spot, profile, [])
risk = weather_risk_summary(raw_hours)

# Il prior termico stateless e' uno scenario, non una misura calibrata sul DB.
# Da solo non puo' promuovere una giornata a VAI/VAI FORTE.
display_decision = raw_assessment.decision
if (
    raw_assessment.decision not in {"VAI", "VAI FORTE"}
    and assessment.decision in {"VAI", "VAI FORTE"}
):
    display_decision = "CONTROLLA 14-16"

decision_color = {
    "LASCIA PERDERE": "#b91c1c", "CONTROLLA 14-16": "#0e7490",
    "MARGINALE": "#a16207", "VAI": "#15803d", "VAI FORTE": "#1d4ed8",
}.get(display_decision, "#334155")

st.markdown(
    f'<div class="kg-verdict" style="color:{decision_color}">{display_decision}</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Il termico non calibrato viene mostrato come scenario, ma non può da solo generare un VAI. "
    "Controlla sempre la stazione prima di entrare in acqua."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Finestra grezza", (
    f"{raw_assessment.best_window.start}–{raw_assessment.best_window.end}"
    if raw_assessment.best_window.available else "Nessuna"
))
c2.metric("Open-Meteo", (
    f"{raw_assessment.wind_avg_min_knots}–{raw_assessment.wind_avg_max_knots} kn"
    if raw_assessment.wind_avg_min_knots is not None else "—"
))
c3.metric("Scenario non calibrato", (
    f"{assessment.wind_avg_min_knots}–{assessment.wind_avg_max_knots} kn"
    if assessment.wind_avg_min_knots is not None else "—"
))
c4.metric("Direzione", raw_assessment.dominant_direction or "—")
c5.metric("Ingresso scenario", onset.get("onset_label") or "Non previsto")
if raw_assessment.dominant_direction:
    st.caption(
        "Orientamento rispetto alla costa: "
        f"**{shore_orientation(raw_assessment.dominant_direction, spot)}**. "
        "La classificazione usa il contratto direzionale configurato per Gizzeria Lido."
    )

useful_raw = {h.datetime.hour: h for h in raw_hours if 10 <= h.datetime.hour <= 19}
useful_corrected = {h.datetime.hour: h for h in corrected_hours if 10 <= h.datetime.hour <= 19}
rows = []
for hour in sorted(useful_raw):
    raw = useful_raw[hour]
    corrected = useful_corrected[hour]
    reg = context.get(hour, {})
    rows.append({
        "Ora": f"{hour:02d}:00",
        "Open-Meteo": round(raw.wind_speed_knots, 1),
        "Scenario termico": round(corrected.wind_speed_knots, 1),
        "Raffica": round(raw.wind_gusts_knots, 1),
        "Direzione": raw.wind_direction_cardinal,
        "Radiazione": raw.radiation,
        "ΔT entroterra-SST": reg.get("dT_land_sst"),
        "Strato limite": raw.boundary_layer_height_m,
    })
df = pd.DataFrame(rows)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Ora"], y=df["Open-Meteo"], name="Open-Meteo",
                         mode="lines+markers", line=dict(color="#2563eb", width=2)))
fig.add_trace(go.Scatter(x=df["Ora"], y=df["Scenario termico"], name="Scenario termico",
                         mode="lines+markers", line=dict(color="#059669", width=3)))
fig.add_trace(go.Scatter(x=df["Ora"], y=df["Raffica"], name="Raffica grezza",
                         mode="lines", line=dict(color="#f59e0b", dash="dot")))
fig.add_hline(y=threshold, line_dash="dash", line_color="#64748b",
              annotation_text=f"soglia {threshold:.0f} kn")
fig.update_layout(height=410, yaxis_title="nodi", xaxis_title="ora locale",
                  legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=45, b=10))
st.plotly_chart(fig, width="stretch")

st.subheader("Sicurezza meteo")
if risk["status"] == "NON_VALUTATO":
    st.warning("Precipitazioni e indicatori convettivi non disponibili: rischio meteo NON_VALUTATO.")
else:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Prob. precipitazione max",
        f"{risk['max_precipitation_probability_pct']:.0f}%"
        if risk["max_precipitation_probability_pct"] is not None else "n/d",
    )
    r2.metric(
        "Precipitazione 10–19",
        f"{risk['precipitation_sum_mm']:.1f} mm"
        if risk["precipitation_sum_mm"] is not None else "n/d",
    )
    r3.metric(
        "CAPE max",
        f"{risk['max_cape_jkg']:.0f} J/kg"
        if risk["max_cape_jkg"] is not None else "n/d",
    )
    thunderstorm_hours = risk["thunderstorm_hours"]
    r4.metric(
        "Temporale (codice WMO)",
        ", ".join(f"{hour:02d}:00" for hour in thunderstorm_hours)
        if thunderstorm_hours else "non indicato",
    )
    if thunderstorm_hours:
        st.error("Il modello indica temporali nelle ore utili: non entrare in acqua.")
    st.caption(
        "Indicatori del forecast, non nowcast: controlla radar, fulminazioni e avvisi locali. "
        "La probabilità di precipitazione Open-Meteo deriva da un ensemble a risoluzione più grossolana."
    )

tab_hours, tab_models, tab_live, tab_skill, tab_method = st.tabs(
    ["Ore", "Confronto modelli", "Stazione ora", "Affidabilità", "Come leggerla"]
)
with tab_hours:
    display = df[["Ora", "Open-Meteo", "Scenario termico", "Raffica", "Direzione"]].copy()
    st.dataframe(display, hide_index=True, width="stretch")

with tab_models:
    models = load_models(target.isoformat())
    model_fig = go.Figure()
    for label, values in models.items():
        xs = [f"{h:02d}:00" for h in sorted(values) if 10 <= h <= 19]
        ys = [values[h] for h in sorted(values) if 10 <= h <= 19]
        if xs:
            model_fig.add_trace(go.Scatter(x=xs, y=ys, name=label, mode="lines+markers"))
    if model_fig.data:
        model_fig.update_layout(height=390, yaxis_title="nodi", xaxis_title="ora locale",
                                legend=dict(orientation="h", y=1.2), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(model_fig, width="stretch")
        st.caption("La dispersione fra modelli è un indicatore pratico dell'incertezza meteorologica.")
    else:
        st.info("Confronto modelli temporaneamente non disponibile.")

with tab_live:
    station = load_station()
    if station:
        observed_at = station["datetime"]
        if isinstance(observed_at, str):
            observed_at = datetime.fromisoformat(observed_at)
        a, b, c = st.columns(3)
        a.metric("Vento reale", f"{station['wind_speed_knots']:.1f} kn")
        b.metric("Raffica", f"{station['wind_gusts_knots']:.1f} kn")
        c.metric("Direzione", station["wind_direction_cardinal"])
        st.caption(f"Holfuy 1178 · ultima lettura {observed_at:%d/%m %H:%M}")
    else:
        st.info("Centralina Holfuy momentaneamente non raggiungibile.")

with tab_skill:
    measured = load_measured_skill()
    if measured and measured.get("raw_forecast", {}).get("hours"):
        skill = measured["raw_forecast"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Giornate verificate", skill["days"])
        s2.metric("Giornate indovinate", skill["day_hits"])
        s3.metric("Falsi negativi orari", skill["false_negative"])
        s4.metric("Errore medio", f"{skill['mae_kn']:.1f} kn")
        st.caption(
            "Metriche prospettiche del forecast grezzo, calcolate solo su snapshot "
            "congelati prima dell'evento e successivamente confrontati con Holfuy."
        )
    else:
        st.info(
            "La raccolta cloud è attiva. Le metriche appariranno quando il primo "
            "snapshot di domani avrà il corrispondente dato reale."
        )

with tab_method:
    st.markdown(
        f"""
        - **Open-Meteo** è il forecast atmosferico grezzo.
        - **Scenario termico** applica il prior fisico locale solo con direzioni compatibili.
        - Lo scenario non calibrato non può da solo promuovere il verdetto a **VAI**.
        - Il massimo rinforzo applicato domani è **{max_boost:.1f} kn**.
        - La soglia del tuo assetto è **{threshold:.0f} kn**.
        - La versione pubblica non espone il database o i file del computer locale.

        La previsione è un supporto decisionale, non una garanzia di sicurezza.
        """
    )

st.divider()
st.caption(
    f"Fonte forecast: {payload['source']} · fuso orario Europe/Rome · cache 15 minuti · "
    f"pagina aggiornata {datetime.now(ZoneInfo(spot.timezone)):%d/%m/%Y %H:%M} · "
    "età del ciclo NWP non esposta dall'API best-match"
)
