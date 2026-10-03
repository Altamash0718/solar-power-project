import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st

from src.ui import setup, header, kpi, load_data, load_json, load_health, N_INVERTERS
from src.impact import energy_kwh

setup("Home", "☀️")
header("☀️ Solar Power Generation Dashboard",
       "Prediction, forecasting and fault detection for two solar power plants")

df = load_data()
metrics = load_json("metrics.json")
daily, summary = load_health()

total_mwh = energy_kwh(df) / 1000
flagged = int((summary["STATUS"] != "Healthy").sum())
plant_r2 = metrics["plant_level_best"]["R2"]

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Energy generated", f"{total_mwh:,.0f} MWh", "Both plants, whole period")
kpi(c2, "Best model", metrics["best_model"], "Chosen on test-set R²")
kpi(c3, "Plant-level accuracy", f"R² {plant_r2:.2f}", "Average of all inverters")
kpi(c4, "Inverters needing attention", f"{flagged} / {len(summary)}", "Warning or fault status")

st.markdown("### What you can do here")
a, b, c = st.columns(3)
a.markdown('<div class="card"><h4>📊 Overview</h4><p>Explore power output, daily energy '
           'and how irradiation and temperature drive generation.</p></div>', unsafe_allow_html=True)
b.markdown('<div class="card"><h4>🔮 Predict</h4><p>Set irradiation, temperature and time to '
           'get the expected power, and see how accurate the model is.</p></div>', unsafe_allow_html=True)
c.markdown('<div class="card"><h4>🛠️ Fault Detection</h4><p>Find inverters that underperform '
           'their peers and estimate the energy they lose.</p></div>', unsafe_allow_html=True)
d, e, f = st.columns(3)
d.markdown('<div class="card"><h4>📈 Forecast</h4><p>Predict plant output 1 to 4 hours ahead, '
           'and tomorrow from a live weather forecast.</p></div>', unsafe_allow_html=True)
e.markdown('<div class="card"><h4>💰 Impact</h4><p>Convert energy into money earned and '
           'CO₂ avoided, with editable tariff and emission factor.</p></div>', unsafe_allow_html=True)
f.markdown('<div class="card"><h4>ℹ️ How it works</h4><p>Random Forest and boosting models '
           'trained on 15-minute data, evaluated on days the model never saw.</p></div>', unsafe_allow_html=True)

with st.expander("About the data and method"):
    st.markdown(
        f"""
- **Data:** Solar Power Generation Data (Kaggle, Ani Rudra Sen Nandi): 2 plants, {N_INVERTERS} inverters each,
  readings every 15 minutes from {df['DATE_TIME'].min():%d %b %Y} to {df['DATE_TIME'].max():%d %b %Y}.
- **Prediction target:** AC power (kW). Inputs: irradiation, ambient and module temperature, time of day, plant.
- **Validation:** time-based split, training on earlier days and testing on the final days.
- **Limitation:** only about 34 days in one season, so other seasons are not covered.
- **Weather forecast:** [Open-Meteo](https://open-meteo.com) (CC BY 4.0).
        """)
st.caption("Use the sidebar to open each page.")