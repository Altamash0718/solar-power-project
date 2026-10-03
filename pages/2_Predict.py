import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import (setup, header, kpi, style_fig, load_solar_model, load_module_model,
                    load_json, load_test_predictions, N_INVERTERS)

setup("Predict", "🔮")
header("Power Prediction", "Expected AC power from weather conditions and time of day")

bundle, module_model = load_solar_model(), load_module_model()


def predict_power(irr, amb, mod, hour, plant):
    """Expected AC power of ONE inverter in kW."""
    X = pd.DataFrame([{"IRRADIATION": irr, "AMBIENT_TEMPERATURE": amb,
                       "MODULE_TEMPERATURE": mod, "TIME_OF_DAY": hour, "PLANT": plant}])
    return max(float(bundle["model"].predict(X[bundle["features"]])[0]), 0.0)


tab_try, tab_acc = st.tabs(["Try it yourself", "Model accuracy"])

# ---------------- tab 1: what-if ----------------
with tab_try:
    left, right = st.columns([1, 2])
    with left:
        plant = st.selectbox("Plant", [1, 2], format_func=lambda p: f"Plant {p}")
        irr = st.slider("Irradiation (kW/m²)", 0.0, 1.2, 0.8, 0.01)
        amb = st.slider("Ambient temperature (°C)", 15.0, 45.0, 30.0, 0.5)
        auto = st.checkbox("Estimate module temperature automatically", value=True)
        auto_mod = float(module_model.predict(
            pd.DataFrame({"AMBIENT_TEMPERATURE": [amb], "IRRADIATION": [irr]}))[0])
        if auto:
            mod = auto_mod
            st.caption(f"Estimated module temperature: {mod:.1f} °C")
        else:
            mod = st.slider("Module temperature (°C)", 15.0, 75.0,
                            float(np.clip(auto_mod, 15.0, 75.0)), 0.5)
        hour = st.slider("Time of day (hour)", 0.0, 23.75, 12.0, 0.25)

    per_inv = predict_power(irr, amb, mod, hour, plant)
    with right:
        k1, k2, k3 = st.columns(3)
        kpi(k1, "Per inverter", f"{per_inv:,.0f} kW", "Expected AC power")
        kpi(k2, f"Whole plant ({N_INVERTERS} inverters)", f"{per_inv * N_INVERTERS / 1000:,.2f} MW",
            "If all inverters are healthy")
        kpi(k3, "Module temperature", f"{mod:.1f} °C", "Used by the model")

        sweep = pd.DataFrame({"IRRADIATION": np.linspace(0, 1.2, 61)})
        sweep["AMBIENT_TEMPERATURE"] = amb
        sweep["MODULE_TEMPERATURE"] = (module_model.predict(sweep[["AMBIENT_TEMPERATURE", "IRRADIATION"]])
                                       if auto else mod)
        sweep["TIME_OF_DAY"] = hour
        sweep["PLANT"] = plant
        sweep["Plant power (MW)"] = np.clip(
            bundle["model"].predict(sweep[bundle["features"]]), 0, None) * N_INVERTERS / 1000
        fig = px.line(sweep, x="IRRADIATION", y="Plant power (MW)",
                      labels={"IRRADIATION": "Irradiation (kW/m²)"},
                      title="How power responds to sunlight (other inputs fixed)")
        fig.update_traces(line_color="#FFB300")
        fig.add_scatter(x=[irr], y=[per_inv * N_INVERTERS / 1000], mode="markers",
                        marker=dict(size=13, color="#FFFFFF"), name="Your input")
        st.plotly_chart(style_fig(fig, 340))

# ---------------- tab 2: accuracy ----------------
with tab_acc:
    m = load_json("metrics.json")
    k1, k2, k3 = st.columns(3)
    best = m["results"][m["best_model"]]
    kpi(k1, "Best model", m["best_model"], "Highest R² on unseen days")
    kpi(k2, "Per inverter", f"R² {best['R2']:.2f}", f"MAE {best['MAE']:.0f} kW")
    kpi(k3, "Plant level", f"R² {m['plant_level_best']['R2']:.2f}",
        f"MAE {m['plant_level_best']['MAE']:.0f} kW (avg of inverters)")

    table = pd.DataFrame([{"Model": k, **v} for k, v in m["results"].items()]).set_index("Model")
    st.markdown("##### Model comparison (test days only)")
    st.dataframe(table)

    left, right = st.columns([1, 2])
    if m.get("feature_importance"):
        fi = (pd.Series(m["feature_importance"]).sort_values().rename("Importance")
              .rename_axis("Feature").reset_index())
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     title="What the model relies on")
        fig.update_traces(marker_color="#FFB300")
        left.plotly_chart(style_fig(fig, 340))

    preds = load_test_predictions()
    p = right.selectbox("Plant", [1, 2], format_func=lambda x: f"Plant {x}", key="acc_plant")
    avg = (preds[preds["PLANT"] == p].groupby("DATE_TIME")[["AC_POWER", "PREDICTED"]].mean()
           .rename(columns={"AC_POWER": "Actual", "PREDICTED": "Predicted"}).reset_index())
    long = avg.melt("DATE_TIME", var_name="Series", value_name="kW")
    fig = px.line(long, x="DATE_TIME", y="kW", color="Series",
                  color_discrete_map={"Actual": "#FFB300", "Predicted": "#29B6F6"},
                  labels={"kW": "Avg power per inverter (kW)", "DATE_TIME": ""},
                  title="Actual vs predicted on unseen test days")
    right.plotly_chart(style_fig(fig, 340))