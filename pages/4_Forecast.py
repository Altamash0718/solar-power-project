import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import (setup, header, kpi, style_fig, load_grid, load_forecast_bundle,
                    load_module_model, load_json, forecast_features, forecast_cut,
                    cached_forecast, N_INVERTERS)
from src.weather_api import predict_day, LOCATIONS

setup("Forecast", "📈")
header("Power Forecast", "What will the plants produce in the next hours, and tomorrow?")

tab_now, tab_tom = st.tabs(["Next 1 to 4 hours", "Tomorrow from the weather forecast"])

# ---------------- tab 1: short-term forecast ----------------
with tab_now:
    bundle, grid = load_forecast_bundle(), load_grid()
    dates = sorted(grid["DATE_TIME"].dt.date.unique())

    c1, c2, c3 = st.columns(3)
    plant = c1.selectbox("Plant", [1, 2], format_func=lambda p: f"Plant {p}", key="f_plant")
    day = c2.selectbox("Date", dates, index=max(len(dates) - 3, 0), format_func=str)
    hour = c3.slider("Forecast issued at (hour)", 6.0, 16.0, 10.0, 0.25)

    ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    if ts < forecast_cut():
        st.info("This moment was part of the training period, so accuracy here looks better than "
                f"it really is. For an honest view pick a date from {forecast_cut():%d %b} onwards.")

    rows = {}
    for h in bundle["models"]:
        f = forecast_features(h)
        r = f[(f["PLANT"] == plant) & (f["DATE_TIME"] == ts)]
        if r.empty:
            rows = None
            break
        rows[h] = r.iloc[0]

    if rows is None:
        st.warning("The data has a gap around that moment. Try another time.")
    else:
        now_kw = rows[1]["POWER_NOW"]
        records = [{"Time": ts, "Forecast": now_kw, "Actual": now_kw, "Persistence": now_kw}]
        table = []
        for h, r in rows.items():
            pred = max(float(bundle["models"][h]["model"].predict(
                pd.DataFrame([r[bundle["features"]]]).astype(float))[0]), 0.0)
            records.append({"Time": r["TARGET_TIME"], "Forecast": pred,
                            "Actual": r["TARGET"], "Persistence": r["POWER_NOW"]})
            table.append({"Horizon": f"+{h} h", "Forecast (MW)": round(pred / 1000, 2),
                          "Actual (MW)": round(r["TARGET"] / 1000, 2),
                          "Error (MW)": round((pred - r["TARGET"]) / 1000, 2)})

        k1, k2, k3 = st.columns(3)
        kpi(k1, "Power now", f"{now_kw / 1000:,.2f} MW", f"{ts:%d %b, %H:%M}")
        kpi(k2, "Forecast in 1 hour", f"{table[0]['Forecast (MW)']:,.2f} MW",
            f"actual {table[0]['Actual (MW)']:,.2f} MW")
        kpi(k3, "Forecast in 4 hours", f"{table[-1]['Forecast (MW)']:,.2f} MW",
            f"actual {table[-1]['Actual (MW)']:,.2f} MW")

        plot = pd.DataFrame(records)
        for col in ("Forecast", "Actual", "Persistence"):
            plot[col] = plot[col] / 1000
        long = plot.melt("Time", var_name="Series", value_name="MW")
        fig = px.line(long, x="Time", y="MW", color="Series", markers=True,
                      color_discrete_map={"Forecast": "#29B6F6", "Actual": "#FFB300",
                                          "Persistence": "#7F8A9A"},
                      labels={"MW": "Plant power (MW)", "Time": ""},
                      title="Forecast vs what really happened (Persistence = 'same as now')")
        st.plotly_chart(style_fig(fig, 380))
        st.dataframe(pd.DataFrame(table), hide_index=True)

    st.markdown("##### How good is it? (test days, daytime only)")
    fm = load_json("forecast_metrics.json")
    summary = []
    for k, v in fm.items():
        base, best = v["results"]["Persistence (baseline)"], v["results"][v["best"]]
        summary.append({"Horizon": f"+{k}", "Best model": v["best"],
                        "Baseline RMSE (kW)": base["RMSE"], "Model RMSE (kW)": best["RMSE"],
                        "Improvement": f"{v['improvement_vs_baseline'] * 100:.0f}%",
                        "R² (model)": best["R2"]})
    st.dataframe(pd.DataFrame(summary), hide_index=True)
    st.caption("Baseline = assume power stays the same as now. Beating it shows the model learned something real. "
               "The test period is only 7 days, so treat these numbers as indicative.")

# ---------------- tab 2: tomorrow from weather forecast ----------------
with tab_tom:
    module_model = load_module_model()
    c1, c2, c3 = st.columns(3)
    city = c1.selectbox("Location", list(LOCATIONS))
    plant_t = c2.selectbox("Plant model", [1, 2], format_func=lambda p: f"Plant {p}", key="t_plant")
    scale = c3.slider("Irradiation calibration", 0.6, 1.4, 1.0, 0.05,
                      help="Weather services give sunlight on a flat surface; panels are tilted. "
                           "Adjust if predictions look too low or high.")
    try:
        fc = cached_forecast(*LOCATIONS[city])
    except Exception:
        st.error("Could not reach the Open-Meteo weather service. Check your internet connection and try again.")
        st.stop()

    out = predict_day(fc, plant_t, module_model, N_INVERTERS, scale)
    days = sorted(out["TIME"].dt.date.unique())
    sel = st.radio("Day", days, format_func=lambda d: d.strftime("%a %d %b"), horizontal=True)
    o = out[out["TIME"].dt.date == sel].copy()
    o["MW"] = o["POWER_KW"] / 1000

    peak_row = o.loc[o["MW"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Expected energy", f"{o['ENERGY_KWH'].sum() / 1000:,.0f} MWh", f"{city}, {sel:%d %b}")
    kpi(k2, "Peak output", f"{peak_row['MW']:,.1f} MW", f"around {peak_row['TIME']:%H:%M}")
    kpi(k3, "Max sunlight", f"{o['RADIATION_WM2'].max():,.0f} W/m²", "Forecast radiation")
    kpi(k4, "Avg temperature", f"{o['TEMP_C'].mean():.1f} °C", "Forecast, whole day")

    fig = px.area(o, x="TIME", y="MW", labels={"MW": "Plant power (MW)", "TIME": ""},
                  title="Predicted hourly plant output")
    fig.update_traces(line_color="#FFB300")
    st.plotly_chart(style_fig(fig, 340))

    fig = px.line(o, x="TIME", y="RADIATION_WM2",
                  labels={"RADIATION_WM2": "Sunlight (W/m²)", "TIME": ""},
                  title="Forecast sunlight")
    fig.update_traces(line_color="#29B6F6")
    st.plotly_chart(style_fig(fig, 260))

    st.caption("Weather data: Open-Meteo.com (CC BY 4.0). The plants' real locations are not in the dataset, "
               "so the city is only a demonstration. The model was trained on May to June data from two plants, "
               "so predictions for other seasons or sites are an extrapolation.")