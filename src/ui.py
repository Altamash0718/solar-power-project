"""Shared helpers for the Streamlit dashboard (styling, cached loaders)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import build_dataset, build_plant_level
from src.fault_detection import add_expected_power, inverter_daily_health, inverter_summary
from src.weather_api import fit_module_temp_model, fetch_forecast
from src.train_forecast import build_features

ROOT = Path(__file__).resolve().parent.parent
N_INVERTERS = 22
INTERVAL_H = 0.25
PLANT_COLORS = {"Plant 1": "#FFB300", "Plant 2": "#29B6F6"}
STATUS_COLORS = {"Healthy": "#2ECC71", "Warning": "#F1C40F", "Fault": "#E74C3C"}


# ---------- look and feel ----------
def setup(title, icon="☀️"):
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} | Solar Power", page_icon=icon, layout="wide")
    css = (ROOT / "assets" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def header(title, subtitle=""):
    st.markdown(f'<div class="page-title">{title}</div>'
                f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(container, label, value, sub=""):
    container.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.12, x=0, title_text=""))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ---------- cached data and models ----------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    return build_dataset()


@st.cache_data(show_spinner=False)
def load_grid():
    return build_plant_level(load_data())


@st.cache_data(show_spinner="Analysing inverters...")
def load_health():
    df = add_expected_power(load_data())
    daily = inverter_daily_health(df)
    return daily, inverter_summary(daily)


@st.cache_resource(show_spinner=False)
def load_solar_model():
    return joblib.load(ROOT / "models" / "solar_model.pkl")


@st.cache_resource(show_spinner=False)
def load_forecast_bundle():
    return joblib.load(ROOT / "models" / "forecast_model.pkl")


@st.cache_resource(show_spinner=False)
def load_module_model():
    return fit_module_temp_model(load_data())[0]


@st.cache_data(show_spinner=False)
def load_json(name):
    with open(ROOT / "models" / name) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_test_predictions():
    return pd.read_csv(ROOT / "models" / "test_predictions.csv", parse_dates=["DATE_TIME"])


@st.cache_data(show_spinner=False)
def forecast_features(horizon_h):
    return build_features(load_grid(), horizon_h)


@st.cache_data(show_spinner=False)
def forecast_cut():
    """First day of the test period (same rule as train_forecast.py)."""
    days = np.sort(load_grid()["DATE_TIME"].dt.date.unique())
    return pd.Timestamp(days[int(len(days) * 0.8)])


@st.cache_data(ttl=1800, show_spinner="Fetching weather forecast...")
def cached_forecast(lat, lon):
    df = fetch_forecast(lat, lon)
    if df is None:                 # raising means failures are never cached
        raise RuntimeError("weather service unreachable")
    return df


# ---------- friendly inverter names (P1-I01 ...) ----------
@st.cache_data(show_spinner=False)
def inverter_labels():
    df = load_data()
    out = {}
    for plant, part in df.groupby("PLANT"):
        for i, key in enumerate(sorted(part["SOURCE_KEY"].unique()), 1):
            out[(int(plant), key)] = f"P{int(plant)}-I{i:02d}"
    return out


def add_labels(frame):
    labels = inverter_labels()
    frame = frame.copy()
    frame["INVERTER"] = [labels[(int(p), k)] for p, k in zip(frame["PLANT"], frame["SOURCE_KEY"])]
    return frame