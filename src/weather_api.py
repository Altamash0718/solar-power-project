"""Day-ahead prediction from a live weather forecast (Open-Meteo).

Open-Meteo is free for non-commercial use and needs no API key. Credit it
(CC BY 4.0) in your README and report: https://open-meteo.com
"""
import joblib
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "solar_model.pkl"
API_URL = "https://api.open-meteo.com/v1/forecast"

# A few cities to choose from. The plants' real coordinates are not in the dataset.
LOCATIONS = {
    "Jodhpur": (26.29, 73.02),
    "Ahmedabad": (23.02, 72.57),
    "Pune": (18.52, 73.86),
    "Hyderabad": (17.38, 78.48),
    "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
}


def parse_forecast(payload):
    """Open-Meteo JSON -> DataFrame(TIME, TEMP_C, RADIATION_WM2)."""
    h = payload["hourly"]
    return pd.DataFrame({
        "TIME": pd.to_datetime(h["time"]),
        "TEMP_C": h["temperature_2m"],
        "RADIATION_WM2": h["shortwave_radiation"],
    })


def fetch_forecast(lat, lon, days=2):
    """Returns the hourly forecast, or None if the API cannot be reached."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,shortwave_radiation",
        "timezone": "auto", "forecast_days": days,
    }
    try:
        r = requests.get(API_URL, params=params, timeout=15)
        r.raise_for_status()
        return parse_forecast(r.json())
    except (requests.RequestException, KeyError, ValueError):
        return None


def fit_module_temp_model(df):
    """The forecast has no module temperature, so learn it from your data:
    module temp ~ ambient temp + irradiation."""
    X, y = df[["AMBIENT_TEMPERATURE", "IRRADIATION"]], df["MODULE_TEMPERATURE"]
    model = LinearRegression().fit(X, y)
    return model, float(model.score(X, y))


def predict_day(forecast, plant, module_model, n_inverters=22, irr_scale=1.0):
    """Predict hourly plant power (kW) from a forecast table.

    irr_scale lets you calibrate: the API gives radiation on a flat surface,
    the plant sensor measures on the tilted panels.
    """
    bundle = joblib.load(MODEL_PATH)
    f = forecast.copy()
    f["IRRADIATION"] = f["RADIATION_WM2"] / 1000 * irr_scale      # W/m2 -> kW/m2
    f["AMBIENT_TEMPERATURE"] = f["TEMP_C"]
    f["MODULE_TEMPERATURE"] = module_model.predict(
        f[["AMBIENT_TEMPERATURE", "IRRADIATION"]])
    # Radiation values are the mean of the preceding hour -> centre of that hour
    f["TIME_OF_DAY"] = f["TIME"].dt.hour - 0.5
    f["PLANT"] = plant
    per_inverter = np.clip(bundle["model"].predict(f[bundle["features"]]), 0, None)
    f["POWER_KW"] = per_inverter * n_inverters
    f["ENERGY_KWH"] = f["POWER_KW"] * 1.0     # one-hour steps
    return f[["TIME", "TEMP_C", "RADIATION_WM2", "POWER_KW", "ENERGY_KWH"]]


if __name__ == "__main__":
    from src.preprocess import build_dataset
    df = build_dataset()
    module_model, r2 = fit_module_temp_model(df)
    print(f"Module temperature model R2: {r2:.3f}")

    # Self-test without internet: pretend the real weather of one day was a forecast
    day = df[df["DATE"] == pd.Timestamp("2020-06-14").date()]
    for plant in (1, 2):
        p = day[day["PLANT"] == plant]
        hourly = p.groupby(p["DATE_TIME"].dt.floor("h")).agg(
            TEMP_C=("AMBIENT_TEMPERATURE", "mean"), IRR=("IRRADIATION", "mean"),
            AC=("AC_POWER", "mean")).reset_index().rename(columns={"DATE_TIME": "TIME"})
        fc = pd.DataFrame({"TIME": hourly["TIME"] + pd.Timedelta(hours=1),
                           "TEMP_C": hourly["TEMP_C"],
                           "RADIATION_WM2": hourly["IRR"] * 1000})
        out = predict_day(fc, plant, module_model)
        actual = hourly["AC"].sum() * 22
        print(f"Plant {plant} 2020-06-14: predicted {out['ENERGY_KWH'].sum():,.0f} kWh"
              f" vs actual {actual:,.0f} kWh")