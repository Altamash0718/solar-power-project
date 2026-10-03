"""Inverter fault detection.

Two checks per inverter, using daytime rows only (irradiation >= DAY_IRR):
  1. Peer comparison: actual power vs the median of all inverters in the same
     plant at the same timestamp (robust, needs no model).
  2. Model comparison: actual power vs the ML model's expected power.
Zero output while the sun is up (and peers are producing) is also counted.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "solar_model.pkl"

DAY_IRR = 0.2         # kW/m2, below this the sun is too weak to judge
MIN_PEER_KW = 50      # ignore moments when even the median inverter is ~off
INTERVAL_H = 0.25     # 15 minutes in hours

# Thresholds for the status label (tune these and explain them in your report)
FAULT_RATIO, WARN_RATIO = 0.70, 0.90
FAULT_ZERO, WARN_ZERO = 0.30, 0.10


def classify(ratio, zero_share):
    if ratio < FAULT_RATIO or zero_share > FAULT_ZERO:
        return "Fault"
    if ratio < WARN_RATIO or zero_share > WARN_ZERO:
        return "Warning"
    return "Healthy"


def add_expected_power(df):
    """Adds EXPECTED_AC from the trained model (skipped if no model file)."""
    df = df.copy()
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        pred = bundle["model"].predict(df[bundle["features"]])
        df["EXPECTED_AC"] = np.clip(pred, 0, None)
    else:
        df["EXPECTED_AC"] = np.nan
    return df


def inverter_daily_health(df):
    """One row per inverter per day with ratios, zero share, loss and status."""
    d = df[df["IRRADIATION"] >= DAY_IRR].copy()
    d["PEER_MEDIAN"] = d.groupby(["PLANT", "DATE_TIME"])["AC_POWER"].transform("median")
    d = d[d["PEER_MEDIAN"] > MIN_PEER_KW].copy()
    d["IS_ZERO"] = d["AC_POWER"] == 0
    d["SHORTFALL"] = (d["PEER_MEDIAN"] - d["AC_POWER"]).clip(lower=0)

    daily = d.groupby(["PLANT", "SOURCE_KEY", "DATE"]).agg(
        ACTUAL_KWH=("AC_POWER", lambda s: s.sum() * INTERVAL_H),
        PEER_KWH=("PEER_MEDIAN", lambda s: s.sum() * INTERVAL_H),
        EXPECTED_KWH=("EXPECTED_AC", lambda s: s.sum() * INTERVAL_H),
        ZERO_SHARE=("IS_ZERO", "mean"),
        SHORTFALL_KWH=("SHORTFALL", lambda s: s.sum() * INTERVAL_H),
    ).reset_index()

    daily["PEER_RATIO"] = daily["ACTUAL_KWH"] / daily["PEER_KWH"]
    daily["MODEL_RATIO"] = daily["ACTUAL_KWH"] / daily["EXPECTED_KWH"]
    daily["STATUS"] = [classify(r, z) for r, z in
                       zip(daily["PEER_RATIO"], daily["ZERO_SHARE"])]
    # Energy lost is only counted for inverters that are not healthy that day
    daily["LOSS_KWH"] = np.where(daily["STATUS"] == "Healthy", 0.0,
                                 daily["SHORTFALL_KWH"])
    return daily.drop(columns=["EXPECTED_KWH", "SHORTFALL_KWH"])


def inverter_summary(daily):
    """One row per inverter over the whole period."""
    g = daily.groupby(["PLANT", "SOURCE_KEY"])
    s = g.agg(ACTUAL_KWH=("ACTUAL_KWH", "sum"), PEER_KWH=("PEER_KWH", "sum"),
              ZERO_SHARE=("ZERO_SHARE", "mean"), LOSS_KWH=("LOSS_KWH", "sum"),
              DAYS=("DATE", "nunique"),
              FAULT_DAYS=("STATUS", lambda x: (x == "Fault").sum()),
              WARNING_DAYS=("STATUS", lambda x: (x == "Warning").sum())
              ).reset_index()
    s["PEER_RATIO"] = s["ACTUAL_KWH"] / s["PEER_KWH"]
    s["STATUS"] = [classify(r, z) for r, z in zip(s["PEER_RATIO"], s["ZERO_SHARE"])]
    s["HEALTH_SCORE"] = (s["PEER_RATIO"] * 100).clip(0, 100).round(1)
    return s.sort_values(["PLANT", "PEER_RATIO"]).reset_index(drop=True)


if __name__ == "__main__":
    from src.preprocess import build_dataset
    df = add_expected_power(build_dataset())
    daily = inverter_daily_health(df)
    summary = inverter_summary(daily)
    print(summary.groupby(["PLANT", "STATUS"]).size().unstack(fill_value=0))
    print(summary.head(8)[["PLANT", "SOURCE_KEY", "PEER_RATIO", "ZERO_SHARE",
                           "FAULT_DAYS", "LOSS_KWH", "STATUS"]].round(3).to_string())
    print("Total estimated loss (kWh):", round(daily["LOSS_KWH"].sum()))