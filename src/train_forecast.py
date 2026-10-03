"""Short-term forecasting: predict plant power 1, 2, 3 and 4 hours ahead.

Uses only information available "now": current and recent power, irradiation
and temperatures (lag features), plus the clock time of the target moment.
Every model is compared against a naive "persistence" baseline (future = now).
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.preprocess import build_dataset, build_plant_level

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

HORIZONS_H = [1, 2, 3, 4]          # hours ahead
LAGS = [1, 2, 4]                   # 15, 30, 60 minutes back (in 15-min steps)
FEATURES = (["PLANT", "POWER_NOW"] + [f"POWER_LAG{l}" for l in LAGS]
            + ["IRR_NOW"] + [f"IRR_LAG{l}" for l in LAGS]
            + ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "TARGET_TIME_OF_DAY"])


def build_features(grid, horizon_h):
    """Feature table for one horizon. Rows with any gap are dropped."""
    steps = horizon_h * 4
    frames = []
    for plant, part in grid.groupby("PLANT"):
        p = part.sort_values("DATE_TIME").reset_index(drop=True)
        f = pd.DataFrame({"DATE_TIME": p["DATE_TIME"], "PLANT": plant})
        f["POWER_NOW"] = p["AC_POWER"]
        f["IRR_NOW"] = p["IRRADIATION"]
        for l in LAGS:
            f[f"POWER_LAG{l}"] = p["AC_POWER"].shift(l)
            f[f"IRR_LAG{l}"] = p["IRRADIATION"].shift(l)
        f["AMBIENT_TEMPERATURE"] = p["AMBIENT_TEMPERATURE"]
        f["MODULE_TEMPERATURE"] = p["MODULE_TEMPERATURE"]
        tt = p["DATE_TIME"] + pd.Timedelta(hours=horizon_h)
        f["TARGET_TIME"] = tt
        f["TARGET_TIME_OF_DAY"] = tt.dt.hour + tt.dt.minute / 60
        f["TARGET"] = p["AC_POWER"].shift(-steps)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).dropna()


def metrics(y, pred):
    return {"R2": round(float(r2_score(y, pred)), 4),
            "MAE": round(float(mean_absolute_error(y, pred)), 1),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 1)}


def main():
    grid = build_plant_level(build_dataset())
    days = np.sort(grid["DATE_TIME"].dt.date.unique())
    cut = pd.Timestamp(days[int(len(days) * 0.8)])

    bundle, report = {}, {}
    for h in HORIZONS_H:
        feats = build_features(grid, h)
        train, test = feats[feats["DATE_TIME"] < cut], feats[feats["DATE_TIME"] >= cut]
        # Judge accuracy on daytime targets only, otherwise easy night zeros inflate scores
        day = test[(test["TARGET_TIME_OF_DAY"] >= 7) & (test["TARGET_TIME_OF_DAY"] <= 17)]

        candidates = {
            "Random Forest": RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
            "Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, random_state=42),
        }
        res = {"Persistence (baseline)": metrics(day["TARGET"], day["POWER_NOW"])}
        fitted = {}
        for name, m in candidates.items():
            m.fit(train[FEATURES], train["TARGET"])
            pred = np.clip(m.predict(day[FEATURES]), 0, None)
            res[name] = metrics(day["TARGET"], pred)
            fitted[name] = m

        best = min(fitted, key=lambda k: res[k]["RMSE"])
        base_rmse = res["Persistence (baseline)"]["RMSE"]
        skill = round(1 - res[best]["RMSE"] / base_rmse, 3)
        bundle[h] = {"model": fitted[best], "name": best}
        report[f"{h}h"] = {"best": best, "improvement_vs_baseline": skill, "results": res}
        print(f"\n+{h}h  best: {best}  (RMSE {res[best]['RMSE']} vs baseline {base_rmse},"
              f" {skill*100:.0f}% better)")
        for k, v in res.items():
            print(f"   {k:24s}", v)

    joblib.dump({"models": bundle, "features": FEATURES}, MODELS / "forecast_model.pkl",
                compress=3)
    with open(MODELS / "forecast_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved: models/forecast_model.pkl, forecast_metrics.json")


if __name__ == "__main__":
    main()