import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "merged_solar_data.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

FEATURES = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
            "TIME_OF_DAY", "PLANT"]
TARGET = "AC_POWER"


def evaluate(y_true, y_pred):
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def main():
    df = pd.read_csv(DATA, parse_dates=["DATE_TIME"])
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    # Time-based split: first 80% of days train, last 20% of days test
    days = np.sort(df["DATE_TIME"].dt.date.unique())
    cut_day = days[int(len(days) * 0.8)]
    is_train = df["DATE_TIME"].dt.date < cut_day
    train, test = df[is_train], df[~is_train]
    print(f"Train: {len(train)} rows (until {cut_day}) | Test: {len(test)} rows")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42)
    else:
        print("xgboost not installed, skipping it (pip install xgboost)")

    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, None)  # power cannot be negative
        results[name] = evaluate(y_test, pred)
        fitted[name] = (model, pred)
        print(f"{name:18s}", results[name])

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model, best_pred = fitted[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump({"model": best_model, "features": FEATURES},
                MODELS / "solar_model.pkl", compress=3)

    # Plant-level accuracy: average of all inverters at each timestamp
    tmp = test[["DATE_TIME", "PLANT", TARGET]].copy()
    tmp["PREDICTED"] = best_pred
    g = tmp.groupby(["DATE_TIME", "PLANT"])[[TARGET, "PREDICTED"]].mean()
    plant_level = evaluate(g[TARGET], g["PREDICTED"])
    print("Plant-level (avg of inverters):", plant_level)

    # Feature importance (tree models only)
    importance = {}
    if hasattr(best_model, "feature_importances_"):
        importance = {f: round(float(v), 4)
                      for f, v in zip(FEATURES, best_model.feature_importances_)}

    with open(MODELS / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results,
                   "plant_level_best": plant_level,
                   "feature_importance": importance,
                   "train_rows": int(len(train)), "test_rows": int(len(test))},
                  f, indent=2)

    # Save test predictions so the dashboard can plot actual vs predicted
    out = test[["DATE_TIME", "PLANT", "SOURCE_KEY", TARGET]].copy()
    out["PREDICTED"] = best_pred
    out.to_csv(MODELS / "test_predictions.csv", index=False)
    print("Saved: models/solar_model.pkl, metrics.json, test_predictions.csv")


if __name__ == "__main__":
    main()