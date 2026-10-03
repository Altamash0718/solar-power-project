# ☀️ Solar Power Generation: Prediction, Forecasting and Fault Detection

A machine learning project that predicts and forecasts the power output of two solar power plants,
detects underperforming inverters, estimates financial and environmental impact, and presents
everything in an interactive dark-themed dashboard.

**Live demo:** [https://solar-power-project-generation.streamlit.app/](https://solar-power-project-generation.streamlit.app/)
&nbsp;|&nbsp; Built with Python, scikit-learn, XGBoost, Streamlit and Plotly

![Dashboard overview](assets/screenshots/overview.png)

---

## Features

| Feature | What it does |
|---|---|
| **Power prediction** | Predicts AC power from irradiation, ambient and module temperature, time of day and plant |
| **Short-term forecast** | Predicts plant output 1, 2, 3 and 4 hours ahead from recent power and weather (lag features) |
| **Day-ahead forecast** | Fetches tomorrow's weather from Open-Meteo and predicts hourly plant output |
| **Fault detection** | Compares every inverter with its peers and with the model, flags weak inverters and estimates lost energy |
| **Impact analysis** | Converts energy to revenue and CO₂ avoided, with an editable tariff and emission factor |
| **Dashboard** | 5 pages: Overview, Predict, Fault Detection, Forecast, Impact |

## Dataset

[Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
(Kaggle, by Ani Rudra Sen Nandi). See the dataset page for its licence.

| File | Content |
|---|---|
| `Plant_1_Generation_Data.csv` / `Plant_2_Generation_Data.csv` | DC and AC power per inverter, every 15 minutes |
| `Plant_1_Weather_Sensor_Data.csv` / `Plant_2_Weather_Sensor_Data.csv` | Irradiation, ambient and module temperature |

Two plants, 22 inverters each, 15 May to 17 June 2020 (about 34 days).

**Data notes**
- Plant 1 generation timestamps use `DD-MM-YYYY`; the other files use ISO format. Parsing handles both.
- Plant 1 `DC_POWER` is on a different scale (about 10x its AC power), so **AC power is the prediction target**.
- Weather has one sensor per plant, so it is merged onto every inverter by timestamp and plant.

## Method

1. **Preprocessing:** parse dates, merge generation with weather, add time features, build a plant-level 15-minute grid.
2. **Power model:** Linear Regression (baseline), Random Forest and XGBoost, compared on R², MAE and RMSE.
3. **Validation:** time-based split. Training uses the earlier days, testing uses the final days, so the model is judged on days it has never seen.
4. **Forecast model:** lag features (power and irradiation 15, 30 and 60 minutes ago) for four horizons, compared against a naive "persistence" baseline (future = now).
5. **Fault detection:** daytime readings only. Each inverter is compared with the median inverter at the same timestamps (peer ratio), with the model's expected power, and by how often it outputs zero while others produce.
6. **Day-ahead forecast:** Open-Meteo temperature and radiation feed the power model. Module temperature, which the forecast lacks, is estimated from ambient temperature and irradiation (R² 0.97 on this data).

## Results

### Power prediction (test days only)

| Model | R² | MAE (kW) | RMSE (kW) |
|---|---|---|---|
| Linear Regression | 0.822 | 76.95 | 136.90 |
| Random Forest | **0.880** | **36.22** | **112.14** |
| XGBoost | _add from your run_ | | |

| Level | R² | MAE (kW) | RMSE (kW) |
|---|---|---|---|
| Per inverter | 0.880 | 36.22 | 112.14 |
| **Plant level** (average of inverters) | **0.964** | **19.74** | **58.62** |

Irradiation accounts for about 95% of the model's decisions. Per-inverter accuracy is lower than plant-level
accuracy because individual inverters differ from each other and some underperform, which weather alone cannot explain.

### Short-term forecast (plant power in kW, daytime test hours)

| Horizon | Baseline RMSE | Model RMSE | Improvement | Model R² |
|---|---|---|---|---|
| +1 hour | 5,244 | 3,851 | 27% | 0.55 |
| +2 hours | 7,026 | 4,370 | 38% | 0.42 |
| +3 hours | 8,492 | 4,474 | 47% | 0.40 |
| +4 hours | 9,741 | 4,468 | 54% | 0.39 |

The model beats the baseline at every horizon, and the advantage grows with the horizon.

### Fault detection

| Plant | Healthy | Warning | Fault |
|---|---|---|---|
| Plant 1 | 22 | 0 | 0 |
| Plant 2 | 8 | 10 | 4 |

Plant 2 inverters often output zero during daylight while their neighbours are producing. The estimated energy lost
to weak inverters is about 893 MWh at Plant 2 (roughly ₹27 lakh at the placeholder tariff of ₹3 per kWh) and about 19 MWh at Plant 1.

### Energy and impact (placeholder tariff and emission factor, editable in the dashboard)

| Plant | Energy | Revenue at ₹3/kWh | CO₂ avoided at 0.80 kg/kWh |
|---|---|---|---|
| Plant 1 | 5,292 MWh | ₹1.59 Cr | 4,233 t |
| Plant 2 | 4,084 MWh | ₹1.23 Cr | 3,267 t |

## Dashboard pages

| Page | Content |
|---|---|
| **Home** | KPI summary and navigation |
| **Overview** | Power over time, daily energy, typical-day curve, irradiation vs power, correlation heatmap, plant and date filters |
| **Predict** | What-if sliders with instant prediction, model comparison, feature importance, actual vs predicted |
| **Fault Detection** | Inverter-by-day heatmap, health scores, status table, one-inverter drill-down |
| **Forecast** | 1 to 4 hour forecast vs actual vs baseline, and tomorrow's output from a live weather forecast |
| **Impact** | Revenue, CO₂ avoided and recoverable loss with editable assumptions |

## Project structure

```
solar-power-project/
├── .streamlit/config.toml        # dark theme
├── assets/
│   ├── style.css                 # dashboard styling
│   └── screenshots/              # images used in this README
├── data/                         # the 4 original CSV files
├── models/                       # trained models, metrics and test predictions
├── notebooks/                    # exploration
├── pages/                        # Streamlit pages (Overview, Predict, Fault Detection, Forecast, Impact)
├── src/
│   ├── preprocess.py             # cleaning, merging, plant-level grid
│   ├── train_model.py            # power prediction models
│   ├── train_forecast.py         # 1 to 4 hour forecast models
│   ├── fault_detection.py        # inverter health analysis
│   ├── impact.py                 # energy, revenue, CO2
│   ├── weather_api.py            # Open-Meteo forecast and day-ahead prediction
│   └── ui.py                     # dashboard helpers and cached loaders
├── app.py                        # dashboard home page
├── requirements.txt
└── README.md
```

## Run it locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/solar-power-project.git
cd solar-power-project

python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

python src/preprocess.py       # builds the merged dataset
python src/train_model.py      # trains the power models
python -m src.train_forecast   # trains the forecast models
streamlit run app.py           # opens the dashboard
```

Run the scripts from the project root. The `python -m src...` form is required for modules that import each other.
The trained models are included in the repository, so `streamlit run app.py` works without retraining.

## Limitations

- **Short data window:** about 34 days in one season (May to June 2020). Other seasons, such as winter or monsoon, are not covered, and training on a full year is the most valuable next step.
- **Forecast test period is only 7 days**, so forecast metrics are indicative.
- **No confirmed fault records:** fault detection thresholds were chosen by inspecting the data. Flags are indications to investigate, not proven faults, and accuracy cannot be scored without maintenance logs.
- **Plant locations are not in the dataset:** the day-ahead forecast uses a chosen city as a demonstration. Forecast radiation is measured on a flat surface while plant sensors are on tilted panels, so a calibration slider is provided.
- **Tariff and emission factor are placeholders:** replace them with sourced values before drawing conclusions.

## Future work

- Train on a full year or more of data, and add seasonal features
- Test deep learning models (LSTM) against the current models
- Validate fault detection against real maintenance records
- Use irradiance on the tilted plane and real plant coordinates for the day-ahead forecast
- Automatic alerts when an inverter drops below its peers

## Credits

- Dataset: Ani Rudra Sen Nandi, [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data), Kaggle
- Weather forecasts: [Open-Meteo.com](https://open-meteo.com), licensed under CC BY 4.0