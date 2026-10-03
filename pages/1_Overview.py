import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import setup, header, kpi, style_fig, load_data, load_grid, PLANT_COLORS
from src.impact import daily_energy

setup("Overview", "📊")
header("Plant Overview", "How much power the plants produce and what drives it")

df, grid = load_data(), load_grid()

# ---- sidebar filters ----
st.sidebar.header("Filters")
choice = st.sidebar.radio("Plant", ["Both", "Plant 1", "Plant 2"])
dmin, dmax = df["DATE"].min(), df["DATE"].max()
picked = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(picked, (tuple, list)):
    start, end = (picked[0], picked[-1]) if len(picked) else (dmin, dmax)
else:
    start = end = picked

plants = [1, 2] if choice == "Both" else [int(choice[-1])]
t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)

d = df[df["PLANT"].isin(plants) & (df["DATE_TIME"] >= t0) & (df["DATE_TIME"] < t1)].copy()
g = grid[grid["PLANT"].isin(plants) & (grid["DATE_TIME"] >= t0) & (grid["DATE_TIME"] < t1)].copy()
if d.empty:
    st.warning("No data for this selection.")
    st.stop()
d["Plant"] = "Plant " + d["PLANT"].astype(str)
g["Plant"] = "Plant " + g["PLANT"].astype(str)

# ---- KPIs ----
energy_mwh = d["AC_POWER"].sum() * 0.25 / 1000
n_days = d["DATE"].nunique()
peak_mw = g["AC_POWER"].max() / 1000
sun = d.loc[d["IRRADIATION"] > 0.05, "IRRADIATION"].mean()

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Total energy", f"{energy_mwh:,.0f} MWh", f"{n_days} days selected")
kpi(c2, "Avg daily energy", f"{energy_mwh / n_days / len(plants):,.0f} MWh", "Per plant per day")
kpi(c3, "Peak plant output", f"{peak_mw:,.1f} MW", "Highest 15-minute value")
kpi(c4, "Avg daytime irradiation", f"{sun:.2f} kW/m²", "Sun above the horizon")

# ---- charts ----
fig = px.line(g, x="DATE_TIME", y="AC_POWER", color="Plant", color_discrete_map=PLANT_COLORS,
              labels={"AC_POWER": "Plant power (kW)", "DATE_TIME": ""},
              title="Plant power over time")
st.plotly_chart(style_fig(fig, 380))

left, right = st.columns(2)
de = daily_energy(d)
de["Plant"] = "Plant " + de["PLANT"].astype(str)
de["ENERGY_MWH"] = de["ENERGY_KWH"] / 1000
fig = px.bar(de, x="DATE", y="ENERGY_MWH", color="Plant", barmode="group",
             color_discrete_map=PLANT_COLORS, labels={"ENERGY_MWH": "Energy (MWh)", "DATE": ""},
             title="Daily energy")
left.plotly_chart(style_fig(fig))

curve = d.groupby(["Plant", "TIME_OF_DAY"])["AC_POWER"].mean().reset_index()
fig = px.line(curve, x="TIME_OF_DAY", y="AC_POWER", color="Plant", color_discrete_map=PLANT_COLORS,
              labels={"TIME_OF_DAY": "Hour of day", "AC_POWER": "Avg power per inverter (kW)"},
              title="Typical day: the bell curve")
right.plotly_chart(style_fig(fig))

left, right = st.columns(2)
sample = d.sample(min(4000, len(d)), random_state=1)
fig = px.scatter(sample, x="IRRADIATION", y="AC_POWER", color="Plant", opacity=0.5,
                 color_discrete_map=PLANT_COLORS,
                 labels={"IRRADIATION": "Irradiation (kW/m²)", "AC_POWER": "Inverter power (kW)"},
                 title="Irradiation vs power")
left.plotly_chart(style_fig(fig))

names = {"AC_POWER": "AC power", "IRRADIATION": "Irradiation",
         "MODULE_TEMPERATURE": "Module temp", "AMBIENT_TEMPERATURE": "Ambient temp",
         "TIME_OF_DAY": "Time of day"}
corr = d[list(names)].rename(columns=names).corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="YlOrRd", aspect="auto",
                title="What is power correlated with?")
right.plotly_chart(style_fig(fig))