import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import setup, header, kpi, style_fig, load_data, load_health, PLANT_COLORS
from src.impact import (daily_energy, impact, format_inr, DEFAULT_TARIFF, DEFAULT_EMISSION)

setup("Impact", "💰")
header("Financial and Environmental Impact", "What the generated energy is worth")

st.sidebar.header("Assumptions")
tariff = st.sidebar.number_input("Tariff (₹ per kWh)", min_value=0.0, max_value=50.0,
                                 value=float(DEFAULT_TARIFF), step=0.1)
emission = st.sidebar.number_input("Grid emission factor (kg CO₂ per kWh)", min_value=0.0,
                                   max_value=2.0, value=float(DEFAULT_EMISSION), step=0.01)
st.sidebar.caption("These are placeholder assumptions. Replace them with sourced values "
                   "and cite them in your report.")

df = load_data()
daily, _ = load_health()
de = daily_energy(df)
de["Plant"] = "Plant " + de["PLANT"].astype(str)

rows = []
for p in (1, 2):
    kwh = de.loc[de["PLANT"] == p, "ENERGY_KWH"].sum()
    loss = daily.loc[daily["PLANT"] == p, "LOSS_KWH"].sum()
    r, rl = impact(kwh, tariff, emission), impact(loss, tariff, emission)
    rows.append({"Plant": f"Plant {p}", "Energy (MWh)": r["energy_mwh"],
                 "Revenue (₹)": r["revenue_inr"], "CO₂ avoided (t)": r["co2_tonnes"],
                 "Recoverable loss (MWh)": rl["energy_mwh"], "Loss value (₹)": rl["revenue_inr"]})
tot = pd.DataFrame(rows)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Energy generated", f"{tot['Energy (MWh)'].sum():,.0f} MWh", "Both plants")
kpi(c2, "Revenue", format_inr(tot["Revenue (₹)"].sum()), f"at ₹{tariff:.2f} per kWh")
kpi(c3, "CO₂ avoided", f"{tot['CO₂ avoided (t)'].sum():,.0f} t", f"at {emission:.2f} kg per kWh")
kpi(c4, "Recoverable loss", format_inr(tot["Loss value (₹)"].sum()),
    f"{tot['Recoverable loss (MWh)'].sum():,.0f} MWh lost to weak inverters")

left, right = st.columns(2)
de["Revenue (₹ lakh)"] = de["ENERGY_KWH"] * tariff / 1e5
fig = px.bar(de, x="DATE", y="Revenue (₹ lakh)", color="Plant", barmode="stack",
             color_discrete_map=PLANT_COLORS, labels={"DATE": ""}, title="Daily revenue")
left.plotly_chart(style_fig(fig))

de = de.sort_values(["PLANT", "DATE"])
de["Cumulative CO₂ avoided (t)"] = de.groupby("PLANT")["ENERGY_KWH"].cumsum() * emission / 1000
fig = px.line(de, x="DATE", y="Cumulative CO₂ avoided (t)", color="Plant",
              color_discrete_map=PLANT_COLORS, labels={"DATE": ""}, title="CO₂ avoided over time")
right.plotly_chart(style_fig(fig))

st.markdown("### What weak inverters cost")
stack = tot.melt("Plant", value_vars=["Energy (MWh)", "Recoverable loss (MWh)"],
                 var_name="Type", value_name="MWh")
fig = px.bar(stack, x="Plant", y="MWh", color="Type", barmode="stack",
             color_discrete_map={"Energy (MWh)": "#FFB300", "Recoverable loss (MWh)": "#E74C3C"},
             title="Actual energy plus energy lost to underperforming inverters")
st.plotly_chart(style_fig(fig, 340))

show = tot.copy()
for col in show.columns[1:]:
    show[col] = show[col].round(1)
st.dataframe(show, hide_index=True)
st.caption("Recoverable loss = shortfall of non-healthy inverters against the peer median (see Fault Detection).")