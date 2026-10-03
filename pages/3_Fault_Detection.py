import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import (setup, header, kpi, style_fig, load_data, load_health, load_solar_model,
                    add_labels, STATUS_COLORS)
from src.impact import impact, format_inr, DEFAULT_TARIFF
from src import fault_detection as fd

setup("Fault Detection", "🛠️")
header("Inverter Fault Detection", "Finding inverters that produce less than they should")

daily, summary = load_health()
daily, summary = add_labels(daily), add_labels(summary)

st.sidebar.header("Filters")
plant = st.sidebar.radio("Plant", [1, 2], index=1, format_func=lambda p: f"Plant {p}")
tariff = st.sidebar.number_input("Tariff (₹ per kWh)", min_value=0.0, max_value=50.0,
                                 value=float(DEFAULT_TARIFF), step=0.1)

dp = daily[daily["PLANT"] == plant]
sp = summary[summary["PLANT"] == plant].copy()

# ---- KPIs ----
counts = sp["STATUS"].value_counts()
loss_kwh = dp["LOSS_KWH"].sum()
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Healthy", int(counts.get("Healthy", 0)), f"of {len(sp)} inverters")
kpi(c2, "Warning", int(counts.get("Warning", 0)), "Below 90% of peers or frequent zero output")
kpi(c3, "Fault", int(counts.get("Fault", 0)), "Below 70% of peers or mostly offline")
kpi(c4, "Estimated energy lost", f"{loss_kwh / 1000:,.0f} MWh",
    f"worth about {format_inr(impact(loss_kwh, tariff)['revenue_inr'])}")

# ---- heatmap: inverter x day ----
pivot = dp.pivot_table(index="INVERTER", columns="DATE", values="PEER_RATIO")
order = sp.sort_values("PEER_RATIO")["INVERTER"].tolist()
pivot = pivot.reindex(order)
fig = go.Figure(go.Heatmap(
    z=pivot.values * 100, x=[str(c) for c in pivot.columns], y=list(pivot.index),
    colorscale="RdYlGn", zmin=40, zmax=110, colorbar=dict(title="% of peers")))
fig.update_layout(title="Daily output compared with peer inverters (red = underperforming)")
st.plotly_chart(style_fig(fig, 520))

# ---- health score + table ----
left, right = st.columns([1, 1])
bars = sp.sort_values("HEALTH_SCORE")
fig = px.bar(bars, x="INVERTER", y="HEALTH_SCORE", color="STATUS",
             color_discrete_map=STATUS_COLORS, labels={"HEALTH_SCORE": "Health score"},
             title="Health score per inverter")
fig.update_yaxes(range=[0, 105])
left.plotly_chart(style_fig(fig, 400))

table = pd.DataFrame({
    "Inverter": sp["INVERTER"], "Status": sp["STATUS"], "Health score": sp["HEALTH_SCORE"],
    "Zero output %": (sp["ZERO_SHARE"] * 100).round(1), "Fault days": sp["FAULT_DAYS"],
    "Warning days": sp["WARNING_DAYS"], "Energy lost (MWh)": (sp["LOSS_KWH"] / 1000).round(1),
    "Source key": sp["SOURCE_KEY"]})
right.markdown("##### Inverter table (worst first)")
right.dataframe(table, hide_index=True, height=400, column_config={
    "Health score": st.column_config.ProgressColumn(
        "Health score", min_value=0, max_value=100, format="%.0f")})

# ---- drill-down ----
st.markdown("### Drill-down: one inverter, one day")
c1, c2 = st.columns(2)
inv = c1.selectbox("Inverter", sp["INVERTER"].tolist())
dates = sorted(dp["DATE"].unique())
day = c2.selectbox("Date", dates, format_func=str)

key = sp.loc[sp["INVERTER"] == inv, "SOURCE_KEY"].iloc[0]
raw = load_data()
day_df = raw[(raw["PLANT"] == plant) & (raw["DATE"] == day)]
mine = day_df[day_df["SOURCE_KEY"] == key].set_index("DATE_TIME")
if mine.empty:
    st.info("No readings for this inverter on that day.")
else:
    model = load_solar_model()
    expected = pd.Series(np.clip(model["model"].predict(mine[model["features"]]), 0, None),
                         index=mine.index, name="Model expectation")
    peer = day_df.groupby("DATE_TIME")["AC_POWER"].median().rename("Peer median")
    plot = pd.concat([mine["AC_POWER"].rename("Actual"), peer, expected], axis=1)
    plot = plot.dropna(subset=["Actual"]).reset_index()
    long = plot.melt("DATE_TIME", var_name="Series", value_name="kW")
    fig = px.line(long, x="DATE_TIME", y="kW", color="Series",
                  color_discrete_map={"Actual": "#E74C3C", "Peer median": "#2ECC71",
                                      "Model expectation": "#29B6F6"},
                  labels={"kW": "AC power (kW)", "DATE_TIME": ""},
                  title=f"{inv} on {day}: drops to zero while peers keep producing = fault")
    st.plotly_chart(style_fig(fig, 380))

with st.expander("How detection works and how to read it"):
    st.markdown(f"""
- Only **daytime** readings count (irradiation at least {fd.DAY_IRR} kW/m²), and only when the median inverter is producing.
- **Peer ratio** = an inverter's energy ÷ the median inverter's energy at the same moments. Comparing with peers means
  weather affects everyone equally.
- **Zero output** = share of daytime readings where the inverter produced nothing while others did.
- **Status:** *Fault* if peer ratio < {fd.FAULT_RATIO:.0%} or zero output > {fd.FAULT_ZERO:.0%};
  *Warning* if peer ratio < {fd.WARN_RATIO:.0%} or zero output > {fd.WARN_ZERO:.0%}; otherwise *Healthy*.
- **Energy lost** = the shortfall against the peer median on days an inverter was not healthy.
- These thresholds were chosen by inspecting this dataset. There are no confirmed fault records, so the flags are
  indications to investigate, not proven faults.
    """)