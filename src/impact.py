"""Energy, money and CO2 calculations.

AC_POWER in the dataset is in kW and readings are every 15 minutes,
so energy (kWh) = power (kW) x 0.25 h.
"""
import pandas as pd

INTERVAL_H = 0.25

# ASSUMPTIONS - editable in the dashboard sidebar. Replace with sourced values
# and cite them in your report.
DEFAULT_TARIFF = 3.0       # INR per kWh
DEFAULT_EMISSION = 0.80    # kg CO2 per kWh of grid electricity avoided


def energy_kwh(df, power_col="AC_POWER"):
    return float(df[power_col].sum() * INTERVAL_H)


def daily_energy(df):
    """Energy per plant per day (kWh), from inverter-level data."""
    out = (df.groupby(["PLANT", "DATE"])["AC_POWER"].sum() * INTERVAL_H)
    return out.rename("ENERGY_KWH").reset_index()


def impact(kwh, tariff=DEFAULT_TARIFF, emission_factor=DEFAULT_EMISSION):
    return {
        "energy_mwh": kwh / 1000,
        "revenue_inr": kwh * tariff,
        "co2_tonnes": kwh * emission_factor / 1000,
    }


def format_inr(x):
    """Indian-style short format: lakh / crore."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


if __name__ == "__main__":
    from src.preprocess import build_dataset
    from src.fault_detection import add_expected_power, inverter_daily_health
    df = build_dataset()
    for plant, part in df.groupby("PLANT"):
        kwh = energy_kwh(part)
        r = impact(kwh)
        print(f"Plant {plant}: {r['energy_mwh']:.0f} MWh | {format_inr(r['revenue_inr'])}"
              f" | {r['co2_tonnes']:.0f} t CO2")
    daily = inverter_daily_health(add_expected_power(df))
    loss = daily.groupby("PLANT")["LOSS_KWH"].sum()
    for plant, kwh in loss.items():
        r = impact(kwh)
        print(f"Plant {plant} estimated loss from weak inverters: {r['energy_mwh']:.0f} MWh"
              f" ({format_inr(r['revenue_inr'])})")