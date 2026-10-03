import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_plant(n):
    gen = pd.read_csv(DATA / f"Plant_{n}_Generation_Data.csv")
    wea = pd.read_csv(DATA / f"Plant_{n}_Weather_Sensor_Data.csv")
    # Plant 1 generation uses DD-MM-YYYY, the others use ISO format
    if n == 1:
        gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format="%d-%m-%Y %H:%M")
    else:
        gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"])
    wea["DATE_TIME"] = pd.to_datetime(wea["DATE_TIME"])
    wea = wea.drop(columns=["SOURCE_KEY"])
    return gen.merge(wea, on=["DATE_TIME", "PLANT_ID"], how="inner")


def build_dataset():
    """One row per inverter per 15-minute timestamp, with weather attached."""
    df = pd.concat([load_plant(1), load_plant(2)], ignore_index=True)
    df = df.drop_duplicates().sort_values("DATE_TIME").reset_index(drop=True)
    df["PLANT"] = df["PLANT_ID"].map({4135001: 1, 4136001: 2})
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["MINUTE"] = df["DATE_TIME"].dt.minute
    df["TIME_OF_DAY"] = df["HOUR"] + df["MINUTE"] / 60
    df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek
    df["DATE"] = df["DATE_TIME"].dt.date
    return df


def build_plant_level(df):
    """One row per plant per 15-minute timestamp (used for forecasting).

    AC_POWER = average inverter power x total inverters in the plant, so that
    timestamps where a few inverters have no reading are not under-counted.
    The timeline is reindexed to a regular 15-minute grid; gaps stay as NaN.
    """
    n_total = df.groupby("PLANT")["SOURCE_KEY"].nunique()
    g = df.groupby(["PLANT", "DATE_TIME"]).agg(
        AC_MEAN=("AC_POWER", "mean"),
        IRRADIATION=("IRRADIATION", "mean"),
        AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE", "mean"),
        MODULE_TEMPERATURE=("MODULE_TEMPERATURE", "mean"),
    ).reset_index()

    frames = []
    for plant, part in g.groupby("PLANT"):
        part = part.set_index("DATE_TIME").sort_index()
        full = pd.date_range(part.index.min(), part.index.max(), freq="15min")
        part = part.reindex(full)
        part["PLANT"] = plant
        part["AC_POWER"] = part["AC_MEAN"] * n_total[plant]
        part.index.name = "DATE_TIME"
        frames.append(part.drop(columns="AC_MEAN").reset_index())
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(DATA / "merged_solar_data.csv", index=False)
    print("Saved:", df.shape)
    print(df.isnull().sum().sum(), "nulls")
    grid = build_plant_level(df)
    print("Plant-level grid:", grid.shape,
          "| missing timestamps:", int(grid["AC_POWER"].isnull().sum()))