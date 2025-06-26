# %% [markdown]
#
# Install extra dependencies for this notebook if needed (when running
# jupyterlite). We need the development version of skrub to be able to use the
# skrub expressions.
# %%
%pip install -q holidays https://pypi.anaconda.org/ogrisel/simple/skrub/0.6.dev0/skrub-0.6.dev0-py3-none-any.whl

# %%
import polars as pl
import skrub
from pathlib import Path

# %%
time = skrub.var(
    "time",
    pl.DataFrame().with_columns(
        pl.datetime_range(
            pl.datetime(2021, 1, 1, hour=0),
            pl.datetime(2025, 6, 30, hour=23),
            time_zone="UTC",
            interval="1h",
        ).alias("time"),
    ),
)
time

# %% [markdown]
# TODO: add instructions to download data manually

# %%
data_source_folder = Path("../datasets")
for data_file in sorted(data_source_folder.iterdir()):
    print(data_file)

# %%
electricity_raw = skrub.var(
    "electricity_raw",
    pl.concat(
        [
            pl.read_csv(data_file, null_values=["N/A"])
            for data_file in sorted(data_source_folder.iterdir())
            if data_file.name.startswith("Total Load - Day Ahead")
            and data_file.name.endswith(".csv")
        ],
        how="vertical",
    ),
)
electricity_raw

# %%
electricity = (
    electricity_raw.with_columns(
        [
            pl.col("Time (UTC)")
            .str.split(by=" - ")
            .list.first()
            .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
            .alias("time"),
        ]
    )
    .drop(["Time (UTC)", "Day-ahead Total Load Forecast [MW] - BZN|FR"])
    .rename({"Actual Total Load [MW] - BZN|FR": "load_mw"})
    .select(["time", "load_mw"])
)
electricity

# %% [markdown]
# Check that the number of rows matches our expectations based on the number of hours that separate the first and the last dates:

# %%
time.join(electricity, on="time", how="left")



# %%
some_city_weather_raw = skrub.var(
    "paris_weather_raw",
    pl.read_parquet("../datasets/weather_paris.parquet"),
).with_columns(
    [
        pl.col("time").dt.cast_time_unit("us"),  # Ensure time column has the same type
    ]
)
some_city_weather_raw

# %%
some_city_weather = some_city_weather_raw.rename(lambda x: x if x == "time" else x + " some_city")
time.join(some_city_weather, on="time", how="left")

# %%
import holidays

holidays_fr = holidays.France(years=range(2019, 2026))


fr_time = pl.col("time").dt.convert_time_zone("Europe/Paris")
calendar = time.with_columns(
    [
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("is_holiday_fr"),
        # TODO: add school holidays flags for the 3 main zones in France
        fr_time.dt.weekday().alias("day_of_week_fr"),
        fr_time.dt.ordinal_day().alias("day_of_year_fr"),
        fr_time.dt.hour().alias("hour_of_day_fr"),
    ],
)
calendar

# %%
