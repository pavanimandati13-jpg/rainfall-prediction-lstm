"""
generate_sample_data.py
------------------------
This script CREATES a synthetic (fake but realistic-looking) daily weather
dataset so that the Rainfall Prediction LSTM project can be run immediately,
even if you don't have a real dataset yet.

It produces a CSV file with columns:
    Date, Temperature, Humidity, Pressure, WindSpeed, Rainfall

The values follow believable seasonal patterns (e.g. more rain and higher
humidity in "monsoon" months, cooler temperatures in "winter" months) plus
random noise, so the LSTM has realistic patterns to learn from.

You can REPLACE this generated CSV with any real dataset that has the same
column structure (see README.md -> "Dataset" section for public dataset
sources).

Run it with:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

# Setting a random seed makes the "random" data reproducible
np.random.seed(42)

# ---- 1. Decide how many days of data to generate ----
NUM_DAYS = 1460  # 4 years of daily data (gives the LSTM enough history)

start_date = pd.to_datetime("2020-01-01")
dates = pd.date_range(start=start_date, periods=NUM_DAYS, freq="D")

# ---- 2. Build seasonal patterns using a sine wave ----
# day_of_year controls the yearly seasonal cycle (0 to 365)
day_of_year = dates.dayofyear.values
seasonal_cycle = np.sin(2 * np.pi * (day_of_year - 80) / 365.0)  # peaks in mid-year

# ---- 3. Temperature (deg C): warmer in "summer" part of the cycle ----
temperature = 25 + 8 * seasonal_cycle + np.random.normal(0, 1.5, NUM_DAYS)

# ---- 4. Humidity (%): higher when it's about to rain / rainy season ----
humidity = 60 + 20 * seasonal_cycle + np.random.normal(0, 5, NUM_DAYS)
humidity = np.clip(humidity, 20, 100)

# ---- 5. Atmospheric Pressure (hPa): slightly lower during rainy season ----
pressure = 1013 - 6 * seasonal_cycle + np.random.normal(0, 2, NUM_DAYS)

# ---- 6. Wind Speed (km/h): a bit higher during rainy season ----
wind_speed = 12 + 4 * seasonal_cycle + np.random.normal(0, 2, NUM_DAYS)
wind_speed = np.clip(wind_speed, 0, None)

# ---- 7. Rainfall (mm): depends on humidity + seasonal cycle + randomness ----
# Rain is more likely (and heavier) when humidity is high and it's the rainy season.
rain_probability = 1 / (1 + np.exp(-(0.06 * (humidity - 65) + 1.5 * seasonal_cycle)))
is_raining = np.random.binomial(1, np.clip(rain_probability, 0, 1))
rain_amount = np.random.gamma(shape=2.0, scale=6.0, size=NUM_DAYS)  # skewed, mostly small values
rainfall = is_raining * rain_amount
rainfall = np.round(np.clip(rainfall, 0, None), 1)

# ---- 8. Assemble into a DataFrame ----
df = pd.DataFrame({
    "Date": dates,
    "Temperature": np.round(temperature, 1),
    "Humidity": np.round(humidity, 1),
    "Pressure": np.round(pressure, 1),
    "WindSpeed": np.round(wind_speed, 1),
    "Rainfall": rainfall,
})

# ---- 9. Deliberately introduce a few missing values ----
# This mimics real-world sensor data, so the notebook's "handle missing values"
# step actually has something to do.
missing_idx = np.random.choice(df.index, size=15, replace=False)
missing_cols = np.random.choice(["Temperature", "Humidity", "Pressure", "WindSpeed"], size=15)
for idx, col in zip(missing_idx, missing_cols):
    df.loc[idx, col] = np.nan

# ---- 10. Save to CSV ----
output_path = "sample_rainfall_data.csv"
df.to_csv(output_path, index=False)
print(f"Sample dataset created: {output_path}")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Missing values per column:\n{df.isna().sum()}")
