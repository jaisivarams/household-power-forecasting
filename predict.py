"""
predict.py - Predict the next hour's household power consumption (kW)

The model reads the previous 24 hours of readings (7 measurements per hour,
plus the time of day / day of week) and predicts the average
Global_active_power of the following hour.

Run train_power.py first. It creates the files this script needs:
    lstm_power_model.keras, power_meta.json, hourly_power.csv, sample_input.csv

HOW TO USE
  1) Interactive (it asks you for a date and time):
         python predict.py
  2) One prediction for a specific hour from the dataset:
         python predict.py --at "2008-12-10 18:00"
  3) Your own data - a CSV with 24 hourly rows, predicts the hour AFTER the last row:
         python predict.py --csv sample_input.csv

CSV FORMAT for option 3 (see sample_input.csv):
    datetime,Global_active_power,Global_reactive_power,Voltage,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3
    A file in the original dataset format (Date and Time columns, minute data) also works.
"""
import os
import sys
import json
import random
import argparse

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

for needed in ("lstm_power_model.keras", "power_meta.json"):
    if not os.path.exists(needed):
        sys.exit(f"Missing {needed}. Run train_power.py first (in this same folder).")

with open("power_meta.json") as f:
    meta = json.load(f)

FEATURES = meta["raw_features"]
TARGET = meta["target"]
WINDOW = meta["window"]
HORIZON = meta["horizon"]
model = load_model("lstm_power_model.keras")


def time_features(index):
    hour, dow = index.hour.values, index.dayofweek.values
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ])


def predict_kw(window_df):
    """window_df: exactly WINDOW hourly rows (hourly DatetimeIndex, all FEATURES columns)."""
    scaled = np.column_stack(
        [(window_df[c].values - meta["min"][c]) / meta["range"][c] for c in FEATURES]
    )
    x = np.hstack([scaled, time_features(window_df.index)]).astype("float32")[np.newaxis, ...]
    p = float(model.predict(x, verbose=0)[0][0])
    return p * meta["range"][TARGET] + meta["min"][TARGET]


# ------------------------- Mode: hour from the dataset -------------------------
def load_hourly_dataset():
    if not os.path.exists("hourly_power.csv"):
        sys.exit("Missing hourly_power.csv. Run train_power.py first (in this same folder).")
    return pd.read_csv("hourly_power.csv", index_col="datetime", parse_dates=True)


def predict_at(hourly, when):
    try:
        target_hour = pd.Timestamp(when).floor("h")
    except (ValueError, TypeError):
        print("Could not read that date. Use a format like 2008-12-10 18:00")
        return
    last_input = target_hour - pd.Timedelta(hours=HORIZON)
    first_input = last_input - pd.Timedelta(hours=WINDOW - 1)
    window = hourly.loc[first_input:last_input]

    if target_hour not in hourly.index or len(window) != WINDOW:
        earliest = hourly.index[0] + pd.Timedelta(hours=WINDOW + HORIZON - 1)
        print(f"No complete {WINDOW}-hour history for that hour. "
              f"Choose a time between {earliest} and {hourly.index[-1]}.")
        return

    predicted = predict_kw(window)
    actual = float(hourly.loc[target_hour, TARGET])
    last_known = float(window[TARGET].iloc[-1])

    print(f"\nPredicting the average power for {target_hour} "
          f"using the {WINDOW} hours before it ({first_input} to {last_input}):")
    print(f"  LSTM prediction     : {predicted:6.3f} kW")
    print(f"  Actual              : {actual:6.3f} kW   (error {abs(predicted - actual):.3f} kW)")
    print(f"  Previous hour (naive): {last_known:6.3f} kW   (error {abs(last_known - actual):.3f} kW)")
    if target_hour < pd.Timestamp(meta["test_start"]):
        print(f"  Note: this hour was part of the training/validation data, so the result is "
              f"optimistic. Hours from {meta['test_start']} onward were never seen in training.")


def interactive():
    hourly = load_hourly_dataset()
    test_start = pd.Timestamp(meta["test_start"])
    test_hours = hourly.loc[test_start:].index
    print(f"\nData covers {hourly.index[0]} to {hourly.index[-1]}.")
    print(f"Hours the model has never seen (the test period) start at {test_start}.")
    print("Type a date and time (e.g. 2008-12-10 18:00), 'r' for a random unseen hour, or 'q' to quit.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if text.lower() in ("r", "random"):
            text = str(random.choice(list(test_hours)))
        if text:
            predict_at(hourly, text)


# ------------------------- Mode: your own CSV -------------------------
def predict_from_csv(path):
    df = pd.read_csv(path, na_values=["?", ""])
    if "datetime" in df.columns:
        df.index = pd.to_datetime(df["datetime"])
    elif {"Date", "Time"} <= set(df.columns):
        df.index = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    else:
        sys.exit("The CSV needs a 'datetime' column (or 'Date' and 'Time' columns).")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"The CSV is missing these columns: {missing}")

    hourly = df[FEATURES].sort_index().resample("h").mean().interpolate(method="time", limit_direction="both")
    if len(hourly) < WINDOW:
        sys.exit(f"Need at least {WINDOW} hours of data, but the file only covers {len(hourly)} hours.")
    window = hourly.iloc[-WINDOW:]
    if window.isna().any().any():
        sys.exit("The last 24 hours contain empty values that could not be filled.")

    next_hour = window.index[-1] + pd.Timedelta(hours=HORIZON)
    print(f"\nUsing the {WINDOW} hours from {window.index[0]} to {window.index[-1]}")
    print(f"Predicted average power for {next_hour}: {predict_kw(window):.3f} kW")


# ------------------------- Entry point -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next hour of household power use.")
    parser.add_argument("--at", help='predict one hour from the dataset, e.g. "2008-12-10 18:00"')
    parser.add_argument("--csv", help="predict the hour after the last row of your own 24-hour CSV")
    args = parser.parse_args()

    if args.csv:
        predict_from_csv(args.csv)
    elif args.at:
        predict_at(load_hourly_dataset(), args.at)
    else:
        interactive()
