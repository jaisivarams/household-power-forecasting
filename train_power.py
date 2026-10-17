"""
train_power.py - Household electric power forecasting with an LSTM

TASK
    Given the previous 24 hours of readings, predict the average
    Global_active_power (in kW) of the NEXT hour.

WHAT THIS SCRIPT DOES
    1. Loads household_power_consumption.csv (1 reading per minute)
    2. Converts it to hourly averages
    3. Splits it by time: first 70% train, next 15% validation, last 15% test
    4. Scales the features using the TRAINING part only (no data leakage)
    5. Trains a stacked LSTM
    6. Compares the LSTM against simple baselines on the unseen test period
    7. Saves the model, the scaling info, a sample input file and two plots

RUN:   python train_power.py
OUTPUT FILES (all written to the current folder):
    lstm_power_model.keras   trained model
    power_meta.json          scaling values, window size, test start, etc.
    hourly_power.csv         hourly data (used by predict.py)
    sample_input.csv         24 hourly rows you can edit and feed to predict.py
    training_curves.png      loss during training
    actual_vs_predicted.png  last 7 days of the test period
"""
import os
import json

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots as files, no pop-up window that blocks the script
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------- Settings -----------------------------
CSV_FILE = "household_power_consumption.csv"
TARGET = "Global_active_power"
RAW_FEATURES = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
WINDOW = 24        # hours of history given to the model
HORIZON = 1        # how many hours ahead to predict (1 = the next hour)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15    # the remaining 15% is the test set
EPOCHS = 60
BATCH_SIZE = 64
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# ----------------------------- Helpers -----------------------------
def load_hourly(path):
    """Read the minute-level file and return hourly averages."""
    df = pd.read_csv(path, na_values=["?", ""])
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    hourly = df[RAW_FEATURES].resample("h").mean()
    missing_hours = int(hourly.isna().any(axis=1).sum())
    hourly = hourly.interpolate(method="time", limit_direction="both")  # fill short gaps
    return hourly, missing_hours


def time_features(index):
    """Hour-of-day and day-of-week as sine/cosine so 23:00 is 'close' to 00:00."""
    hour, dow = index.hour.values, index.dayofweek.values
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ])


def make_windows(features, target_scaled, first_target, last_target_exclusive):
    """Build (24 hours of inputs -> target value) samples for targets in a range."""
    X, y, positions = [], [], []
    for t in range(max(first_target, WINDOW + HORIZON - 1), last_target_exclusive):
        start = t - WINDOW - HORIZON + 1
        X.append(features[start:start + WINDOW])
        y.append(target_scaled[t])
        positions.append(t)
    return np.array(X, dtype="float32"), np.array(y, dtype="float32"), np.array(positions)


def metrics(y_true, y_pred):
    err = y_true - y_pred
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


# ----------------------------- Data -----------------------------
print("Loading data...")
hourly, missing_hours = load_hourly(CSV_FILE)
n = len(hourly)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
print(f"Hourly rows: {n}  ({hourly.index[0]}  ->  {hourly.index[-1]})")
print(f"Hours that were missing in the raw data and got interpolated: {missing_hours}")
print(f"Train: {hourly.index[0]} .. {hourly.index[train_end - 1]}")
print(f"Val:   {hourly.index[train_end]} .. {hourly.index[val_end - 1]}")
print(f"Test:  {hourly.index[val_end]} .. {hourly.index[-1]}")

# Scale using TRAINING rows only
col_min = hourly.iloc[:train_end].min()
col_max = hourly.iloc[:train_end].max()
col_range = (col_max - col_min).replace(0, 1.0)
scaled = (hourly - col_min) / col_range

features = np.hstack([scaled.values, time_features(hourly.index)]).astype("float32")
n_features = features.shape[1]
target_col = RAW_FEATURES.index(TARGET)
target_scaled = scaled[TARGET].values.astype("float32")


def to_kw(v):
    return np.asarray(v) * float(col_range[TARGET]) + float(col_min[TARGET])


X_train, y_train, _ = make_windows(features, target_scaled, 0, train_end)
X_val, y_val, _ = make_windows(features, target_scaled, train_end, val_end)
X_test, y_test, test_pos = make_windows(features, target_scaled, val_end, n)
print(f"Samples -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# ----------------------------- Model -----------------------------
model = models.Sequential([
    layers.Input(shape=(WINDOW, n_features)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ],
)

# ----------------------------- Evaluation -----------------------------
y_true = to_kw(y_test)
y_pred = to_kw(model.predict(X_test, batch_size=256, verbose=0).ravel())

power = hourly[TARGET].values
baseline_last = power[test_pos - HORIZON]   # "same as the last hour we know"
baseline_yday = power[test_pos - 24]        # "same hour yesterday"

rows = {
    "LSTM": metrics(y_true, y_pred),
    "Repeat last hour": metrics(y_true, baseline_last),
    "Same hour yesterday": metrics(y_true, baseline_yday),
}
print("\n=========== TEST RESULTS (unseen period) ===========")
print(f"{'Method':<22}{'MAE (kW)':>10}{'RMSE (kW)':>12}")
for name, (mae, rmse) in rows.items():
    print(f"{name:<22}{mae:>10.4f}{rmse:>12.4f}")
print(f"Average power in the test period: {y_true.mean():.3f} kW")

best_base_mae = min(rows["Repeat last hour"][0], rows["Same hour yesterday"][0])
gain = 100 * (1 - rows["LSTM"][0] / best_base_mae)
if gain > 0:
    print(f"LSTM MAE is {gain:.1f}% lower than the best baseline.")
else:
    print(f"LSTM MAE is {abs(gain):.1f}% HIGHER than the best baseline - the model is not adding value.")

# ----------------------------- Plots -----------------------------
plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.xlabel("Epoch"); plt.ylabel("MAE loss (scaled)"); plt.title("Training curves"); plt.legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.close()

last = 24 * 7
times = hourly.index[test_pos][-last:]
plt.figure(figsize=(14, 4))
plt.plot(times, y_true[-last:], label="Actual", linewidth=1.5)
plt.plot(times, y_pred[-last:], label="LSTM", linewidth=1.2)
plt.plot(times, baseline_last[-last:], label="Repeat last hour", linewidth=0.8, alpha=0.6)
plt.ylabel("kW"); plt.title("Next-hour power: last 7 days of the test period"); plt.legend()
plt.tight_layout(); plt.savefig("actual_vs_predicted.png", dpi=150); plt.close()

# ----------------------------- Save -----------------------------
model.save("lstm_power_model.keras")

meta = {
    "target": TARGET,
    "raw_features": RAW_FEATURES,
    "window": WINDOW,
    "horizon": HORIZON,
    "min": {k: float(v) for k, v in col_min.items()},
    "range": {k: float(v) for k, v in col_range.items()},
    "test_start": str(hourly.index[val_end]),
    "data_end": str(hourly.index[-1]),
    "test_mae": rows["LSTM"][0],
    "test_rmse": rows["LSTM"][1],
}
with open("power_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

hourly.rename_axis("datetime").round(4).to_csv("hourly_power.csv")
hourly.iloc[-WINDOW:].rename_axis("datetime").round(4).to_csv("sample_input.csv")

print("\nSaved: lstm_power_model.keras, power_meta.json, hourly_power.csv, "
      "sample_input.csv, training_curves.png, actual_vs_predicted.png")
