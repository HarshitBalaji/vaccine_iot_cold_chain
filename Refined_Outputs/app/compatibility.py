import pandas as pd
from pathlib import Path

path = Path("synthetic_train.csv")   # change to your file
df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)

required = ["Device_ID","timestamp","temp_reported","humidity_reported","lat_reported","lon_reported","distance_from_route_reported"]
recommended = ["timestamp_reported","accel_mag_reported","tamper_reported","temp_true","cyberattack_anomaly","temperature_anomaly_true","primary_label_reported","cvss","epss","dvd"]

missing_required = [c for c in required if c not in df.columns]
missing_recommended = [c for c in recommended if c not in df.columns]

print("Missing required columns:", missing_required)
print("Missing recommended columns:", missing_recommended)

# Simple fallbacks (non-destructive) — only add if columns missing
if "timestamp_reported" not in df.columns:
    df["timestamp_reported"] = df["timestamp"]
if "accel_mag_reported" not in df.columns:
    df["accel_mag_reported"] = 0.0
if "tamper_reported" not in df.columns:
    df["tamper_reported"] = 0
if "cyberattack_anomaly" not in df.columns:
    df["cyberattack_anomaly"] = 0
if "temperature_anomaly_true" not in df.columns:
    if "temp_true" in df.columns:
        df["temperature_anomaly_true"] = ((df["temp_true"]<2.0) | (df["temp_true"]>8.0)).astype(int)
    else:
        df["temperature_anomaly_true"] = 0
# Posture defaults
for c in ["cvss","epss","dvd"]:
    if c not in df.columns:
        df[c] = 0.0

out = path.with_name(path.stem + "_validated.csv")
df.to_csv(out, index=False)
print("Saved validated file to", out)
