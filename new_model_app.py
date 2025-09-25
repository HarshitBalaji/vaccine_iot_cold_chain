# new_model_app.py
"""
Streamlit app for the NEW model:
- Layered time-aware detectors + cyberattack detector + fusion MLP
- Trains models on the simulated dataset (or uploaded CSV)
- Includes evaluation metrics: classification reports, PR-AUC/ROC-AUC, detection latency, false alarms/device-day
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="New Model - Layered Detector", layout="wide")
st.title("Layered Time-Aware Anomaly Detector (New Model)")

# Paths & defaults
DEFAULT_DATA_PATH = "simulated_iot_attacks_dataset.csv"
MODELS_DIR = "./models_new_model"
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------
# Utility / feature funcs
# ----------------------
@st.cache_data
def load_dataset(path=None):
    if path is None:
        path = DEFAULT_DATA_PATH
    df = pd.read_csv(path, parse_dates=["timestamp","timestamp_reported"])
    df = df.sort_values(["Device_ID","timestamp"]).reset_index(drop=True)
    return df

def engineer_features(df):
    # make a copy to avoid modifying caller's df unexpectedly
    df = df.sort_values(["Device_ID", "timestamp"]).reset_index(drop=True)

    # Rolling features, deltas, gps repeat counts, timestamp diffs
    for w in [3, 5]:
        df[f"temp_roll_mean_{w}"] = df.groupby("Device_ID")["temp_reported"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        df[f"temp_roll_std_{w}"] = df.groupby("Device_ID")["temp_reported"].transform(
            lambda x: x.rolling(window=w, min_periods=1).std().fillna(0)
        )
        df[f"hum_roll_mean_{w}"] = df.groupby("Device_ID")["humidity_reported"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        df[f"hum_roll_std_{w}"] = df.groupby("Device_ID")["humidity_reported"].transform(
            lambda x: x.rolling(window=w, min_periods=1).std().fillna(0)
        )

    # deltas
    df["delta_temp"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.diff().fillna(0))
    df["delta_hum"] = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.diff().fillna(0))
    df["delta_dist"] = df.groupby("Device_ID")["distance_from_route_reported"].transform(lambda x: x.diff().fillna(0))

    # gps repeat count (identical consecutive reported positions) computed per-device using transform
    lat_same = df.groupby("Device_ID")["lat_reported"].transform(lambda x: x.round(6) == x.round(6).shift(1))
    lon_same = df.groupby("Device_ID")["lon_reported"].transform(lambda x: x.round(6) == x.round(6).shift(1))
    df["gps_same"] = (lat_same & lon_same).astype(int)
    df["gps_repeat_count"] = df.groupby("Device_ID")["gps_same"].transform(
        lambda x: x.groupby((x == 0).cumsum()).cumcount() + 1
    ).fillna(0).astype(int)

    # timestamp diffs per device (in minutes)
    if "timestamp_reported" in df.columns:
        df["reported_ts_diff_min"] = df.groupby("Device_ID")["timestamp_reported"].transform(
            lambda x: x.diff().dt.total_seconds().div(60).fillna(5)
        )
    else:
        df["reported_ts_diff_min"] = 5

    return df

def ensure_labels(df):
    # Ensure cyberattack_anomaly exists
    if "cyberattack_anomaly" not in df.columns:
        df["cyberattack_anomaly"] = 0
    # Temperature anomaly true label (deterministic) if not present
    if "temperature_anomaly_true" not in df.columns:
        if "temp_true" in df.columns:
            df["temp_breach_true"] = ((df["temp_true"] < 2.0) | (df["temp_true"] > 8.0)).astype(int)
            df["temp_breach_true_consec"] = df.groupby("Device_ID")["temp_breach_true"].transform(
                lambda x: x.groupby((x == 0).cumsum()).cumcount() + 1
            ).fillna(0).astype(int)
            df["temperature_anomaly_true"] = (df["temp_breach_true_consec"] >= 5).astype(int)
        else:
            # if no true channel, set 0
            df["temperature_anomaly_true"] = 0
    return df

# ----------------------
# Metrics & utilities
# ----------------------
def binary_metrics(y_true, y_score, threshold=0.7):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except Exception:
        report = {}
    # PR AUC
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
    except Exception:
        pr_auc = None
    # ROC AUC
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = None
    return {"report": report, "pr_auc": pr_auc, "roc_auc": roc, "y_pred": y_pred}

def compute_detection_latency(df, true_flag_col, pred_flag_col, timestamp_col="timestamp", device_col="Device_ID"):
    events = []
    # work per device
    for dev, g in df.groupby(device_col):
        g = g.sort_values(timestamp_col).reset_index(drop=True)
        is_true = g[true_flag_col].fillna(0).astype(int).values
        n = len(is_true)
        i = 0
        while i < n:
            if is_true[i] == 1:
                start_idx = i
                while i+1 < n and is_true[i+1] == 1:
                    i += 1
                end_idx = i
                start_time = pd.to_datetime(g.loc[start_idx, timestamp_col])
                end_time = pd.to_datetime(g.loc[end_idx, timestamp_col])
                # find first pred==1 at or after start_idx
                pred_after = g.loc[start_idx:, pred_flag_col].fillna(0).astype(int).values
                rel_idx = np.where(pred_after == 1)[0]
                if len(rel_idx) > 0:
                    detect_idx = start_idx + int(rel_idx[0])
                    detect_time = pd.to_datetime(g.loc[detect_idx, timestamp_col])
                    latency_min = (detect_time - start_time).total_seconds() / 60.0
                    detected = True
                else:
                    latency_min = np.nan
                    detected = False
                events.append({
                    "Device_ID": dev,
                    "start_time": start_time,
                    "end_time": end_time,
                    "detected": detected,
                    "latency_min": latency_min,
                    "true_len_min": (end_time - start_time).total_seconds() / 60.0
                })
            i += 1
    return pd.DataFrame(events)

def false_alarms_per_device_day(df, pred_col, true_col, timestamp_col="timestamp", device_col="Device_ID"):
    df_local = df[[device_col, timestamp_col, pred_col, true_col]].copy()
    df_local["date"] = pd.to_datetime(df_local[timestamp_col]).dt.floor("D")
    df_local["fp"] = ((df_local[pred_col] == 1) & (df_local[true_col] == 0)).astype(int)
    grouped = df_local.groupby([device_col, "date"])["fp"].sum().reset_index()
    avg_fp_per_dev_day = grouped["fp"].mean() if len(grouped) > 0 else 0.0
    per_device = grouped.groupby(device_col)["fp"].mean().reset_index().rename(columns={"fp": "avg_fp_per_day"})
    return avg_fp_per_dev_day, per_device

# ----------------------
# Training / model funcs
# ----------------------
def train_models(df, force_retrain=False):
    meta_path = os.path.join(MODELS_DIR, "model_meta.joblib")
    temp_path = os.path.join(MODELS_DIR, "temp_detector.joblib")
    attack_path = os.path.join(MODELS_DIR, "cyberattack_detector.joblib")
    fusion_path = os.path.join(MODELS_DIR, "fusion_mlp.joblib")
    fusion_scaler_path = os.path.join(MODELS_DIR, "fusion_scaler.joblib")
    le_path = os.path.join(MODELS_DIR, "fusion_label_encoder.joblib")

    if (not force_retrain) and os.path.exists(temp_path) and os.path.exists(attack_path) and os.path.exists(fusion_path):
        st.info("Loading existing models from disk.")
        temp_clf = joblib.load(temp_path)
        attack_clf = joblib.load(attack_path)
        fusion_clf = joblib.load(fusion_path)
        fusion_scaler = joblib.load(fusion_scaler_path)
        le = joblib.load(le_path)
        meta = joblib.load(meta_path)
        return temp_clf, attack_clf, fusion_clf, fusion_scaler, le, meta

    st.info("Training models. This will take a short while.")
    df = engineer_features(df)
    df = ensure_labels(df)

    # time-based split index
    split_idx = int(0.7 * len(df))

    # --- temperature detector ---
    temp_features = ["temp_reported", "temp_roll_mean_3", "temp_roll_std_3", "temp_roll_mean_5", "temp_roll_std_5", "delta_temp"]
    for c in temp_features:
        if c not in df.columns:
            df[c] = 0.0
    X_temp = df[temp_features].fillna(0).values
    y_temp = df["temperature_anomaly_true"].fillna(0).astype(int).values
    X_temp_train = X_temp[:split_idx]
    y_temp_train = y_temp[:split_idx]

    temp_clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    temp_clf.fit(X_temp_train, y_temp_train)
    joblib.dump(temp_clf, temp_path)

    # --- cyberattack detector ---
    attack_features = ["temp_reported", "humidity_reported", "temp_roll_std_3", "hum_roll_std_3",
                       "delta_temp", "delta_hum", "distance_from_route_reported", "delta_dist",
                       "accel_mag_reported", "gps_repeat_count", "reported_ts_diff_min", "tamper_reported"]
    for c in attack_features:
        if c not in df.columns:
            df[c] = 0.0
    X_attack = df[attack_features].fillna(0).astype(float).values
    y_attack = df["cyberattack_anomaly"].fillna(0).astype(int).values
    X_attack_train = X_attack[:split_idx]
    y_attack_train = y_attack[:split_idx]

    attack_clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    attack_clf.fit(X_attack_train, y_attack_train)
    joblib.dump(attack_clf, attack_path)

    # detector scores for fusion
    df["score_temp"] = temp_clf.predict_proba(X_temp)[:, 1]
    df["score_attack"] = attack_clf.predict_proba(X_attack)[:, 1]

    # fusion labels: use primary_label_reported if present else derive
    if "primary_label_reported" not in df.columns:
        # derive reported temperature anomaly if missing
        if "temperature_anomaly_reported" not in df.columns:
            df["temp_breach_rep"] = ((df["temp_reported"] < 2.0) | (df["temp_reported"] > 8.0)).astype(int)
            df["temp_breach_rep_consec"] = df.groupby("Device_ID")["temp_breach_rep"].transform(lambda x: x.groupby((x == 0).cumsum()).cumcount() + 1).fillna(0).astype(int)
            df["temperature_anomaly_reported"] = (df["temp_breach_rep_consec"] >= 5).astype(int)
        df["Loss_of_storage_condition"] = ((df["temperature_anomaly_reported"] == 1) | (df.get("humidity_anomaly_reported", 0) == 1)).astype(int)
        df["Tamper_Damage_anomaly"] = (df.get("tamper_anomaly_reported", df.get("tamper_reported", 0))).astype(int)
        df["Geofence_anomaly"] = df.get("geofence_anomaly_reported", 0).astype(int)

        def derive_label(r):
            if r["Tamper_Damage_anomaly"] == 1:
                return "Tamper_Damage_anomaly"
            if r["cyberattack_anomaly"] == 1:
                return "Cyberattack_anomaly"
            if r["Loss_of_storage_condition"] == 1:
                return "Loss_of_storage_condition"
            if r["Geofence_anomaly"] == 1:
                return "Geofence_anomaly"
            return "normal"
        df["primary_label_reported"] = df.apply(derive_label, axis=1)

    # fusion features & training
    fusion_meta = ["cvss", "epss", "dvd"]
    for c in fusion_meta:
        if c not in df.columns:
            df[c] = 0.0
    fusion_features = ["score_temp", "score_attack", "distance_from_route_reported", "gps_repeat_count", "temp_roll_std_3"] + fusion_meta
    X_fusion = df[fusion_features].fillna(0).values
    le = LabelEncoder()
    y_fusion = le.fit_transform(df["primary_label_reported"].astype(str).values)

    X_f_train = X_fusion[:split_idx]
    y_f_train = y_fusion[:split_idx]

    fusion_scaler = StandardScaler()
    X_f_train_scaled = fusion_scaler.fit_transform(X_f_train)

    fusion_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    fusion_clf.fit(X_f_train_scaled, y_f_train)
    joblib.dump(fusion_clf, fusion_path)
    joblib.dump(fusion_scaler, fusion_scaler_path)
    joblib.dump(le, le_path)

    # save meta
    meta = {"temp_features": temp_features, "attack_features": attack_features, "fusion_features": fusion_features}
    joblib.dump(meta, meta_path)

    st.success("Training complete. Models saved to '{}'".format(MODELS_DIR))
    return temp_clf, attack_clf, fusion_clf, fusion_scaler, le, meta

# ----------------------
# Sidebar / controls
# ----------------------
st.sidebar.header("Data & Training")
uploaded = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])
retrain = st.sidebar.button("Retrain models (force)")

# Load dataset
if uploaded is not None:
    try:
        df = load_dataset(uploaded)
    except Exception as e:
        st.error("Failed to load uploaded CSV: " + str(e))
        st.stop()
else:
    try:
        df = load_dataset()
    except Exception as e:
        st.error("Default dataset not found at {}. Upload CSV in the sidebar.".format(DEFAULT_DATA_PATH))
        st.stop()

st.sidebar.write("Rows: {}, Devices: {}".format(len(df), df["Device_ID"].nunique()))

# Train or load models
if retrain:
    temp_clf, attack_clf, fusion_clf, fusion_scaler, le, meta = train_models(df, force_retrain=True)
else:
    temp_clf, attack_clf, fusion_clf, fusion_scaler, le, meta = train_models(df, force_retrain=False)

# Run inference on dataset (compute detector scores and fusion labels)
df = engineer_features(df)
df = ensure_labels(df)

# Make sure features exist
for c in meta["temp_features"]:
    if c not in df.columns: df[c] = 0
for c in meta["attack_features"]:
    if c not in df.columns: df[c] = 0
for c in meta["fusion_features"]:
    if c not in df.columns: df[c] = 0

X_temp = df[meta["temp_features"]].fillna(0).values
X_attack = df[meta["attack_features"]].fillna(0).values
df["score_temp"] = temp_clf.predict_proba(X_temp)[:, 1]
df["score_attack"] = attack_clf.predict_proba(X_attack)[:, 1]
X_fusion = df[meta["fusion_features"]].fillna(0).values
X_fusion_scaled = fusion_scaler.transform(X_fusion)
df["fusion_code"] = fusion_clf.predict(X_fusion_scaled)
df["final_label"] = le.inverse_transform(df["fusion_code"])

# ----------------------
# Evaluation UI & metrics
# ----------------------
st.header("Model evaluation & metrics")
temp_thresh = st.slider("Temp detector threshold", 0.0, 1.0, 0.5, 0.01)
attack_thresh = st.slider("Attack detector threshold", 0.0, 1.0, 0.5, 0.01)

# Temperature detector metrics
if "temperature_anomaly_true" in df.columns:
    res_temp = binary_metrics(df["temperature_anomaly_true"].fillna(0).astype(int).values, df["score_temp"].fillna(0).values, threshold=temp_thresh)
    st.subheader("Temperature detector")
    st.write("PR-AUC:", res_temp["pr_auc"], "ROC-AUC:", res_temp["roc_auc"])
    st.json(res_temp["report"])
    df["temp_pred_bin"] = res_temp["y_pred"]
    latency_temp = compute_detection_latency(df, "temperature_anomaly_true", "temp_pred_bin")
    if not latency_temp.empty:
        st.write("Temp detection latency (median min):", float(np.nanmedian(latency_temp["latency_min"])))
        st.write("Temp detection coverage (detected fraction):", float(latency_temp["detected"].mean()))
        st.dataframe(latency_temp.describe().transpose())
    avg_fp, per_dev = false_alarms_per_device_day(df, "temp_pred_bin", "temperature_anomaly_true")
    st.write(f"Avg false positives per device-day (temp): {avg_fp:.3f}")
    st.dataframe(per_dev)

else:
    st.info("temperature_anomaly_true label missing — temperature detector eval skipped.")

# Cyberattack detector metrics
if "cyberattack_anomaly" in df.columns:
    res_attack = binary_metrics(df["cyberattack_anomaly"].fillna(0).astype(int).values, df["score_attack"].fillna(0).values, threshold=attack_thresh)
    st.subheader("Cyberattack detector")
    st.write("PR-AUC:", res_attack["pr_auc"], "ROC-AUC:", res_attack["roc_auc"])
    st.json(res_attack["report"])
    df["attack_pred_bin"] = res_attack["y_pred"]
    latency_attack = compute_detection_latency(df, "cyberattack_anomaly", "attack_pred_bin")
    if not latency_attack.empty:
        st.write("Attack detection latency (median min):", float(np.nanmedian(latency_attack["latency_min"])))
        st.write("Attack detection coverage (detected fraction):", float(latency_attack["detected"].mean()))
        st.dataframe(latency_attack.describe().transpose())
    avg_fp_a, per_dev_a = false_alarms_per_device_day(df, "attack_pred_bin", "cyberattack_anomaly")
    st.write(f"Avg false positives per device-day (attack): {avg_fp_a:.3f}")
    st.dataframe(per_dev_a)
else:
    st.info("cyberattack_anomaly label missing — attack detector eval skipped.")

# Fusion multi-class evaluation
if "primary_label_reported" in df.columns:
    st.subheader("Fusion (multi-class) evaluation")
    y_true = df["primary_label_reported"].fillna("normal").astype(str).values
    y_pred = df["final_label"].astype(str).values
    st.text(classification_report(y_true, y_pred, zero_division=0))
    labels_union = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels_union)
    st.write("Confusion matrix labels:", labels_union.tolist())
    st.dataframe(pd.DataFrame(cm, index=labels_union, columns=labels_union))
else:
    st.info("No primary_label_reported found — fusion multi-class evaluation skipped.")

# ----------------------
# Main UI: dataset, per-device outputs, plots, downloads
# ----------------------
st.subheader("Dataset & model outputs")
col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("Rows", f"{len(df):,}")
with col2:
    st.metric("Devices", df["Device_ID"].nunique())
with col3:
    st.write("Model folder:", MODELS_DIR)

st.subheader("Per-device recent outputs")
device_list = df["Device_ID"].unique().tolist()
sel_dev = st.selectbox("Select device", device_list, index=0)
dev_df = df[df["Device_ID"] == sel_dev].sort_values("timestamp").reset_index(drop=True)

display_cols = ["timestamp", "temp_reported"]
if "temp_true" in dev_df.columns:
    display_cols.append("temp_true")
if "temperature_anomaly_true" in dev_df.columns:
    display_cols.append("temperature_anomaly_true")
display_cols += ["score_temp", "temp_pred_bin" if "temp_pred_bin" in df.columns else None, "score_attack", "attack_pred_bin" if "attack_pred_bin" in df.columns else None, "final_label", "cyberattack_anomaly"]
# filter None and missing
display_cols = [c for c in display_cols if c is not None and c in dev_df.columns]
st.dataframe(dev_df[display_cols].tail(200))

st.subheader("Time-series inspector")
idx = st.slider("Select index (for selected device)", 0, max(0, len(dev_df) - 1), 0)
window = st.slider("Window around index (samples)", 10, 200, 80)
start = max(0, idx - window // 2)
end = min(len(dev_df) - 1, idx + window // 2)
slice_df = dev_df.iloc[start:end + 1]

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(slice_df["timestamp"], slice_df["temp_reported"], label="temp_reported", linewidth=1)
if "temp_true" in slice_df.columns:
    axes[0].plot(slice_df["timestamp"], slice_df["temp_true"], label="temp_true", linewidth=1, linestyle="--")
axes[0].legend(); axes[0].set_ylabel("Temp (°C)")
axes[1].plot(slice_df["timestamp"], slice_df["distance_from_route_reported"], label="dist_reported", linewidth=1)
if "distance_from_route_true" in slice_df.columns:
    axes[1].plot(slice_df["timestamp"], slice_df["distance_from_route_true"], label="dist_true", linewidth=1, linestyle="--")
axes[1].legend(); axes[1].set_ylabel("Distance (m)")
st.pyplot(fig)

st.subheader("Final label distribution and download")
st.bar_chart(df["final_label"].value_counts())

# Download predictions + metrics
metrics_cols = ["Device_ID", "timestamp", "temp_reported", "temperature_anomaly_true", "temp_pred_bin",
                "score_temp", "cyberattack_anomaly", "attack_pred_bin", "score_attack", "final_label", "primary_label_reported"]
available = [c for c in metrics_cols if c in df.columns]
csv_bytes = df[available].to_csv(index=False).encode("utf-8")
st.download_button("Download predictions & labels CSV", csv_bytes, "new_model_with_metrics.csv", "text/csv")

st.info("This app trains per-feature detectors and a fusion MLP (new-model pipeline). Use the Retrain button to force retraining. Threshold sliders adjust binary thresholds for detector evaluation.")
