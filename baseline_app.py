# app.py
"""
Streamlit app: Baselines for IoT vaccine cold-chain anomaly detection
- Isolation Forest (telemetry-only baseline)
- Risk Fusion (telemetry + CVSS/EPSS/DVD) using LogisticRegression

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

st.set_page_config(page_title="IoT Cold-Chain Baselines", layout="wide")
st.title("IoT Vaccine Cold-Chain — Baseline Models")
st.markdown(
    "Isolation Forest (telemetry-only) and Risk-Fusion (telemetry + CVSS/EPSS/DVD). "
    "Upload dataset or use the default simulated file."
)

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_dataset(path=None):
    if path is None:
        path = "simulated_iot_attacks_dataset.csv"
    df = pd.read_csv(path, parse_dates=["timestamp", "timestamp_reported"])
    return df

def time_based_split(df, frac=0.7):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * frac)
    return df_sorted.iloc[:cutoff_idx].copy(), df_sorted.iloc[cutoff_idx:].copy()

def train_isolation_forest(X_train, contamination=0.05, random_state=42):
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X_train)
    return iso

def train_risk_fusion(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf

def eval_classification(y_true, y_pred, y_score=None):
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    if y_score is not None:
        try:
            roc = roc_auc_score(y_true, y_score)
        except Exception:
            roc = None
    else:
        roc = None
    return report, roc

def plot_confusion(cm, labels, title):
    fig, ax = plt.subplots(figsize=(3.5,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    st.pyplot(fig)

def make_timeseries_plot(df_device, cols, anomalies_mask=None, title=None):
    fig, ax = plt.subplots(figsize=(12,3))
    for c in cols:
        ax.plot(df_device["timestamp"], df_device[c], label=c, linewidth=1)
    if anomalies_mask is not None:
        ax.scatter(df_device["timestamp"][anomalies_mask], df_device[cols[0]][anomalies_mask],
                   color="red", label="Detected anomaly", s=20, zorder=5)
    ax.legend(loc="upper right")
    if title: ax.set_title(title)
    ax.set_xlabel("timestamp")
    st.pyplot(fig)

# -------------------------
# Sidebar: data & params
# -------------------------
st.sidebar.header("Data & Parameters")
uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])
use_default = st.sidebar.checkbox("Use default simulated dataset (/mnt/data/...)", value=True)

contamination = st.sidebar.slider("IsolationForest contamination (expected anomaly fraction)", 0.001, 0.2, 0.05, 0.001)
test_frac = st.sidebar.slider("Train fraction (time-based)", 0.5, 0.9, 0.7, 0.05)
sampling_minutes = st.sidebar.number_input("Sampling interval (mins)", value=5, step=1)
run_train = st.sidebar.button("Train / (Retrain) baselines")

# -------------------------
# Load data
# -------------------------
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, parse_dates=["timestamp", "timestamp_reported"])
        st.sidebar.success("Uploaded dataset loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    if use_default:
        try:
            df = load_dataset()
            st.sidebar.success("Loaded default simulated dataset")
        except Exception as e:
            st.sidebar.error(f"Could not load default dataset: {e}")
            st.stop()
    else:
        st.info("Upload a dataset or enable default. Nothing to show yet.")
        st.stop()

# Quick overview
st.subheader("Dataset overview")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.metric("Rows", f"{len(df):,}")
with c2:
    st.metric("Devices", df["Device_ID"].nunique())
with c3:
    st.write("Columns sample:", df.columns[:20].tolist())

# -------------------------
# Prepare features
# -------------------------
st.markdown("### Feature preparation (telemetry-only vs telemetry+risk)")
# Telemetry-only feature set (baseline for IsolationForest)
telemetry_features = ["temp_reported", "humidity_reported", "distance_from_route_reported", "accel_mag_reported"]
# Add rolling features if available
for c in ["temp_roll_mean_3","temp_roll_std_3","hum_roll_mean_3","hum_roll_std_3","delta_temp","delta_hum","gps_repeat_count"]:
    if c in df.columns:
        telemetry_features.append(c)

# Risk features (if present)
risk_features = []
for c in ["cvss","epss","dvd"]:
    if c in df.columns:
        risk_features.append(c)

st.write("Telemetry features used:", telemetry_features)
st.write("Risk features present:", risk_features)

# Create deterministic reported-level ground truth any_anomaly for baseline evaluation (user-facing)
if "primary_label_reported" in df.columns:
    df["any_anomaly_reported"] = (df["primary_label_reported"] != "normal").astype(int)
else:
    # fallback: any of reported anomaly flags
    rep_flags = [c for c in df.columns if c.endswith("_anomaly_reported")]
    if len(rep_flags) > 0:
        df["any_anomaly_reported"] = df[rep_flags].any(axis=1).astype(int)
    else:
        # if no label exists, ask user to provide labels
        st.warning("No reported anomaly label columns found. The baselines will train against 'cyberattack_anomaly' if available.")
        if "cyberattack_anomaly" in df.columns:
            df["any_anomaly_reported"] = df["cyberattack_anomaly"]
        else:
            st.error("No labels available for evaluation. Upload a dataset with anomaly labels (e.g., 'cyberattack_anomaly' or 'primary_label_reported').")
            st.stop()

# -------------------------
# Train / evaluate
# -------------------------
if run_train:
    with st.spinner("Performing time-based split and training baselines..."):
        # Time-based split
        train_df, test_df = time_based_split(df, frac=test_frac)

        # Prepare telemetry matrices (scale)
        scaler_tele = StandardScaler()
        X_train_tele = scaler_tele.fit_transform(train_df[telemetry_features].fillna(0))
        X_test_tele = scaler_tele.transform(test_df[telemetry_features].fillna(0))

        # Isolation Forest (telemetry-only)
        iso = train_isolation_forest(X_train_tele, contamination=contamination)
        iso_pred_test = iso.predict(X_test_tele)
        # IsolationForest returns -1 (anomaly) and 1 (normal)
        iso_pred_bin = np.where(iso_pred_test == -1, 1, 0)  # 1=anomaly

        # Risk Fusion model: telemetry + risk
        if len(risk_features) > 0:
            fusion_features = telemetry_features + risk_features
        else:
            fusion_features = telemetry_features  # fallback if no risk
        scaler_fusion = StandardScaler()
        X_train_fusion = scaler_fusion.fit_transform(train_df[fusion_features].fillna(0))
        X_test_fusion = scaler_fusion.transform(test_df[fusion_features].fillna(0))
        y_train = train_df["any_anomaly_reported"].values
        y_test = test_df["any_anomaly_reported"].values

        # Logistic Regression baseline
        rf_clf = train_risk_fusion(X_train_fusion, y_train)
        rf_probs = rf_clf.predict_proba(X_test_fusion)[:,1]
        rf_preds = (rf_probs >= 0.7).astype(int)

        # Evaluation
        iso_report, iso_roc = eval_classification(y_test, iso_pred_bin, None)
        rf_report, rf_roc = eval_classification(y_test, rf_preds, rf_probs)

        # show results
        st.subheader("Baseline results (test set)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Isolation Forest (telemetry-only)**")
            st.write("Test set rows:", len(test_df))
            st.write("Anomaly fraction (test):", round(test_df["any_anomaly_reported"].mean(),4))
            st.text(classification_report(y_test, iso_pred_bin, zero_division=0))
            cm_iso = confusion_matrix(y_test, iso_pred_bin)
            plot_confusion(cm_iso, labels=["normal","anomaly"], title="IF Confusion Matrix")
        with col_b:
            st.markdown("**Risk Fusion (Logistic Regression)**")
            st.write("Features used:", fusion_features)
            st.text(classification_report(y_test, rf_preds, zero_division=0))
            try:
                st.write("ROC-AUC:", round(roc_auc_score(y_test, rf_probs),4))
            except Exception:
                pass
            cm_rf = confusion_matrix(y_test, rf_preds)
            plot_confusion(cm_rf, labels=["normal","anomaly"], title="Risk-Fusion Confusion Matrix")

        # Save models/ scalers into session_state for later live scoring
        st.session_state["iso_model"] = iso
        st.session_state["iso_scaler"] = scaler_tele
        st.session_state["rf_model"] = rf_clf
        st.session_state["rf_scaler"] = scaler_fusion
        st.session_state["fusion_features"] = fusion_features
        st.session_state["telemetry_features"] = telemetry_features
        st.session_state["test_df"] = test_df
        st.session_state["train_df"] = train_df

        st.success("Training completed and models stored in session state.")

# -------------------------
# Live inference / playback demo
# -------------------------
st.markdown("### Live playback / single-device inspection")
devs = df["Device_ID"].unique().tolist()
sel_dev = st.selectbox("Select Device", devs, index=0)

dev_df = df[df["Device_ID"]==sel_dev].sort_values("timestamp").reset_index(drop=True)
play_idx = st.slider("Playback timestamp index", min_value=0, max_value=len(dev_df)-1, value=0, step=1)
row = dev_df.loc[play_idx]

st.write("Timestamp:", row["timestamp"])
st.write("Reported readings (temp, hum, dist):", round(row["temp_reported"],2), round(row["humidity_reported"],2), round(row["distance_from_route_reported"],2))
st.write("Reported tamper:", row.get("tamper_reported", False))
st.write("Ground truth cyberattack flag (if present):", row.get("cyberattack_anomaly", np.nan))

# If models trained, run live scoring for this single row with context features (rolling windows)
if "iso_model" in st.session_state and "rf_model" in st.session_state:
    iso = st.session_state["iso_model"]
    scaler_tele = st.session_state["iso_scaler"]
    rf_clf = st.session_state["rf_model"]
    scaler_fusion = st.session_state["rf_scaler"]
    telemetry_features = st.session_state["telemetry_features"]
    fusion_features = st.session_state["fusion_features"]

    X_live_tele = scaler_tele.transform(row[telemetry_features].fillna(0).values.reshape(1,-1))
    iso_pred = iso.predict(X_live_tele)
    iso_bin = 1 if iso_pred[0] == -1 else 0

    X_live_fusion = scaler_fusion.transform(row[fusion_features].fillna(0).values.reshape(1,-1))
    rf_prob = rf_clf.predict_proba(X_live_fusion)[:,1][0]
    rf_pred = int(rf_prob >= 0.7)

    st.markdown("**Live model outputs for selected sample**")
    st.write(f"IsolationForest anomaly (0/1): {iso_bin}")
    st.write(f"RiskFusion prob / pred: {round(rf_prob,3)} / {rf_pred}")

else:
    st.info("Train the models using the sidebar 'Train / (Retrain) baselines' button to enable live scoring.")

# -------------------------
# Device-level dashboard & downloads
# -------------------------
st.markdown("## Device-level summary & download predictions")
if "test_df" in st.session_state:
    # Show device-level risk posture (mean cvss/epss/dvd)
    risk_table = df.groupby("Device_ID")[[c for c in risk_features]].mean().reset_index() if len(risk_features)>0 else None
    if risk_table is not None:
        st.subheader("Device risk posture (mean)")
        st.dataframe(risk_table)

    # Run entire test set predictions if models are present
    if "iso_model" in st.session_state:
        iso = st.session_state["iso_model"]
        scaler_tele = st.session_state["iso_scaler"]
        X_all_tele = scaler_tele.transform(df[telemetry_features].fillna(0))
        iso_pred_all = iso.predict(X_all_tele)
        df["IF_anomaly_pred"] = np.where(iso_pred_all == -1, 1, 0)

    if "rf_model" in st.session_state:
        rf = st.session_state["rf_model"]
        scaler_f = st.session_state["rf_scaler"]
        X_all_f = scaler_f.transform(df[fusion_features].fillna(0))
        df["RF_prob"] = rf.predict_proba(X_all_f)[:,1]
        df["RF_pred"] = (df["RF_prob"] >= 0.7).astype(int)

    # Download CSV of predictions
    st.subheader("Download predicted labels")
    preview_cols = ["Device_ID","timestamp","temp_reported","humidity_reported","distance_from_route_reported","IF_anomaly_pred","RF_prob","RF_pred","primary_label_reported","cyberattack_anomaly"] if "cyberattack_anomaly" in df.columns else ["Device_ID","timestamp","temp_reported","humidity_reported","distance_from_route_reported","IF_anomaly_pred","RF_prob","RF_pred","primary_label_reported"]
    st.dataframe(df[preview_cols].head(200))

    csv = df[preview_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", csv, "predictions_with_baselines.csv", "text/csv")

# -------------------------
# Example time-series plots (attack vs normal slices)
# -------------------------
st.markdown("## Example traces")
example_dev = st.selectbox("Plot device", devs, index=0, key="plot_dev")
dev_df = df[df["Device_ID"]==example_dev].sort_values("timestamp").reset_index(drop=True)
# Allow user to pick a time window
start_idx = st.number_input("Start index (for plot)", min_value=0, max_value=len(dev_df)-2, value=0, step=1)
end_idx = st.number_input("End index (for plot)", min_value=1, max_value=len(dev_df)-1, value=min(120, len(dev_df)-1), step=1)
slice_df = dev_df.iloc[start_idx:end_idx+1]
# plot temp and reported vs true
fig, ax = plt.subplots(2,1, figsize=(12,5), sharex=True)
ax[0].plot(slice_df["timestamp"], slice_df["temp_true"], label="temp_true", linewidth=1)
ax[0].plot(slice_df["timestamp"], slice_df["temp_reported"], label="temp_reported", linewidth=1, linestyle="--")
ax[0].legend(); ax[0].set_ylabel("Temp (°C)")
ax[1].plot(slice_df["timestamp"], slice_df["distance_from_route_true"], label="dist_true", linewidth=1,)
ax[1].plot(slice_df["timestamp"], slice_df["distance_from_route_reported"], label="dist_reported", linewidth=1, linestyle="--")
ax[1].legend(); ax[1].set_ylabel("Distance (m)")
st.pyplot(fig)

st.info("You can now train baselines, inspect outputs, download predictions, and explore device traces. Next step: implement the new model (layered/time-aware fusion + attack detector) and integrate its predictions into this dashboard.")
