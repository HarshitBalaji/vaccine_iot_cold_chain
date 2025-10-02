# streamlit_app.py (merged research + comparison app)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Merged Anomaly Detection — Baseline + Proposed")

st.title("Merged: Baselines + Proposed (research-grade)")

# ----------------------- utilities -----------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path, parse_dates=['timestamp','timestamp_reported'])

def plot_confusion_heatmap(cm, labels, title, figsize=(5,4)):
    """
    Compact confusion heatmap: annotate with 'count\n(XX.X%)' at small font to avoid overlap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # compute percentage per true row (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sums>0, (cm.astype(float) / row_sums) * 100.0, 0.0)

    # prepare annotation matrix as strings "count\n(XX.X%)"
    annot = [[f"{int(cm[i,j])}\n({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)

    # tune fonts and layout
    for t in ax.texts:
        t.set_fontsize(8)
        t.set_color('black')

    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    return fig


def compute_detection_latency(df, true_flag_col, pred_flag_col, timestamp_col="timestamp", device_col="Device_ID"):
    events = []
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
                events.append({"Device_ID": dev, "start_time": start_time, "end_time": end_time,
                               "detected": detected, "latency_min": latency_min,
                               "true_len_min": (end_time - start_time).total_seconds()/60.0})
            i += 1
    if len(events)==0:
        return pd.DataFrame(columns=["Device_ID","start_time","end_time","detected","latency_min","true_len_min"])
    return pd.DataFrame(events)

def false_alarms_per_device_day(df, pred_col, true_col, timestamp_col="timestamp", device_col="Device_ID"):
    df_local = df[[device_col, timestamp_col, pred_col, true_col]].copy()
    df_local['date'] = pd.to_datetime(df_local[timestamp_col]).dt.floor('D')
    df_local['fp'] = ((df_local[pred_col]==1) & (df_local[true_col]==0)).astype(int)
    grouped = df_local.groupby([device_col,'date'])['fp'].sum().reset_index()
    avg_fp_per_dev_day = grouped['fp'].mean() if len(grouped)>0 else 0.0
    per_device = grouped.groupby(device_col)['fp'].mean().reset_index().rename(columns={'fp':'avg_fp_per_day'})
    return avg_fp_per_dev_day, per_device

# ----------------------- data & posture -----------------------
DATA_TRAIN = "synthetic_train.csv"
DATA_TEST = "synthetic_test.csv"

st.sidebar.header("Data sources")
use_defaults = st.sidebar.checkbox("Use provided synthetic datasets (train/test)", value=True)
uploaded_train = st.sidebar.file_uploader("Upload train CSV", type="csv")
uploaded_test = st.sidebar.file_uploader("Upload test CSV", type="csv")
vapt_file = st.sidebar.file_uploader("Upload VAPT CSV (Device_ID,vuln_id,cvss,epss)", type="csv")

if use_defaults:
    train_df = load_csv(DATA_TRAIN)
    test_df = load_csv(DATA_TEST)
else:
    if uploaded_train is None or uploaded_test is None:
        st.error("Upload both train and test CSVs or enable defaults.")
        st.stop()
    train_df = load_csv(uploaded_train)
    test_df = load_csv(uploaded_test)

st.sidebar.header("Model controls")
contamination = st.sidebar.slider("IF contamination", 0.001, 0.2, 0.05, 0.001)
rf_thresh = st.sidebar.slider("Risk-Fusion threshold", 0.0, 1.0, 0.7, 0.01)
fusion_model_choice = st.sidebar.selectbox("Fusion model", ["HistGB","MLP"], index=0)
use_cvss_max = st.sidebar.checkbox("Use cvss_max (instead of mean) in fusion", value=True)

# aggregate VAPT if provided
def aggregate_vapt(vapt_df):
    out = []
    for dev, g in vapt_df.groupby("Device_ID"):
        cvss_mean = g["cvss"].mean()
        cvss_max = g["cvss"].max()
        epss_comb = 1 - np.prod(1 - g["epss"].fillna(0).values)
        dvd = g["vuln_id"].nunique()
        out.append({"Device_ID": dev, "cvss_mean": cvss_mean, "cvss_max": cvss_max, "epss_comb": epss_comb, "dvd": dvd})
    return pd.DataFrame(out)

if vapt_file:
    vapt_df = pd.read_csv(vapt_file)
    posture_df = aggregate_vapt(vapt_df)
    st.sidebar.success("VAPT aggregated to device posture")
else:
    # fallback: compute from dataset rows
    grp = train_df.groupby("Device_ID")[["cvss","epss","dvd"]].agg({"cvss":"mean","epss":"mean","dvd":"mean"}).reset_index()
    grp = grp.rename(columns={"cvss":"cvss_mean","epss":"epss_comb","dvd":"dvd"})
    # compute max cvss per device
    grp["cvss_max"] = train_df.groupby("Device_ID")["cvss"].max().values
    posture_df = grp.copy()

st.subheader("Device security posture (sample)")
st.dataframe(posture_df.head(10))

# merge posture into datasets (overwrite cvss/epss/dvd fields)
train_df = train_df.drop(columns=["cvss","epss","dvd"], errors="ignore").merge(posture_df, on="Device_ID", how="left")
test_df = test_df.drop(columns=["cvss","epss","dvd"], errors="ignore").merge(posture_df, on="Device_ID", how="left")

# pick posture feature names for fusion
posture_cvss_col = "cvss_max" if use_cvss_max else "cvss_mean"
train_df["cvss_used"] = train_df[posture_cvss_col].fillna(0)
test_df["cvss_used"] = test_df[posture_cvss_col].fillna(0)
train_df["epss_used"] = train_df["epss_comb"].fillna(0)
test_df["epss_used"] = test_df["epss_comb"].fillna(0)
train_df["dvd_used"] = train_df["dvd"].fillna(0)
test_df["dvd_used"] = test_df["dvd"].fillna(0)

st.markdown("---")
st.subheader("Dataset summary")
c1,c2,c3 = st.columns(3)
with c1: st.metric("Train rows", f"{len(train_df):,}")
with c2: st.metric("Test rows", f"{len(test_df):,}")
with c3: st.metric("Devices (train)", train_df["Device_ID"].nunique())

# ----------------------- Baselines -----------------------
st.header("Baselines")
telemetry_features = ["temp_reported","humidity_reported","distance_from_route_reported"]
fusion_risk_features = telemetry_features + ["cvss_used","epss_used","dvd_used"]

if st.button("Run Baselines (IF + Risk-Fusion)"):
    # prepare telemetry
    scaler_tele = StandardScaler()
    X_train_tele = scaler_tele.fit_transform(train_df[telemetry_features].fillna(0))
    X_test_tele = scaler_tele.transform(test_df[telemetry_features].fillna(0))
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_train_tele)
    test_df["IF_pred"] = np.where(iso.predict(X_test_tele)==-1,1,0)
    # Risk-Fusion (Logistic Regression)
    scaler_f = StandardScaler()
    X_train_f = scaler_f.fit_transform(train_df[fusion_risk_features].fillna(0))
    X_test_f = scaler_f.transform(test_df[fusion_risk_features].fillna(0))
    y_train_bin = (train_df["primary_label_reported"]!="normal").astype(int).values
    y_test_bin = (test_df["primary_label_reported"]!="normal").astype(int).values
    rf_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    rf_clf.fit(X_train_f, y_train_bin)
    rf_probs = rf_clf.predict_proba(X_test_f)[:,1]
    test_df["RF_prob"] = rf_probs
    test_df["RF_pred"] = (rf_probs >= 0.7).astype(int)
    # show metrics & small heatmaps
    st.subheader("Isolation Forest (telemetry-only)")
    st.text(classification_report(y_test_bin, test_df["IF_pred"], zero_division=0))
    cm_if = confusion_matrix(y_test_bin, test_df["IF_pred"])
    fig_if = plot_confusion_heatmap(cm_if, ["Normal","Anomaly"], "IF Confusion (counts + %)", figsize=(5,4))
    st.pyplot(fig_if)
    st.subheader("Risk-Fusion (telemetry + posture)")
    st.text(classification_report(y_test_bin, test_df["RF_pred"], zero_division=0))
    cm_rf = confusion_matrix(y_test_bin, test_df["RF_pred"])
    fig_rf = plot_confusion_heatmap(cm_rf, ["Normal","Anomaly"], "RiskFusion Confusion (counts + %)", figsize=(5,4))
    st.pyplot(fig_rf)
    st.session_state["baseline_test"] = test_df.copy()
    st.success("Baselines complete and stored in session_state.")

# ----------------------- Proposed model (research-grade) -----------------------
st.markdown('---')
st.header("Proposed model — per-feature detectors + cyber detector + fusion")

def engineer_features(df):
    df = df.copy().sort_values(["Device_ID","timestamp"])
    # rolling features
    for w in [3,5]:
        df[f"temp_roll_mean_{w}"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        df[f"temp_roll_std_{w}"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
        df[f"hum_roll_mean_{w}"] = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        df[f"hum_roll_std_{w}"] = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0))
    # deltas
    df["delta_temp"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.diff().fillna(0))
    df["delta_hum"] = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.diff().fillna(0))
    df["delta_dist"] = df.groupby("Device_ID")["distance_from_route_reported"].transform(lambda x: x.diff().fillna(0))
    # gps repeat count
    lat_same = df.groupby("Device_ID")["lat_reported"].transform(lambda x: x.round(6)==x.round(6).shift(1))
    lon_same = df.groupby("Device_ID")["lon_reported"].transform(lambda x: x.round(6)==x.round(6).shift(1))
    df["gps_same"] = (lat_same & lon_same).astype(int)
    df["gps_repeat_count"] = df.groupby("Device_ID")["gps_same"].transform(lambda x: x.groupby((x==0).cumsum()).cumcount()+1).fillna(0).astype(int)
    # reported timestamp diffs (min)
    if "timestamp_reported" in df.columns:
        df["reported_ts_diff_min"] = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: x.diff().dt.total_seconds().div(60).fillna(5))
    else:
        df["reported_ts_diff_min"] = 5
    return df

def train_proposed(train_df):
    df = engineer_features(train_df)
    # temperature detector (GBT)
    temp_feats = ["temp_reported","temp_roll_mean_3","temp_roll_std_3","delta_temp"]
    X_temp = df[temp_feats].fillna(0).values
    y_temp = df["temperature_anomaly_true"].fillna(0).astype(int).values
    temp_clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    temp_clf.fit(X_temp, y_temp)
    # cyberattack detector (GBT)
    attack_feats = ["temp_reported","humidity_reported","temp_roll_std_3","reported_ts_diff_min","gps_repeat_count","distance_from_route_reported","delta_temp","tamper_reported"]
    X_attack = df[attack_feats].fillna(0).values
    y_attack = df["cyberattack_anomaly"].fillna(0).astype(int).values
    attack_clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
    attack_clf.fit(X_attack, y_attack)
    # detector scores
    df["score_temp"] = temp_clf.predict_proba(X_temp)[:,1]
    df["score_attack"] = attack_clf.predict_proba(X_attack)[:,1]
    # fusion features and labels
    fusion_feats = ["score_temp","score_attack","distance_from_route_reported","gps_repeat_count","temp_roll_std_3","cvss_used","epss_used","dvd_used"]
    X_f = df[fusion_feats].fillna(0).values
    le = LabelEncoder()
    y_f = le.fit_transform(df["primary_label_reported"].astype(str).values)
    # sample weights: inverse class frequency * (1 + epss)
    class_counts = pd.Series(y_f).value_counts().to_dict()
    class_weight = {k: (1.0/count) for k,count in class_counts.items()}
    sample_weight = np.array([class_weight[c] for c in y_f])
    sample_weight = sample_weight * (1.0 + df["epss_used"].fillna(0).values)
    # scaler and fusion model (choice)
    scaler_fusion = StandardScaler()
    X_f_scaled = scaler_fusion.fit_transform(X_f)
    if fusion_model_choice == "HistGB":
        fusion_clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
        fusion_clf.fit(X_f_scaled, y_f, sample_weight=sample_weight)
    else:
        fusion_clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
        fusion_clf.fit(X_f_scaled, y_f, sample_weight=sample_weight)
    meta = {"temp_feats": temp_feats, "attack_feats": attack_feats, "fusion_feats": fusion_feats, "le": le}
    return temp_clf, attack_clf, fusion_clf, scaler_fusion, meta

if st.button("Train / Run Proposed (detailed)"):
    temp_clf, attack_clf, fusion_clf, fusion_scaler, meta = train_proposed(train_df)
    st.success("Trained detectors and fusion")

    # Evaluate detectors separately on test set
    test = engineer_features(test_df.copy())
    X_temp_t = test[meta["temp_feats"]].fillna(0).values
    X_attack_t = test[meta["attack_feats"]].fillna(0).values
    # detector scores & binary preds
    test["score_temp"] = temp_clf.predict_proba(X_temp_t)[:,1]
    test["score_attack"] = attack_clf.predict_proba(X_attack_t)[:,1]
    # evaluate temperature detector
    y_temp_true = test["temperature_anomaly_true"].fillna(0).astype(int).values
    temp_pred_bin = (test["score_temp"] >= 0.5).astype(int)
    st.subheader("Temperature detector evaluation (test)")
    st.text(classification_report(y_temp_true, temp_pred_bin, zero_division=0))
    try:
        p,r,_ = precision_recall_curve(y_temp_true, test["score_temp"].fillna(0).values)
        st.write("Temp PR-AUC:", auc(r,p), "Temp ROC-AUC:", roc_auc_score(y_temp_true, test["score_temp"].fillna(0).values))
    except:
        pass
    # evaluate attack detector
    y_attack_true = test["cyberattack_anomaly"].fillna(0).astype(int).values
    attack_pred_bin = (test["score_attack"] >= 0.5).astype(int)
    st.subheader("Cyberattack detector evaluation (test)")
    st.text(classification_report(y_attack_true, attack_pred_bin, zero_division=0))
    try:
        p,r,_ = precision_recall_curve(y_attack_true, test["score_attack"].fillna(0).values)
        st.write("Attack PR-AUC:", auc(r,p), "Attack ROC-AUC:", roc_auc_score(y_attack_true, test["score_attack"].fillna(0).values))
    except:
        pass

    # Fusion inference & evaluation (multi-class)
    X_f_t = test[meta["fusion_feats"]].fillna(0).values
    X_f_t_scaled = fusion_scaler.transform(X_f_t)
    codes = fusion_clf.predict(X_f_t_scaled)
    test["fusion_code"] = codes
    test["final_label"] = meta["le"].inverse_transform(codes)
    st.subheader("Fusion multi-class evaluation (test)")
    st.text(classification_report(test["primary_label_reported"].astype(str).values, test["final_label"].astype(str).values, zero_division=0))
    labels_union = np.unique(np.concatenate([test["primary_label_reported"].astype(str).values, test["final_label"].astype(str).values]))
    cm = confusion_matrix(test["primary_label_reported"].astype(str).values, test["final_label"].astype(str).values, labels=labels_union)
    fig_cm = plot_confusion_heatmap(cm, labels_union, "Fusion confusion (counts + %)", figsize=(6,5))
    st.pyplot(fig_cm)

    # Operational metrics: detection latency and false positives per device-day
    test["temp_pred_bin"] = temp_pred_bin
    test["attack_pred_bin"] = attack_pred_bin
    latency_temp = compute_detection_latency(test, "temperature_anomaly_true", "temp_pred_bin")
    latency_attack = compute_detection_latency(test, "cyberattack_anomaly", "attack_pred_bin")
    st.subheader("Operational metrics")
    if not latency_temp.empty:
        st.write("Temp detection latency median (min):", float(np.nanmedian(latency_temp["latency_min"])))
    if not latency_attack.empty:
        st.write("Attack detection latency median (min):", float(np.nanmedian(latency_attack["latency_min"])))
    avg_fp_temp, per_dev_temp = false_alarms_per_device_day(test, "temp_pred_bin", "temperature_anomaly_true")
    avg_fp_attack, per_dev_attack = false_alarms_per_device_day(test, "attack_pred_bin", "cyberattack_anomaly")
    st.write(f"Avg false positives per device-day (temp): {avg_fp_temp:.3f}, (attack): {avg_fp_attack:.3f}")
    st.session_state["proposed_test"] = test.copy()
    st.success("Proposed model inference & metrics stored in session_state.")

# ----------------------- per-device inspector & downloads -----------------------
st.markdown('---')
st.header("Inspector & downloads")
dev_list = train_df["Device_ID"].unique().tolist()
sel_dev = st.selectbox("Select device", dev_list)
if "proposed_test" in st.session_state:
    dev_df = st.session_state["proposed_test"][st.session_state["proposed_test"]["Device_ID"]==sel_dev].sort_values("timestamp")
else:
    dev_df = test_df[test_df["Device_ID"]==sel_dev].sort_values("timestamp")
st.dataframe(dev_df.tail(12))
# download
if st.button("Download latest proposed predictions CSV"):
    if "proposed_test" in st.session_state:
        st.download_button("Download proposed predictions", st.session_state["proposed_test"].to_csv(index=False).encode("utf-8"), "proposed_predictions.csv", "text/csv")
    else:
        st.info("No proposed output yet. Train & run proposed model first.")

if st.button("Download posture table (CSV)"):
    st.download_button("Posture CSV", posture_df.to_csv(index=False).encode("utf-8"), "posture.csv", "text/csv")

st.info("Merged app: baselines + research-focused proposed pipeline. Use the sidebar to tune options. Per-detector metrics, fusion multi-class, detection latency and FP/device-day are reported to help diagnose and iterate.")
