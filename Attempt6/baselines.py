# streamlit_app.py — Baselines only (IF + Risk Fusion) with PR/ROC metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Baseline Anomaly Detection — IF + Risk Fusion")

st.title("Baselines: Isolation Forest & Risk Fusion (with PR-AUC / ROC-AUC)")

# ----------------------- utilities -----------------------
@st.cache_data
def load_csv(path_or_file):
    return pd.read_csv(path_or_file, parse_dates=['timestamp','timestamp_reported'])

def plot_confusion_heatmap(cm, labels, title, figsize=(5,4)):
    fig, ax = plt.subplots(figsize=figsize)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sums>0, (cm.astype(float) / row_sums) * 100.0, 0.0)
    annot = [[f"{int(cm[i,j])}\n({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    for t in ax.texts:
        t.set_fontsize(8); t.set_color('black')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    return fig

def pr_auc_score(y_true, y_score):
    p, r, _ = precision_recall_curve(y_true, y_score)
    return auc(r, p), (r, p), (p, r)

def plot_pr_curve(r, p, title="Precision–Recall Curve"):
    fig, ax = plt.subplots()
    ax.plot(r, p, lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title)
    st.pyplot(fig)

def plot_roc_curve(y_true, y_score, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc:.3f}")
    ax.plot([0,1],[0,1],'--',lw=1,color='grey')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    st.pyplot(fig)

def false_alarms_per_device_day(df, pred_col, true_col, timestamp_col="timestamp", device_col="Device_ID"):
    df_local = df[[device_col, timestamp_col, pred_col, true_col]].copy()
    df_local['date'] = pd.to_datetime(df_local[timestamp_col]).dt.floor('D')
    df_local['fp'] = ((df_local[pred_col]==1) & (df_local[true_col]==0)).astype(int)
    grouped = df_local.groupby([device_col,'date'])['fp'].sum().reset_index()
    avg_fp_per_dev_day = grouped['fp'].mean() if len(grouped)>0 else 0.0
    per_device = grouped.groupby(device_col)['fp'].mean().reset_index().rename(columns={'fp':'avg_fp_per_day'})
    return avg_fp_per_dev_day, per_device

# ----------------------- data & posture -----------------------
st.sidebar.header("Data sources")
use_defaults = st.sidebar.checkbox("Use provided synthetic datasets (train/test)", value=True)
uploaded_train = st.sidebar.file_uploader("Upload train CSV", type="csv")
uploaded_test  = st.sidebar.file_uploader("Upload test CSV",  type="csv")
vapt_file      = st.sidebar.file_uploader("Upload VAPT CSV (Device_ID,vuln_id,cvss,epss)", type="csv")

DEFAULT_TRAIN = "synthetic_train_better_journey_dev_overlap_delay.csv"
DEFAULT_TEST  = "synthetic_test_better_journey_overlap_delay.csv"

if use_defaults:
    train_df = load_csv(DEFAULT_TRAIN)
    test_df  = load_csv(DEFAULT_TEST)
else:
    if uploaded_train is None or uploaded_test is None:
        st.error("Upload both train and test CSVs or enable defaults.")
        st.stop()
    train_df = load_csv(uploaded_train)
    test_df  = load_csv(uploaded_test)

# posture aggregation (if VAPT uploaded, else derive from train)
def aggregate_vapt(vapt_df):
    out = []
    for dev, g in vapt_df.groupby("Device_ID"):
        cvss_mean = g["cvss"].mean()
        cvss_max  = g["cvss"].max()
        epss_comb = 1 - np.prod(1 - g["epss"].fillna(0).values)
        dvd       = g["vuln_id"].nunique()
        out.append({"Device_ID": dev, "cvss_mean": cvss_mean, "cvss_max": cvss_max,
                    "epss_comb": epss_comb, "dvd": dvd})
    return pd.DataFrame(out)

if vapt_file:
    vapt_df   = pd.read_csv(vapt_file)
    posture_df= aggregate_vapt(vapt_df)
    st.sidebar.success("VAPT aggregated to device posture")
else:
    grp = train_df.groupby("Device_ID")[["cvss","epss","dvd"]].agg({"cvss":"mean","epss":"mean","dvd":"mean"}).reset_index()
    grp = grp.rename(columns={"cvss":"cvss_mean","epss":"epss_comb","dvd":"dvd"})
    grp["cvss_max"] = train_df.groupby("Device_ID")["cvss"].max().values
    posture_df = grp.copy()

use_cvss_max = st.sidebar.checkbox("Use cvss_max (instead of mean) in features", value=True)
posture_cvss_col = "cvss_max" if use_cvss_max else "cvss_mean"

# merge posture into data
for df in (train_df, test_df):
    df.drop(columns=["cvss","epss","dvd"], errors="ignore", inplace=True)
    df.merge(posture_df, on="Device_ID", how="left", copy=False)
train_df = train_df.merge(posture_df, on="Device_ID", how="left")
test_df  = test_df.merge(posture_df, on="Device_ID", how="left")

# convenience columns used in RF
for df in (train_df, test_df):
    df["cvss_used"] = df[posture_cvss_col].fillna(0)
    df["epss_used"] = df["epss_comb"].fillna(0)
    df["dvd_used"]  = df["dvd"].fillna(0)

st.subheader("Dataset summary")
c1,c2,c3 = st.columns(3)
c1.metric("Train rows", f"{len(train_df):,}")
c2.metric("Test rows",  f"{len(test_df):,}")
c3.metric("Devices (train)", train_df["Device_ID"].nunique())

st.subheader("Device security posture (sample)")
st.dataframe(posture_df.head(10))

# ----------------------- baseline controls -----------------------
st.sidebar.header("Baseline controls")
contamination = st.sidebar.slider("Isolation Forest contamination", 0.001, 0.2, 0.05, 0.001)
rf_thresh     = st.sidebar.slider("Risk-Fusion decision threshold", 0.0, 1.0, 0.70, 0.01)

telemetry_features     = ["temp_reported","humidity_reported","distance_from_route_reported"]
fusion_risk_features   = telemetry_features + ["cvss_used","epss_used","dvd_used"]

st.markdown("---")
st.header("Run Baselines")

if st.button("Run IF + Risk Fusion"):
    # Binary ground truth: anomaly = not 'normal'
    y_train_bin = (train_df["primary_label_reported"]!="normal").astype(int).values
    y_test_bin  = (test_df["primary_label_reported"]!="normal").astype(int).values

    # -------- Isolation Forest (telemetry-only) --------
    scaler_tele   = StandardScaler()
    X_train_tele  = scaler_tele.fit_transform(train_df[telemetry_features].fillna(0))
    X_test_tele   = scaler_tele.transform(test_df[telemetry_features].fillna(0))

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_train_tele)

    # Predictions (−1 => anomaly) and scores
    if_pred      = np.where(iso.predict(X_test_tele)==-1, 1, 0)
    # IsolationForest score_samples: higher => more normal; flip sign so higher => more anomalous
    if_scores    = -iso.score_samples(X_test_tele)

    # -------- Risk Fusion (telemetry + posture) --------
    scaler_f     = StandardScaler()
    X_train_f    = scaler_f.fit_transform(train_df[fusion_risk_features].fillna(0))
    X_test_f     = scaler_f.transform(test_df[fusion_risk_features].fillna(0))

    rf_clf       = LogisticRegression(max_iter=1000, class_weight="balanced")
    rf_clf.fit(X_train_f, y_train_bin)
    rf_probs     = rf_clf.predict_proba(X_test_f)[:,1]
    rf_pred      = (rf_probs >= rf_thresh).astype(int)

    # -------- Reports: IF --------
    st.subheader("Isolation Forest (telemetry-only)")
    st.code(classification_report(y_test_bin, if_pred, target_names=["Normal","Anomaly"], digits=3), language="text")
    cm_if = confusion_matrix(y_test_bin, if_pred, labels=[0,1])
    st.pyplot(plot_confusion_heatmap(cm_if, ["Normal","Anomaly"], "IF — Confusion (counts + row%)"))

    pr_if, (r_if, p_if), _ = pr_auc_score(y_test_bin, if_scores)
    roc_if = roc_auc_score(y_test_bin, if_scores)
    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC (IF)", f"{pr_if:.3f}")
    c2.metric("ROC-AUC (IF)", f"{roc_if:.3f}")
    c3.metric("Anomaly prevalence", f"{(y_test_bin.mean()):.3f}")
    plot_pr_curve(r_if, p_if, "IF — Precision–Recall")
    plot_roc_curve(y_test_bin, if_scores, "IF — ROC")

    st.markdown("---")

    # -------- Reports: Risk Fusion --------
    st.subheader("Risk Fusion (telemetry + posture)")
    st.code(classification_report(y_test_bin, rf_pred, target_names=["Normal","Anomaly"], digits=3), language="text")
    cm_rf = confusion_matrix(y_test_bin, rf_pred, labels=[0,1])
    st.pyplot(plot_confusion_heatmap(cm_rf, ["Normal","Anomaly"], "RF — Confusion (counts + row%)"))

    pr_rf, (r_rf, p_rf), _ = pr_auc_score(y_test_bin, rf_probs)
    roc_rf = roc_auc_score(y_test_bin, rf_probs)
    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC (RF)", f"{pr_rf:.3f}")
    c2.metric("ROC-AUC (RF)", f"{roc_rf:.3f}")
    c3.metric("Decision threshold", f"{rf_thresh:.2f}")
    plot_pr_curve(r_rf, p_rf, "RF — Precision–Recall")
    plot_roc_curve(y_test_bin, rf_probs, "RF — ROC")

    # -------- Operational extras --------
    st.markdown("---")
    st.subheader("Operational summaries")
    test_copy = test_df.copy()
    test_copy["IF_pred"] = if_pred
    test_copy["RF_pred"] = rf_pred
    avg_fp_if, _ = false_alarms_per_device_day(test_copy, "IF_pred", "primary_label_reported")
    avg_fp_rf, _ = false_alarms_per_device_day(test_copy, "RF_pred", "primary_label_reported")
    st.write(f"Average false positives per device-day — IF: **{avg_fp_if:.3f}**, RF: **{avg_fp_rf:.3f}**")

    # store for download
    test_copy["IF_score"] = if_scores
    test_copy["RF_prob"]  = rf_probs
    st.session_state["baseline_outputs"] = test_copy

# ----------------------- downloads -----------------------
st.markdown('---')
st.header("Downloads")
if "baseline_outputs" in st.session_state:
    st.download_button(
        "Download baseline predictions (CSV)",
        st.session_state["baseline_outputs"].to_csv(index=False).encode("utf-8"),
        file_name="baseline_outputs.csv",
        mime="text/csv"
    )
else:
    st.info("Run the baselines to enable downloads.")
