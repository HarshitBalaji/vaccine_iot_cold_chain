#!/usr/bin/env python3
# streamlit run app.py
# ----------------------
# Cyber Risk Dashboard (Streamlit)
# Visualizes outputs from independent_sup_timeseriescal_riskaware.py
# Files expected (default names; can also be uploaded via UI):
#   - test_predictions_sup_riskaware.csv
#   - risk_dashboard_sup_riskaware.csv
#   - security_posture_update.xlsx (sheets: Initial_Posture, Final_Posture_Summary)
#   - meta_sup_riskaware.json (optional; thresholds & feature names)
#   - VAPT.csv (optional; initial posture source)
# ---------------------------------------------------------------

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Cyber Risk Dashboard", layout="wide")
st.title("🛡️ Cyber Risk Dashboard")
st.caption("Streamlit app for viewing cyber + operational anomalies, risk-aware actions, and posture updates")

# ----------------------
# Utilities
# ----------------------
@st.cache_data(show_spinner=False)
def _read_csv(file_or_bytes):
    if file_or_bytes is None:
        return None
    if isinstance(file_or_bytes, (str, bytes, io.BytesIO)):
        return pd.read_csv(file_or_bytes)
    # UploadedFile
    return pd.read_csv(file_or_bytes)

@st.cache_data(show_spinner=False)
def _read_excel(file_or_bytes, sheet_name=None):
    if file_or_bytes is None:
        return None
    return pd.read_excel(file_or_bytes, sheet_name=sheet_name)

@st.cache_data(show_spinner=False)
def _read_json(file_or_bytes):
    if file_or_bytes is None:
        return None
    if hasattr(file_or_bytes, 'read'):
        return json.load(file_or_bytes)
    with open(file_or_bytes, 'r') as f:
        return json.load(f)

# Try to load defaults from working directory
@st.cache_data(show_spinner=False)
def load_defaults():
    out = {}
    try:
        out['pred'] = pd.read_csv('test_predictions_sup_riskaware.csv')
    except Exception:
        out['pred'] = None
    try:
        out['risk'] = pd.read_csv('risk_dashboard_sup_riskaware.csv')
    except Exception:
        out['risk'] = None
    try:
        out['posture_xlsx'] = pd.read_excel('security_posture_update.xlsx', sheet_name=None)
    except Exception:
        out['posture_xlsx'] = None
    try:
        with open('meta_sup_riskaware.json','r') as f:
            out['meta'] = json.load(f)
    except Exception:
        out['meta'] = None
    try:
        out['vapt'] = pd.read_csv('VAPT.csv')
    except Exception:
        out['vapt'] = None
    return out

# ----------------------
# Sidebar IO
# ----------------------
with st.sidebar:
    st.header("📥 Data Sources")
    st.write("Upload files or rely on defaults in the working directory.")
    up_pred = st.file_uploader("test_predictions_sup_riskaware.csv", type=['csv'])
    up_risk = st.file_uploader("risk_dashboard_sup_riskaware.csv", type=['csv'])
    up_meta = st.file_uploader("meta_sup_riskaware.json", type=['json'])
    up_posture = st.file_uploader("security_posture_update.xlsx", type=['xlsx'])
    up_vapt = st.file_uploader("VAPT.csv (optional)", type=['csv'])

# Load data
defaults = load_defaults()
pred = _read_csv(up_pred) if up_pred else defaults['pred']
risk = _read_csv(up_risk) if up_risk else defaults['risk']
meta = _read_json(up_meta) if up_meta else defaults['meta']
posture_xlsx = _read_excel(up_posture, sheet_name=None) if up_posture else defaults['posture_xlsx']
vapt_df = _read_csv(up_vapt) if up_vapt else defaults['vapt']

if pred is None:
    st.warning("Please upload **test_predictions_sup_riskaware.csv** or place it in the working directory.")
    st.stop()

# Ensure expected columns exist
expected_cols = {
    'Device_ID', 'timestamp', 'primary_label_reported',
    'operational_label', 'op_best_score',
    'score_attack_sup', 'pred_attack_bin', 'recommended_action'
}
missing = expected_cols - set(pred.columns)
if missing:
    st.error(f"Missing expected columns in predictions: {missing}")
    st.stop()

# Parse timestamps
pred['timestamp'] = pd.to_datetime(pred['timestamp'], errors='coerce')

# Basic derived fields
pred['date'] = pred['timestamp'].dt.date
pred['is_cyber_true'] = (pred['primary_label_reported'] == 'Cyberattack_anomaly').astype(int)

# ----------------------
# KPIs / Overview
# ----------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Devices", f"{pred['Device_ID'].nunique():,}")
col2.metric("Rows", f"{len(pred):,}")
cy_rate = pred['is_cyber_true'].mean() if len(pred)>0 else 0
col3.metric("Cyber prevalence", f"{cy_rate*100:.2f}%")
alerts = int(pred['pred_attack_bin'].sum())
col4.metric("Cyber alerts (pred)", f"{alerts:,}")
if meta and isinstance(meta, dict):
    col5.metric("Threshold (sup base)", f"{meta.get('thresholds',{}).get('attack_sup_base','—')}")
else:
    col5.metric("Threshold (sup base)", "—")

# Optional AUCs from latest run (compute on the fly)
try:
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
    p, r, _ = precision_recall_curve(pred['is_cyber_true'], pred['score_attack_sup'])
    pr_auc = auc(r, p)
    roc_auc = roc_auc_score(pred['is_cyber_true'], pred['score_attack_sup']) if pred['is_cyber_true'].nunique()>1 else float('nan')
    st.success(f"**Cyber SUP**: PR-AUC **{pr_auc:.3f}**, ROC-AUC **{roc_auc:.3f}**")
except Exception as e:
    st.info(f"AUCs unavailable: {e}")

st.markdown("---")

# ----------------------
# Tabs
# ----------------------
T1, T2, T3, T4, T5 = st.tabs([
    "📈 Time Series",
    "🧭 Device View",
    "📊 Risk Dashboard",
    "🧱 Security Posture",
    "📜 Metrics"
])

# -------------- Tab 1: Time Series (aggregate) --------------
with T1:
    st.subheader("Cyber score & flags over time (aggregate)")
    # Option to filter timeframe and devices
    devs = sorted(pred['Device_ID'].astype(str).unique())
    sel_devs = st.multiselect("Devices", devs, default=devs[: min(10, len(devs))])
    df = pred[pred['Device_ID'].astype(str).isin(sel_devs)].copy()
    # Aggregate by minute to avoid rendering too many points
    df_agg = (df
        .set_index('timestamp')
        .groupby('Device_ID')
        .resample('5min')
        .agg(score_attack_sup=('score_attack_sup','mean'),
             pred_attack_bin=('pred_attack_bin','max'),
             is_cyber_true=('is_cyber_true','max'))
        .reset_index())
    if df_agg.empty:
        st.info("No data in selected window/devices.")
    else:
        fig = px.line(df_agg, x='timestamp', y='score_attack_sup', color='Device_ID', title='Cyber probability (mean over 5-min bins)')
        st.plotly_chart(fig, use_container_width=True)
        # Flags scatter
        flags = df_agg[df_agg['pred_attack_bin']>0]
        if not flags.empty:
            fig2 = px.scatter(flags, x='timestamp', y='score_attack_sup', color='Device_ID', symbol='pred_attack_bin', title='Predicted cyber flags')
            st.plotly_chart(fig2, use_container_width=True)

# -------------- Tab 2: Device View --------------
with T2:
    st.subheader("Per-device timeline")
    dsel = st.selectbox("Select device", sorted(pred['Device_ID'].astype(str).unique()))
    ddf = pred[pred['Device_ID'].astype(str)==str(dsel)].sort_values('timestamp').copy()
    c1, c2 = st.columns([3,1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ddf['timestamp'], y=ddf['score_attack_sup'], name='cyber_prob', mode='lines'))
        # predicted flags as markers
        mask = ddf['pred_attack_bin']>0
        fig.add_trace(go.Scatter(x=ddf.loc[mask,'timestamp'], y=ddf.loc[mask,'score_attack_sup'],
                                 mode='markers', name='pred_flag', marker=dict(size=8)))
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("**Counts**")
        st.metric("Pred cyber flags", int(ddf['pred_attack_bin'].sum()))
        st.metric("True cyber", int(ddf['is_cyber_true'].sum()))
        st.metric("Records", len(ddf))
        st.write("**Top operational label**")
        top_op = ddf['operational_label'].value_counts().head(3)
        st.dataframe(top_op)

    st.write("**Operational label timeline (stacked counts per hour)**")
    ddf['hour'] = ddf['timestamp'].dt.floor('H')
    op_pivot = ddf.pivot_table(index='hour', columns='operational_label', values='Device_ID', aggfunc='count').fillna(0)
    op_pivot = op_pivot.reset_index().melt(id_vars='hour', var_name='operational_label', value_name='count')
    fig3 = px.area(op_pivot, x='hour', y='count', color='operational_label')
    st.plotly_chart(fig3, use_container_width=True)

# -------------- Tab 3: Risk Dashboard --------------
with T3:
    st.subheader("Per device-day risk dashboard")
    if risk is None:
        st.info("Upload risk_dashboard_sup_riskaware.csv to see this view.")
    else:
        # Filters
        r = risk.copy()
        r['date'] = pd.to_datetime(r['date'], errors='coerce')
        mind, maxd = r['date'].min(), r['date'].max()
        dr = st.date_input("Date range", value=(mind.date(), maxd.date()), min_value=mind.date(), max_value=maxd.date())
        if isinstance(dr, tuple) and len(dr)==2:
            r = r[(r['date']>=pd.Timestamp(dr[0])) & (r['date']<=pd.Timestamp(dr[1]))]
        devs_sel = st.multiselect("Devices", sorted(r['Device_ID'].astype(str).unique()), default=None)
        if devs_sel:
            r = r[r['Device_ID'].astype(str).isin(devs_sel)]
        st.dataframe(r, use_container_width=True, height=350)
        # Simple bar: days with cyber per device
        agg = r.groupby('Device_ID')['has_cyber'].sum().reset_index().rename(columns={'has_cyber':'cyber_days'})
        fig = px.bar(agg, x='Device_ID', y='cyber_days', title='Days with predicted cyber per device')
        st.plotly_chart(fig, use_container_width=True)

# -------------- Tab 4: Security Posture --------------
with T4:
    st.subheader("Security posture — initial vs final")
    if posture_xlsx is None:
        st.info("Upload security_posture_update.xlsx to see posture update summary.")
    else:
        init_df = posture_xlsx.get('Initial_Posture', None)
        final_df = posture_xlsx.get('Final_Posture_Summary', None)
        if init_df is None or final_df is None:
            st.error("Excel must contain sheets: Initial_Posture and Final_Posture_Summary")
        else:
            # Ensure Device_ID str
            init_df['Device_ID'] = init_df['Device_ID'].astype(str)
            final_df['Device_ID'] = final_df['Device_ID'].astype(str)
            # Merge initial + final for display
            show_cols = ['Device_ID','cvss','epss','dvd','cvss_final','epss_final','dvd_final','cyber_days','op_days','samples']
            inter = final_df[show_cols].copy()
            inter['Δcvss'] = inter['cvss_final'] - inter['cvss']
            inter['Δepss'] = inter['epss_final'] - inter['epss']
            inter['Δdvd']  = inter['dvd_final']  - inter['dvd']
            st.dataframe(inter.sort_values(['Δepss','Δdvd','Δcvss'], ascending=False), use_container_width=True, height=360)
            # Bar charts for deltas
            st.write("**Top posture increases**")
            topk = st.slider("Top K", 5, 50, 10)
            for col, title in [("Δepss","Δ EPSS"),("Δcvss","Δ CVSS"),("Δdvd","Δ DVD")]:
                top = inter.nlargest(topk, col)
                fig = px.bar(top, x='Device_ID', y=col, title=title)
                st.plotly_chart(fig, use_container_width=True)

# -------------- Tab 5: Metrics --------------
with T5:
    st.subheader("Classification metrics")
    from sklearn.metrics import classification_report, confusion_matrix

    # Operational multi-class
    def to_operational(lbl: str):
        return lbl if lbl in ("Geofence_anomaly","Loss_of_storage_condition","Tamper_Damage_anomaly") else "normal"
    op_true = pred['primary_label_reported'].astype(str).apply(to_operational)
    op_pred = pred['operational_label'].astype(str)
    st.write("**Operational (multi-class)**")
    st.text(classification_report(op_true, op_pred, zero_division=0, digits=3))

    # Cyber binary
    y_true = (pred['primary_label_reported'] == 'Cyberattack_anomaly').astype(int).to_numpy()
    y_pred = pred['pred_attack_bin'].fillna(0).astype(int).clip(0,1).to_numpy()

    st.write("**Cyber (binary)**")
    st.text(classification_report(y_true, y_pred, target_names=["normal","Cyberattack_anomaly"], zero_division=0, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["true_normal","true_cyber"], columns=["pred_normal","pred_cyber"])
    fig = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Downloads
# ----------------------
st.markdown("---")
st.subheader("⬇️ Downloads")
colA, colB, colC, colD = st.columns(4)
with colA:
    try:
        st.download_button("Predictions CSV", data=pred.to_csv(index=False).encode('utf-8'), file_name='test_predictions_sup_riskaware.csv', mime='text/csv')
    except Exception:
        pass
with colB:
    if risk is not None:
        st.download_button("Risk Dashboard CSV", data=risk.to_csv(index=False).encode('utf-8'), file_name='risk_dashboard_sup_riskaware.csv', mime='text/csv')
with colC:
    if posture_xlsx is not None:
        # Rebuild Excel in-memory for download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            posture_xlsx['Initial_Posture'].to_excel(writer, index=False, sheet_name='Initial_Posture')
            posture_xlsx['Final_Posture_Summary'].to_excel(writer, index=False, sheet_name='Final_Posture_Summary')
        st.download_button("Security Posture Excel", data=buf.getvalue(), file_name='security_posture_update.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
with colD:
    if meta is not None:
        st.download_button("Meta JSON", data=json.dumps(meta, indent=2).encode('utf-8'), file_name='meta_sup_riskaware.json', mime='application/json')

st.caption("Built for telemetry-only cyber/ops anomaly monitoring with risk-aware decisions and posture updates.")
