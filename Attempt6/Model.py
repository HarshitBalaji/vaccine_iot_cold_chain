#!/usr/bin/env python3
import os, json, warnings, math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

PERSISTENCE = {"temp":4, "geo":4, "tamper":1, "attack":2}

def _parse_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)

def pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)

def read_df(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    need = {"Device_ID","timestamp","primary_label_reported"}
    if not need.issubset(df.columns): raise ValueError(f"CSV must contain {need}")
    df["Device_ID"] = df["Device_ID"].astype(str).str.strip()
    df["timestamp"] = _parse_dt(df["timestamp"])
    if "timestamp_reported" in df.columns:
        df["timestamp_reported"] = _parse_dt(df["timestamp_reported"])
    else:
        df["timestamp_reported"] = pd.NaT
    for c in ["cvss","epss","dvd"]:
        if c not in df.columns: df[c] = 0.0
    for c in ["temp_reported","humidity_reported","distance_from_route_reported","lat_reported","lon_reported"]:
        if c not in df.columns: df[c] = 0.0
    if "tamper_reported" not in df.columns: df["tamper_reported"] = 0
    if "is_augmented" not in df.columns: df["is_augmented"] = 0
    return df

def report_feature_availability(df):
    cols = set(df.columns)
    gps_ok = {"lat_reported","lon_reported"}.issubset(cols)
    tsr_ok = df["timestamp_reported"].notna().any()
    print(f"Feature availability → GPS: {gps_ok} | timestamp_reported: {tsr_ok} | tamper_reported: {'tamper_reported' in cols}")

# Haversine distance in meters
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p = np.pi/180.0
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def engineer_features(df):
    df = df.sort_values(["Device_ID","timestamp"]).copy()
    # base time features
    df["delay_min"] = (df["timestamp"] - df["timestamp_reported"]).dt.total_seconds().div(60)
    df["rpt_gap_min"] = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: x.diff().dt.total_seconds().div(60))
    df["rcv_gap_min"] = df.groupby("Device_ID")["timestamp"].transform(lambda x: x.diff().dt.total_seconds().div(60))
    df["gap_err_min"] = (df["rpt_gap_min"] - df["rcv_gap_min"]).fillna(0)

    df["delay_med_1h"] = df.groupby("Device_ID")["delay_min"].transform(lambda x: x.rolling(12, min_periods=3).median())
    df["delay_std_1h"] = df.groupby("Device_ID")["delay_min"].transform(lambda x: x.rolling(12, min_periods=3).std())
    df["delay_z"] = (df["delay_min"] - df["delay_med_1h"]) / (df["delay_std_1h"] + 1e-6)
    df["delay_big"] = (df["delay_min"] >= 5).astype(int)

    def _cv(a):
        a = np.asarray(a, dtype=float)
        return (np.nanstd(a)+1e-6)/(np.nanmean(a)+1e-6)
    df["rpt_gap_cv_5"] = df.groupby("Device_ID")["rpt_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).apply(_cv, raw=True)).fillna(0)
    df["rcv_gap_cv_5"] = df.groupby("Device_ID")["rcv_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).apply(_cv, raw=True)).fillna(0)

    df["rpt_ooo"] = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: (x < x.shift(1))).fillna(False).astype(int)

    # GPS freeze / replay-like
    if {"lat_reported","lon_reported"}.issubset(df.columns):
        lat_same = df.groupby("Device_ID")["lat_reported"].transform(lambda s: s.round(5).eq(s.round(5).shift(1)))
        lon_same = df.groupby("Device_ID")["lon_reported"].transform(lambda s: s.round(5).eq(s.round(5).shift(1)))
        df["gps_same"] = (lat_same & lon_same).fillna(False).astype(int)
        def _consec_ones(s):
            run, out = 0, []
            for v in s.astype(int).tolist():
                run = run + 1 if v == 1 else 0
                out.append(run)
            return pd.Series(out, index=s.index, dtype="int64")
        df["gps_repeat_count"] = df.groupby("Device_ID")["gps_same"].transform(_consec_ones)
        df["gps_repeat_cap10"] = np.minimum(df["gps_repeat_count"], 10).astype(int)
    else:
        df["gps_same"] = 0; df["gps_repeat_count"] = 0; df["gps_repeat_cap10"] = 0

    # Sensor rolling stats
    for w in [3,5,7]:
        df[f"temp_roll_mean_{w}"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"temp_roll_std_{w}"]  = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
        df[f"hum_roll_mean_{w}"]  = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"hum_roll_std_{w}"]   = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
        df[f"dist_roll_max_{w}"]  = df.groupby("Device_ID")["distance_from_route_reported"].transform(lambda x: x.rolling(w, min_periods=1).max())

    df["delta_temp"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.diff()).fillna(0)
    df["delta_hum"]  = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.diff()).fillna(0)
    df["delta_dist"] = df.groupby("Device_ID")["distance_from_route_reported"].transform(lambda x: x.diff()).fillna(0)

    def _ewma_resid(x, span=5):
        m = x.ewm(span=span, adjust=False).mean()
        return (x - m)
    df["temp_resid"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: _ewma_resid(x, span=5)).fillna(0)
    df["hum_resid"]  = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: _ewma_resid(x, span=5)).fillna(0)

    for col, new in [("temp_reported","temp_var_5"), ("humidity_reported","hum_var_5")]:
        df[new] = df.groupby("Device_ID")[col].transform(lambda x: x.rolling(5, min_periods=2).var()).fillna(0)

    def _zcr_roll(series, win=5):
        s = series.fillna(0).values
        sign = np.sign(s)
        zc = (np.roll(sign, 1) != sign).astype(int); zc[0] = 0
        return pd.Series(zc).rolling(win, min_periods=2).mean().values
    df["zcr_temp_5"] = df.groupby("Device_ID")["delta_temp"].transform(_zcr_roll)
    df["zcr_hum_5"]  = df.groupby("Device_ID")["delta_hum"].transform(_zcr_roll)

    df["temp_resid_std_ratio"] = df["temp_resid"] / (df["temp_roll_std_3"] + 1e-6)
    df["hum_resid_std_ratio"]  = df["hum_resid"]  / (df["hum_roll_std_3"]  + 1e-6)

    # --------- Kinematic features from GPS & timestamp_reported ---------
    if {"lat_reported","lon_reported"}.issubset(df.columns):
        # compute speed using reported time (device perspective)
        lat = df["lat_reported"].astype(float)
        lon = df["lon_reported"].astype(float)
        t = df["timestamp_reported"]
        dt = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: x.diff().dt.total_seconds()).fillna(0).values
        # haversine distance to previous point per device
        lat_prev = df.groupby("Device_ID")["lat_reported"].shift(1).astype(float)
        lon_prev = df.groupby("Device_ID")["lon_reported"].shift(1).astype(float)
        d_m = haversine_m(lat_prev.values, lon_prev.values, lat.values, lon.values)
        d_m[np.isnan(d_m)] = 0.0
        speed = np.divide(d_m, np.maximum(dt, 1.0), where=np.isfinite(d_m))
        speed[~np.isfinite(speed)] = 0.0
        df["speed_mps"] = speed
        # accel & jerk
        accel = df.groupby("Device_ID")["speed_mps"].diff() / np.maximum(df.groupby("Device_ID")["timestamp_reported"].diff().dt.total_seconds(), 1.0)
        df["accel_mps2"] = accel.fillna(0)
        df["jerk_mps3"]  = df.groupby("Device_ID")["accel_mps2"].diff().fillna(0)
        # flags
        df["teleport_flag"] = (df["speed_mps"] > 60.0).astype(int)  # >216 km/h unlikely for your devices
        # rolling stats
        for w in [5, 12]:
            df[f"speed_mean_{w}"] = df.groupby("Device_ID")["speed_mps"].transform(lambda x: x.rolling(w, min_periods=2).mean()).fillna(0)
            df[f"speed_std_{w}"]  = df.groupby("Device_ID")["speed_mps"].transform(lambda x: x.rolling(w, min_periods=2).std()).fillna(0)
            df[f"accel_std_{w}"]  = df.groupby("Device_ID")["accel_mps2"].transform(lambda x: x.rolling(w, min_periods=2).std()).fillna(0)
    else:
        df["speed_mps"]=0; df["accel_mps2"]=0; df["jerk_mps3"]=0; df["teleport_flag"]=0
        df["speed_mean_5"]=0; df["speed_std_5"]=0; df["accel_std_5"]=0
        df["speed_mean_12"]=0; df["speed_std_12"]=0; df["accel_std_12"]=0

    # --------- Replay/duplication features ---------
    # near-duplicate content ratio in last 5/10 frames
    key_tuple = (
        df["lat_reported"].round(5).astype(str) + "|" +
        df["lon_reported"].round(5).astype(str) + "|" +
        df["temp_reported"].round(1).astype(str) + "|" +
        df["humidity_reported"].round(1).astype(str)
    )
    df["dup_flag"] = (key_tuple == key_tuple.shift(1)).astype(int)
    for w in [5, 10]:
        df[f"dup_ratio_{w}"] = df.groupby("Device_ID")["dup_flag"].transform(lambda x: x.rolling(w, min_periods=2).mean()).fillna(0)

    return df

def add_binary_targets(df):
    lbl = df["primary_label_reported"].astype(str)
    df["y_temp"]   = (lbl=="Loss_of_storage_condition").astype(int)
    df["y_geo"]    = (lbl=="Geofence_anomaly").astype(int)
    df["y_tamper"] = (lbl=="Tamper_Damage_anomaly").astype(int)
    df["y_attack"] = (lbl=="Cyberattack_anomaly").astype(int)
    return df

def time_based_val_split(df, val_frac=0.2):
    def split_one(g):
        n = len(g); cut = int((1.0 - val_frac)*n)
        return g.iloc[:max(cut,1)], g.iloc[max(cut,1):]
    parts = [split_one(g) for _, g in df.sort_values("timestamp").groupby("Device_ID")]
    left = [p[0] for p in parts]
    right= [p[1] for p in parts if len(p[1])>0]
    train_df = pd.concat(left, ignore_index=True)
    val_df   = pd.concat(right, ignore_index=True) if len(right)>0 else train_df.sample(frac=0.2, random_state=RANDOM_STATE)
    return train_df, val_df

def sweep_best_threshold(y_true, scores, beta=1.0):
    p, r, t = precision_recall_curve(y_true, scores)
    if t.size == 0: return 0.5
    f = (1+beta**2)*(p[:-1]*r[:-1]) / np.clip(beta**2*p[:-1] + r[:-1], 1e-9, None)
    best = int(np.nanargmax(f))
    return float(t[best])

def fit_calibrated_gbt_sup(train_det, val_det, attack_feats):
    cy_train_mask = train_det["primary_label_reported"].isin(["normal","Cyberattack_anomaly"])
    cy_val_mask   = val_det["primary_label_reported"].isin(["normal","Cyberattack_anomaly"])
    train_attack  = train_det[cy_train_mask].copy()
    val_attack    = val_det[cy_val_mask].copy()
    train_attack["y_attack_bin"] = (train_attack["primary_label_reported"]=="Cyberattack_anomaly").astype(int)
    val_attack["y_attack_bin"]   = (val_attack["primary_label_reported"]=="Cyberattack_anomaly").astype(int)

    X_sup = train_attack[attack_feats].fillna(0).values
    y_sup = train_attack["y_attack_bin"].values
    sw_sup= np.where(y_sup==1, 3.0, 1.0)

    base = HistGradientBoostingClassifier(max_iter=400, random_state=RANDOM_STATE)
    base.fit(X_sup, y_sup, sample_weight=sw_sup)

    cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
    cal.fit(val_attack[attack_feats].fillna(0).values, val_attack["y_attack_bin"].values)

    val_sup_scores = cal.predict_proba(val_attack[attack_feats].fillna(0).values)[:,1]
    thr_sup = sweep_best_threshold(val_attack["y_attack_bin"].values, val_sup_scores, beta=1.0)
    return cal, thr_sup

def posture_update(vapt_df, test_eval):
    """Update device posture based on observed anomalies. Returns updated VAPT and a per-device summary."""
    v = vapt_df.copy()
    # Aggregate anomalies per device-day
    dev = (test_eval.assign(date=pd.to_datetime(test_eval["timestamp"]).dt.date)
           .groupby(["Device_ID","date"])
           .agg(
               cyber_hits=("pred_attack_bin","sum"),
               cyber_days=("pred_attack_bin", lambda s: int(s.max()>0)),
               op_anom=("operational_label", lambda s: int((s!="normal").any())),
               n=("pred_attack_bin","count")
           )
           .reset_index())

    agg = dev.groupby("Device_ID").agg(
        cyber_days=("cyber_days","sum"),
        cyber_total=("cyber_hits","sum"),
        op_days=("op_anom","sum"),
        samples=("n","sum")
    ).reset_index()

    # Join with VAPT
    v["Device_ID"] = v["Device_ID"].astype(str)
    agg["Device_ID"] = agg["Device_ID"].astype(str)
    merged = v.merge(agg, on="Device_ID", how="left").fillna({"cyber_days":0,"cyber_total":0,"op_days":0,"samples":0})

    # Update rules (simple, transparent):
    # - epss: increase by min(0.25, 0.02 * cyber_days + 0.0005 * cyber_total)
    # - dvd : increase by min(0.5, 0.05 * op_days)
    # - cvss: slight bump if many cyber days and current cvss < 7
    epss_bump = np.minimum(0.25, 0.02*merged["cyber_days"] + 0.0005*merged["cyber_total"])
    dvd_bump  = np.minimum(0.5, 0.05*merged["op_days"])
    cvss_bump = np.where((merged.get("cvss",0)<7.0) & (merged["cyber_days"]>=3), 0.5, 0.0)

    out = merged.copy()
    out["cvss_final"] = np.clip(out.get("cvss",0).astype(float) + cvss_bump, 0.0, 10.0)
    out["epss_final"] = np.clip(out.get("epss",0).astype(float) + epss_bump, 0.0, 1.0)
    out["dvd_final"]  = np.clip(out.get("dvd",0).astype(float)  + dvd_bump , 0.0, 10.0)

    summary = out[["Device_ID","cvss","epss","dvd","cvss_final","epss_final","dvd_final",
                   "cyber_days","cyber_total","op_days","samples"]].copy()
    return out, summary

def main():
    train_path = pick_first_existing(["synthetic_train_better_journey_dev_overlap_delay.csv","synthetic_train_better_journey_dev.csv"])
    test_path  = pick_first_existing(["synthetic_test_better_journey_overlap_delay.csv","synthetic_test_better_journey.csv"])
    vapt_path  = pick_first_existing(["VAPT.csv"])

    # Read
    train_raw = read_df(train_path)
    test_raw  = read_df(test_path)
    vapt_df   = pd.read_csv(vapt_path)

    report_feature_availability(train_raw)

    # Feature engineering & targets
    train = engineer_features(train_raw)
    test  = engineer_features(test_raw)
    train = add_binary_targets(train)
    test  = add_binary_targets(test)

    # Split
    def split_by_time(df, val_frac=0.2):
        def split_one(g):
            n = len(g); cut = int((1.0 - val_frac)*n)
            return g.iloc[:max(cut,1)], g.iloc[max(cut,1):]
        parts = [split_one(g) for _, g in df.sort_values("timestamp").groupby("Device_ID")]
        left = [p[0] for p in parts]
        right= [p[1] for p in parts if len(p[1])>0]
        train_df = pd.concat(left, ignore_index=True)
        val_df   = pd.concat(right, ignore_index=True) if len(right)>0 else train_df.sample(frac=0.2, random_state=RANDOM_STATE)
        return train_df, val_df

    train_det, val_det = split_by_time(train, 0.2)
    if "is_augmented" in val_det.columns:
        val_det = val_det[val_det["is_augmented"] == 0].copy()

    # Feature sets
    temp_feats   = ["temp_reported","temp_roll_mean_3","temp_roll_std_3","temp_roll_std_5","delta_temp","temp_resid"]
    geo_feats    = ["distance_from_route_reported","dist_roll_max_5","delta_dist","gps_repeat_count"]
    tamper_feats = ["tamper_reported","delta_temp","temp_roll_std_3","distance_from_route_reported"]

    attack_feats = [
        # time & delays
        "reported_ts_diff_min","rpt_gap_min","rcv_gap_min","gap_err_min",
        "rpt_gap_cv_5","rcv_gap_cv_5","delay_min","delay_z","delay_big",
        # replay / duplicates
        "gps_repeat_count","gps_repeat_cap10","dup_ratio_5","dup_ratio_10",
        # smoothing/variance suppression
        "temp_var_5","hum_var_5","zcr_temp_5","zcr_hum_5",
        "temp_resid_std_ratio","hum_resid_std_ratio",
        # clock consistency
        "rpt_ooo",
        # kinematics
        "speed_mean_5","speed_std_5","accel_std_5","teleport_flag"
    ]

    # Derive reported_ts_diff_min (already computed as rpt_gap_min; ensure column present for backwards compat)
    train_det["reported_ts_diff_min"] = train_det["rpt_gap_min"].fillna(0)
    val_det["reported_ts_diff_min"]   = val_det["rpt_gap_min"].fillna(0)
    test["reported_ts_diff_min"]      = test["rpt_gap_min"].fillna(0)

    # Operational detectors (calibrated)
    dets = {}
    X = train_det[temp_feats].fillna(0).values; y = train_det["y_temp"].values
    temp_clf = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_STATE), method="isotonic", cv=3)
    temp_clf.fit(X, y); dets["temp"] = (temp_clf, temp_feats)

    X = train_det[geo_feats].fillna(0).values;  y = train_det["y_geo"].values
    geo_clf  = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_STATE), method="isotonic", cv=3)
    geo_clf.fit(X, y); dets["geo"] = (geo_clf, geo_feats)

    X = train_det[tamper_feats].fillna(0).values; y = train_det["y_tamper"].values
    tamper_clf = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_STATE), method="isotonic", cv=3)
    tamper_clf.fit(X, y); dets["tamper"] = (tamper_clf, tamper_feats)

    # thresholds for ops
    val = val_det.copy()
    for name, (clf, feats) in dets.items():
        val[f"score_{name}"] = clf.predict_proba(val[feats].fillna(0).values)[:,1]
    thr = {}
    def sweep(y, s, b): 
        p, r, t = precision_recall_curve(y, s)
        if t.size == 0: return 0.5
        f = (1+b**2)*(p[:-1]*r[:-1]) / np.clip(b**2*p[:-1] + r[:-1], 1e-9, None)
        return float(t[int(np.nanargmax(f))])
    thr["temp"]   = sweep(val["y_temp"].values,   val["score_temp"].values,   1.3)
    thr["geo"]    = sweep(val["y_geo"].values,    val["score_geo"].values,    1.3)
    thr["tamper"] = sweep(val["y_tamper"].values, val["score_tamper"].values, 2.0)

    # Cyber supervised
    sup_attack_clf, thr_sup = fit_calibrated_gbt_sup(train_det, val_det, attack_feats)
    thr["attack_sup_base"] = float(thr_sup)

    # Inference (TEST)
    test_eval = test.copy()
    for name, (clf, feats) in dets.items():
        test_eval[f"score_{name}"] = clf.predict_proba(test_eval[feats].fillna(0).values)[:,1]
        test_eval[f"pred_{name}_bin"] = 0

    for dev, g in test_eval.groupby("Device_ID"):
        idx = g.index
        for k in ["temp","geo","tamper"]:
            n_req = PERSISTENCE.get(k, 5)
            bin_raw = (g[f"score_{k}"].values >= thr[k]).astype(int)
            if n_req > 1:
                run, out = 0, np.zeros_like(bin_raw)
                for i, b in enumerate(bin_raw):
                    run = run+1 if b else 0
                    out[i] = 1 if run >= n_req else 0
                bin_raw = out
            test_eval.loc[idx, f"pred_{k}_bin"] = bin_raw

    if "tamper_reported" in test_eval.columns:
        tr = (test_eval["tamper_reported"].fillna(0) > 0).astype(int).values
        test_eval["pred_tamper_bin"] = np.maximum(test_eval["pred_tamper_bin"].values, tr)

    op_candidates = [
        ("Loss_of_storage_condition","pred_temp_bin","score_temp"),
        ("Geofence_anomaly","pred_geo_bin","score_geo"),
        ("Tamper_Damage_anomaly","pred_tamper_bin","score_tamper"),
    ]
    op_labels, op_scores = [], []
    for i in range(len(test_eval)):
        best_lbl, best_sc = "normal", -1.0
        row = test_eval.iloc[i]
        for lbl, bcol, scol in op_candidates:
            if row[bcol]==1:
                sc = float(row[scol])
                if sc > best_sc: best_lbl, best_sc = lbl, sc
        op_labels.append(best_lbl); op_scores.append(best_sc if best_sc>=0 else 0.0)
    test_eval["operational_label"] = op_labels
    test_eval["op_best_score"] = op_scores

    # Cyber → supervised only
    X_test_attack = test_eval[attack_feats].fillna(0).values
    test_eval["score_attack_sup"] = sup_attack_clf.predict_proba(X_test_attack)[:,1]

    # Risk-aware thresholding + dynamic persistence
    thr_base = thr["attack_sup_base"]
    epss_w, cvss_w, dvd_w = 0.35, 0.15, 0.15
    risk_term = (
        epss_w * test_eval.get("epss",0.0).fillna(0.0) +
        cvss_w * (test_eval.get("cvss",0.0).fillna(0.0)/10.0) +
        dvd_w  * test_eval.get("dvd",0.0).fillna(0.0)
    ).clip(0.0, 0.7)
    thr_attack_vec = np.clip(thr_base * (1.0 - risk_term), 0.03, 0.95)

    test_eval["pred_attack_bin"] = 0
    high_risk = ((test_eval.get("epss",0.0) >= 0.60) | (test_eval.get("cvss",0.0) >= 8.0)).astype(bool)

    def apply_persistence_variable(scores, thrs, n):
        bin_raw = (scores >= thrs).astype(int)
        if n <= 1: return bin_raw
        out, run = np.zeros_like(bin_raw), 0
        for i, b in enumerate(bin_raw):
            run = run+1 if b else 0
            out[i] = 1 if run >= n else 0
        return out

    for dev, g in test_eval.groupby("Device_ID"):
        idx = g.index
        n_default = PERSISTENCE.get("attack", 2)
        n_vec = np.where(high_risk.loc[idx].values, 1, n_default)
        bin_base = (g["score_attack_sup"].values >= thr_attack_vec.loc[idx].values).astype(int)
        bin_persist = apply_persistence_variable(g["score_attack_sup"].values, thr_attack_vec.loc[idx].values, n_default)
        bin_persist = np.where(n_vec==1, bin_base, bin_persist)
        test_eval.loc[idx, "pred_attack_bin"] = bin_persist

    # Reports
    y_true = test_eval["primary_label_reported"].astype(str).values
    def to_operational(lbl):
        return lbl if lbl in ("Geofence_anomaly","Loss_of_storage_condition","Tamper_Damage_anomaly") else "normal"
    op_true = np.array([to_operational(x) for x in y_true])

    print("\nOperational report (multi-class, cyber ignored):")
    print(classification_report(op_true, test_eval["operational_label"].values, zero_division=0, digits=3))

    y_cy_true = (test_eval["primary_label_reported"]=="Cyberattack_anomaly").astype(int).values
    try:
        p,r,_ = precision_recall_curve(y_cy_true, test_eval["score_attack_sup"].values)
        pr = auc(r,p); roc = roc_auc_score(y_cy_true, test_eval["score_attack_sup"].values) if len(np.unique(y_cy_true))>1 else float("nan")
        print(f"Cyber SUP → PR-AUC: {pr:.3f} | ROC-AUC: {roc:.3f}")
    except Exception as e:
        print("Cyber SUP metrics error:", e)

    # Posture update + Excel output
    updated_vapt, posture_summary = posture_update(vapt_df, test_eval)
    # Write Excel with initial & final posture
    with pd.ExcelWriter("security_posture_update.xlsx", engine="xlsxwriter") as writer:
        vapt_df.to_excel(writer, index=False, sheet_name="Initial_Posture")
        posture_summary.to_excel(writer, index=False, sheet_name="Final_Posture_Summary")

    # Risk dashboard
    def action(row):
        if row["pred_attack_bin"]==1 and row["op_best_score"]<0.5: return "ISOLATE_DEVICE"
        if row["operational_label"]=="Loss_of_storage_condition": return "PRIORITY_RECOVERY"
        if row["operational_label"]!="normal" and float(row.get("cvss",0))>=7.0: return "PATCH_AND_MONITOR"
        return "MONITOR"

    test_eval["risk_score_combined"] = (
        0.5*test_eval.get("epss",0.0).fillna(0.0) +
        0.3*(test_eval.get("cvss",0.0).fillna(0.0)/10.0) +
        0.2*test_eval.get("dvd",0.0).fillna(0.0)
    )
    test_eval["recommended_action"] = test_eval.apply(action, axis=1)

    dash = (test_eval.assign(date=pd.to_datetime(test_eval["timestamp"]).dt.date)
        .groupby(["Device_ID","date"]).agg(
            has_cyber=("pred_attack_bin","max"),
            op_label_top=("operational_label", lambda s: s.value_counts().idxmax()),
            max_op_score=("op_best_score","max"),
            risk_mean=("risk_score_combined","mean"),
            cvss=("cvss","max"), epss=("epss","max"), dvd=("dvd","max"),
            rec_action=("recommended_action", lambda s: s.value_counts().idxmax())
        ).reset_index())

    out_csv  = "test_predictions_sup_riskaware.csv"
    out_meta = "meta_sup_riskaware.json"
    dash_csv = "risk_dashboard_sup_riskaware.csv"
    test_eval.to_csv(out_csv, index=False)
    dash.to_csv(dash_csv, index=False)
    meta = {
        "persistence": PERSISTENCE,
        "thresholds": thr,
        "features": {
            "attack_feats": attack_feats
        },
        "notes": "SUP-only cyber, risk-aware thresholding, kinematics & replay features, posture update to Excel."
    }
    with open(out_meta, "w") as f: json.dump(meta, f, indent=2)

    print(f"Saved: {out_csv}, {out_meta}, {dash_csv}, security_posture_update.xlsx")

if __name__ == "__main__":
    import numpy as np
    raise SystemExit(main())
