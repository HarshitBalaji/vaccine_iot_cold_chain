#!/usr/bin/env python3
import sys
import os, json, warnings
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
    # optional
    if "timestamp_reported" in df.columns:
        df["timestamp_reported"] = _parse_dt(df["timestamp_reported"])
    else:
        df["timestamp_reported"] = pd.NaT
    for c in ["cvss","epss","dvd"]:
        if c not in df.columns: df[c] = 0.0
    for c in ["temp_reported","humidity_reported","distance_from_route_reported"]:
        if c not in df.columns: df[c] = 0.0
    if "tamper_reported" not in df.columns: df["tamper_reported"] = 0
    if "is_augmented" not in df.columns: df["is_augmented"] = 0
    return df

def impute_timestamp_reported_from_train(train, test):
    """If test lacks timestamp_reported, synthesize it using per-device median delay from train."""
    if test["timestamp_reported"].notna().any():
        return test
    train = train.copy()
    test = test.copy()
    train["delay_min"] = (train["timestamp"] - train["timestamp_reported"]).dt.total_seconds().div(60)
    per_dev = train.groupby(train["Device_ID"].astype(str))["delay_min"].median()
    global_delay = np.clip(np.nanmedian(train["delay_min"]), 0.0, 5.0)
    test["Device_ID"] = test["Device_ID"].astype(str)
    test = test.merge(per_dev.rename("median_delay_min"), how="left", left_on="Device_ID", right_index=True)
    test["median_delay_min"] = test["median_delay_min"].fillna(global_delay).clip(0.0, 5.0)
    rng = np.random.default_rng(42)
    jitter_min = rng.normal(0.0, 2.0, len(test)) / 60.0
    test["timestamp_reported"] = test["timestamp"] - pd.to_timedelta(test["median_delay_min"] + jitter_min, unit="m")
    # monotonicize per device
    def monotonicize(g):
        ts = g["timestamp_reported"].values.astype("datetime64[ns]")
        for i in range(1, len(ts)):
            if ts[i] < ts[i-1]:
                ts[i] = ts[i-1]
        g["timestamp_reported"] = ts
        return g
    test = test.sort_values(["Device_ID","timestamp"]).groupby("Device_ID", as_index=False, group_keys=False).apply(monotonicize)
    return test

def report_feature_availability(df):
    cols = set(df.columns)
    gps_ok = {"lat_reported","lon_reported"}.issubset(cols)
    tsr_ok = df["timestamp_reported"].notna().any()
    print(f"Feature availability → GPS: {gps_ok} | timestamp_reported: {tsr_ok} | tamper_reported: {'tamper_reported' in cols}")
    if not gps_ok: print("WARNING: GPS missing → freeze features weakened.")
    if not tsr_ok:  print("WARNING: device timestamp missing → using receive-time cadence surrogates.")

def engineer_features(df):
    df = df.sort_values(["Device_ID","timestamp"]).copy()
    has_rpt = df["timestamp_reported"].notna().any()
    if has_rpt:
        df["delay_min"] = (df["timestamp"] - df["timestamp_reported"]).dt.total_seconds().div(60)
        df["rpt_gap_min"] = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: x.diff().dt.total_seconds().div(60))
        df["rcv_gap_min"] = df.groupby("Device_ID")["timestamp"].transform(lambda x: x.diff().dt.total_seconds().div(60))
        df["gap_err_min"] = (df["rpt_gap_min"] - df["rcv_gap_min"]).fillna(0)

        df["delay_med_1h"] = df.groupby("Device_ID")["delay_min"].transform(lambda x: x.rolling(12, min_periods=3).median())
        df["delay_std_1h"] = df.groupby("Device_ID")["delay_min"].transform(lambda x: x.rolling(12, min_periods=3).std())
        df["delay_z"] = (df["delay_min"] - df["delay_med_1h"]) / (df["delay_std_1h"] + 1e-6)
        df["delay_big"] = (df["delay_min"] >= 10).astype(int)

        def _cv(a):
            a = np.asarray(a, dtype=float)
            return (np.nanstd(a)+1e-6)/(np.nanmean(a)+1e-6)
        df["rpt_gap_cv_5"] = df.groupby("Device_ID")["rpt_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).apply(_cv, raw=True)).fillna(0)
        df["rcv_gap_cv_5"] = df.groupby("Device_ID")["rcv_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).apply(_cv, raw=True)).fillna(0)

        df["rpt_ooo"] = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: (x < x.shift(1))).fillna(False).astype(int)

        rpt_s = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: x.astype("int64")/1e9)
        rcv_s = df.groupby("Device_ID")["timestamp"].transform(lambda x: x.astype("int64")/1e9)
        df["rpt_roll_mean_12"] = rpt_s.groupby(df["Device_ID"]).transform(lambda x: pd.Series(x).rolling(12, min_periods=6).mean())
        df["rcv_roll_mean_12"] = rcv_s.groupby(df["Device_ID"]).transform(lambda x: pd.Series(x).rolling(12, min_periods=6).mean())
        df["clock_resid"] = (rcv_s - df["rcv_roll_mean_12"]) - (rpt_s - df["rpt_roll_mean_12"])
        df["clock_resid"] = df["clock_resid"].fillna(0)

        df["rcv_gap_small"] = (df["rcv_gap_min"].fillna(0) <= 1.0).astype(int)
        df["rpt_gap_std_5"] = df.groupby("Device_ID")["rpt_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).std()).fillna(0)
        df["rcv_gap_std_5"] = df.groupby("Device_ID")["rcv_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).std()).fillna(0)
        df["burst_count_5"] = df.groupby("Device_ID")["rcv_gap_small"].transform(lambda x: x.rolling(5, min_periods=1).sum()).fillna(0)
        df["delay_spike"] = (np.abs(df["delay_z"].fillna(0)) >= 3.0).astype(int)
        df["replay_sig"] = ((df["rpt_gap_std_5"] < 0.1) & (df["rcv_gap_std_5"] > 0.5)).astype(int)
        df["rpt_ooo_run5"] = df.groupby("Device_ID")["rpt_ooo"].transform(lambda x: x.rolling(5, min_periods=2).sum()).fillna(0)
        df["rpt_ooo_burst"] = (df["rpt_ooo_run5"] >= 2).astype(int)

        df["reported_ts_diff_min"] = df["rpt_gap_min"].fillna(0)
    else:
        df["reported_ts_diff_min"] = df.groupby("Device_ID")["timestamp"].transform(lambda x: x.diff().dt.total_seconds().div(60)).fillna(0)
        for c, v in [("delay_min",0.0),("gap_err_min",0.0),("delay_z",0.0),("delay_big",0),
                     ("rpt_gap_cv_5",0.0),("rcv_gap_cv_5",0.0),("rpt_ooo",0),("clock_resid",0.0),
                     ("burst_count_5",0.0),("delay_spike",0),("replay_sig",0),("rpt_ooo_burst",0)]:
            df[c] = v

    if {"lat_reported","lon_reported"}.issubset(df.columns):
        lat_same = df.groupby("Device_ID")["lat_reported"].transform(lambda s: s.round(6).eq(s.round(6).shift(1)))
        lon_same = df.groupby("Device_ID")["lon_reported"].transform(lambda s: s.round(6).eq(s.round(6).shift(1)))
        df["gps_same"] = (lat_same & lon_same).fillna(False).astype(int)

        def _consec_ones(s: pd.Series) -> pd.Series:
            run, out = 0, []
            for v in s.astype(int).tolist():
                run = run + 1 if v == 1 else 0
                out.append(run)
            return pd.Series(out, index=s.index, dtype="int64")

        df["gps_repeat_count"] = df.groupby("Device_ID")["gps_same"].transform(_consec_ones)
        df["gps_repeat_cap10"] = np.minimum(df["gps_repeat_count"], 10).astype(int)
    else:
        df["gps_same"] = 0
        df["gps_repeat_count"] = 0
        df["gps_repeat_cap10"] = 0

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
    # Train on earlier, calibrate on later (strictly time-aware)
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

def main():
    train_path = pick_first_existing(["synthetic_train_better_journey_dev.csv"])
    test_path  = pick_first_existing(["synthetic_test_better_journey_with_tsr.csv"])

    train_raw = read_df(train_path)
    test_raw  = read_df(test_path)

    # If test lacks timestamp_reported, synthesize it from train
    if not test_raw["timestamp_reported"].notna().any():
        print("Test lacks timestamp_reported; synthesizing from train per-device delay...")
        test_raw = impute_timestamp_reported_from_train(train_raw, test_raw)

    report_feature_availability(train_raw)

    # Features + targets
    train = engineer_features(train_raw)
    test  = engineer_features(test_raw)
    train = add_binary_targets(train)
    test  = add_binary_targets(test)

    # carry augmentation flag through pipeline
    train["is_augmented"] = train_raw["is_augmented"].values if "is_augmented" in train_raw.columns else 0

    # Feature sets
    temp_feats   = ["temp_reported","temp_roll_mean_3","temp_roll_std_3","temp_roll_std_5","delta_temp","temp_resid"]
    geo_feats    = ["distance_from_route_reported","dist_roll_max_5","delta_dist","gps_repeat_count"]
    tamper_feats = ["tamper_reported","delta_temp","temp_roll_std_3","distance_from_route_reported"]

    attack_feats = [
        "reported_ts_diff_min","rpt_gap_min","rcv_gap_min","gap_err_min",
        "rpt_gap_cv_5","rcv_gap_cv_5","delay_min","delay_z","delay_big",
        "burst_count_5","delay_spike","replay_sig","rpt_ooo_burst",
        "gps_repeat_count","gps_repeat_cap10",
        "temp_var_5","hum_var_5","zcr_temp_5","zcr_hum_5",
        "temp_resid_std_ratio","hum_resid_std_ratio",
        "rpt_ooo","clock_resid"
    ]

    # Time-based split
    train_det, val_det = time_based_val_split(train, val_frac=0.2)
    if "is_augmented" in val_det.columns:
        val_det = val_det[val_det["is_augmented"] == 0].copy()

    # Operational detectors (supervised, calibrated)
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
    thr["temp"]   = sweep_best_threshold(val["y_temp"].values,   val["score_temp"].values,   beta=1.3)
    thr["geo"]    = sweep_best_threshold(val["y_geo"].values,    val["score_geo"].values,    beta=1.3)
    thr["tamper"] = sweep_best_threshold(val["y_tamper"].values, val["score_tamper"].values, beta=2.0)

    # Cyber supervised (train earlier, calibrate later)
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
            # persistence on calibrated scores
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
    # ===================== Reports (robust) =====================
    from sklearn.metrics import (classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix)
    import sys

    print("\n=== BEGIN REPORTS ==="); sys.stdout.flush()

    # --- Operational anomalies (multi-class, cyber ignored) ---
    y_true = test_eval["primary_label_reported"].astype(str).values
    def to_operational(lbl):
        return lbl if lbl in ("Geofence_anomaly", "Loss_of_storage_condition", "Tamper_Damage_anomaly") else "normal"
    op_true = np.array([to_operational(x) for x in y_true])

    print("\nOperational report (multi-class, cyber ignored):")
    print(classification_report( op_true, test_eval["operational_label"].astype(str).values, zero_division=0, digits=3 )); sys.stdout.flush()

    # --- Cyber anomalies (binary) ---
    # y_true (0/1)
    y_cy_true = (test_eval["primary_label_reported"] == "Cyberattack_anomaly").astype(int).to_numpy()

    # scores (for AUCs)
    if "score_attack_sup" in test_eval.columns:
        y_cy_score = test_eval["score_attack_sup"].astype(float).to_numpy()
    else:
        # fallback: if scores missing, use zeros to avoid crash (AUCs will fail gracefully)
        y_cy_score = np.zeros(len(test_eval), dtype=float)

    # predictions (for PR/RC/F1) — tolerate missing/NaN column
    y_cy_pred = ( test_eval.get("pred_attack_bin", pd.Series([0]*len(test_eval))) .fillna(0).astype(int).clip(0, 1).to_numpy())

    # Sentinels to confirm we got here with sane shapes
    print(f"\n[debug] cyber y_true shape={y_cy_true.shape}, "
      f"y_pred shape={y_cy_pred.shape}, scores shape={y_cy_score.shape}")
    print(f"[debug] positives in truth={int(y_cy_true.sum())}, "
      f"positives in preds={int(y_cy_pred.sum())}"); sys.stdout.flush()

    # AUCs (don’t wrap the whole block in try; just guard the AUCs)
    try:
        p, r, _ = precision_recall_curve(y_cy_true, y_cy_score)
        pr_auc = auc(r, p)
        roc_auc = roc_auc_score(y_cy_true, y_cy_score) if len(np.unique(y_cy_true)) > 1 else float("nan")
        print(f"\nCyber SUP → PR-AUC: {pr_auc:.3f} | ROC-AUC: {roc_auc:.3f}")
    except Exception as e:
        print(f"Cyber SUP AUCs error: {e}")

    # Classification report (always prints)
    print("\nCyber anomaly report (binary):")
    print(classification_report( y_cy_true, y_cy_pred, labels=[0, 1], target_names=["normal", "Cyberattack_anomaly"], zero_division=0, digits=3 ));sys.stdout.flush()

    # Confusion matrix + quick metrics
    cm = confusion_matrix(y_cy_true, y_cy_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
    print(f"Confusion matrix [[tn fp]; [fn tp]] = {cm.tolist()}")
    print(f"precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}"); sys.stdout.flush()

    print("=== END REPORTS ===\n"); sys.stdout.flush()
# ============================================================

    # Risk dashboard (per device-day)
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

    out_csv  = "test_predictions_sup_only.csv"
    out_meta = "meta_sup_only.json"
    dash_csv = "risk_dashboard_sup_only.csv"
    test_eval.to_csv(out_csv, index=False)
    dash.to_csv(dash_csv, index=False)
    meta = {
        "persistence": PERSISTENCE,
        "thresholds": {
            "temp": float(thr["temp"]), "geo": float(thr["geo"]), "tamper": float(thr["tamper"]),
            "attack_sup_base": float(thr["attack_sup_base"])
        },
        "features": {
            "temp_feats": temp_feats, "geo_feats": geo_feats, "tamper_feats": tamper_feats,
            "attack_feats": attack_feats
        },
        "notes": "SUP-only cyber with time-aware calibration (train earlier, calibrate on later val). If test lacks timestamp_reported, it's synthesized from per-device train delay."
    }
    with open(out_meta, "w") as f: json.dump(meta, f, indent=2)

    print(f"Saved: {out_csv}, {out_meta}, {dash_csv}")
    print("Key columns: operational_label, op_best_score, score_attack_sup, pred_attack_bin, recommended_action")

if __name__ == "__main__":
    import numpy as np
    raise SystemExit(main())
