#!/usr/bin/env python3
"""
Improved 2LFFMAD — Operational + Cyber (independent)
- Operational detectors: calibrated GBT + persistence
- Cyber detector: Hybrid (Supervised GBT + Isolation Forest w/ RobustScaler), risk-aware thresholds & dynamic persistence
- Clean validation: augmented rows excluded from validation tuning
- Strong timestamp features (if timestamp_reported present)
- Tamper recall-first + rule failsafe

Expected files (same folder):
  Train (first existing): synthetic_train_better_journey_dev.csv | synthetic_train_better_mapped.csv | synthetic_train_better.csv
  Test  (first existing): synthetic_test_better_mapped.csv     | synthetic_test_better.csv
  VAPT posture         : VAPT.csv   (Device_ID, cvss, epss, dvd [, nist_gap_count])

Outputs:
  test_predictions_independent_hybrid.csv
  meta_independent_hybrid.json
  risk_dashboard_hybrid.csv
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Persistence (consecutive samples)
PERSISTENCE = {"temp":4, "geo":4, "tamper":1, "attack":2}  # tamper=1 (recall-first)

# ----------------------------- IO helpers -----------------------------
def pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)

def _parse_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)

def read_df(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if "Device_ID" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("CSV must contain Device_ID and timestamp")
    df["Device_ID"] = df["Device_ID"].astype(str).str.strip()
    df["timestamp"] = _parse_dt(df["timestamp"])
    # device-reported time (if present)
    if "timestamp_reported" in df.columns:
        df["timestamp_reported"] = _parse_dt(df["timestamp_reported"])
    else:
        df["timestamp_reported"] = pd.NaT
    # posture placeholders if any missing
    for c in ["cvss","epss","dvd"]:
        if c not in df.columns: df[c] = 0.0
    # core sensors
    for c in ["temp_reported","humidity_reported","distance_from_route_reported"]:
        if c not in df.columns: df[c] = 0.0
    if "tamper_reported" not in df.columns: df["tamper_reported"] = 0
    if "primary_label_reported" not in df.columns:
        raise ValueError("CSV must contain primary_label_reported")
    # augmentation flag (optional)
    if "is_augmented" not in df.columns:
        df["is_augmented"] = 0
    return df

def merge_vapt(train, test, vapt_path="VAPT.csv"):
    if not os.path.exists(vapt_path):
        print("VAPT posture file not found; proceeding without it.")
        return train, test
    v = pd.read_csv(vapt_path)
    need = {"Device_ID","cvss","epss","dvd"}
    if not need.issubset(v.columns):
        print("VAPT.csv missing one of", need, "→ posture features will be zeros.")
        return train, test
    v = v[["Device_ID","cvss","epss","dvd"] + [c for c in ["nist_gap_count"] if c in v.columns]].copy()
    for c in ["cvss","epss","dvd"]: v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0)
    v["Device_ID"] = v["Device_ID"].astype(str).str.strip()
    for name, df in (("train", train), ("test", test)):
        df["Device_ID"] = df["Device_ID"].astype(str).str.strip()
        df.drop(columns=["cvss","epss","dvd","nist_gap_count"], errors="ignore", inplace=True)
        df = df.merge(v, on="Device_ID", how="left")
        for c in ["cvss","epss","dvd","nist_gap_count"]:
            if c in df.columns: df[c] = df[c].fillna(0.0)
        if name=="train": train = df
        else: test = df
    print("Merged VAPT posture:", vapt_path)
    return train, test

# ----------------------------- Feature engineering -----------------------------
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
    # --- time axes & manipulation signals
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

        # Drift via centered-difference residual (use astype int64 for portability)
        rpt_s = df.groupby("Device_ID")["timestamp_reported"].transform(lambda x: x.astype("int64")/1e9)
        rcv_s = df.groupby("Device_ID")["timestamp"].transform(lambda x: x.astype("int64")/1e9)
        df["rpt_roll_mean_12"] = rpt_s.groupby(df["Device_ID"]).transform(lambda x: pd.Series(x).rolling(12, min_periods=6).mean())
        df["rcv_roll_mean_12"] = rcv_s.groupby(df["Device_ID"]).transform(lambda x: pd.Series(x).rolling(12, min_periods=6).mean())
        df["clock_resid"] = (rcv_s - df["rcv_roll_mean_12"]) - (rpt_s - df["rpt_roll_mean_12"])
        df["clock_resid"] = df["clock_resid"].fillna(0)

        # burstiness & replay cues
        df["rcv_gap_small"] = (df["rcv_gap_min"].fillna(0) <= 1.0).astype(int)
        df["rpt_gap_std_5"] = df.groupby("Device_ID")["rpt_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).std()).fillna(0)
        df["rcv_gap_std_5"] = df.groupby("Device_ID")["rcv_gap_min"].transform(lambda x: x.rolling(5, min_periods=3).std()).fillna(0)
        df["burst_count_5"] = df.groupby("Device_ID")["rcv_gap_small"].transform(lambda x: x.rolling(5, min_periods=1).sum()).fillna(0)
        df["delay_spike"] = (np.abs(df["delay_z"].fillna(0)) >= 3.0).astype(int)
        df["replay_sig"] = ((df["rpt_gap_std_5"] < 0.1) & (df["rcv_gap_std_5"] > 0.5)).astype(int)
        # out-of-order bursts
        df["rpt_ooo_run5"] = df.groupby("Device_ID")["rpt_ooo"].transform(lambda x: x.rolling(5, min_periods=2).sum()).fillna(0)
        df["rpt_ooo_burst"] = (df["rpt_ooo_run5"] >= 2).astype(int)

        df["reported_ts_diff_min"] = df["rpt_gap_min"].fillna(0)
    else:
        df["reported_ts_diff_min"] = df.groupby("Device_ID")["timestamp"].transform(lambda x: x.diff().dt.total_seconds().div(60)).fillna(0)
        for c, v in [("delay_min",0.0),("gap_err_min",0.0),("delay_z",0.0),("delay_big",0),
                     ("rpt_gap_cv_5",0.0),("rcv_gap_cv_5",0.0),("rpt_ooo",0),("clock_resid",0.0),
                     ("burst_count_5",0.0),("delay_spike",0),("replay_sig",0),("rpt_ooo_burst",0)]:
            df[c] = v

    # --- GPS freeze (robust)
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

    # --- rolling stats & residuals
    for w in [3,5,7]:
        df[f"temp_roll_mean_{w}"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"temp_roll_std_{w}"]  = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
        df[f"hum_roll_mean_{w}"]  = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"hum_roll_std_{w}"]   = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
        df[f"dist_roll_max_{w}"]  = df.groupby("Device_ID")["distance_from_route_reported"].transform(lambda x: x.rolling(w, min_periods=1).max())

    # deltas
    df["delta_temp"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: x.diff()).fillna(0)
    df["delta_hum"]  = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: x.diff()).fillna(0)
    df["delta_dist"] = df.groupby("Device_ID")["distance_from_route_reported"].transform(lambda x: x.diff()).fillna(0)

    # residuals
    def _ewma_resid(x, span=5):
        m = x.ewm(span=span, adjust=False).mean()
        return (x - m)
    df["temp_resid"] = df.groupby("Device_ID")["temp_reported"].transform(lambda x: _ewma_resid(x, span=5)).fillna(0)
    df["hum_resid"]  = df.groupby("Device_ID")["humidity_reported"].transform(lambda x: _ewma_resid(x, span=5)).fillna(0)

    # manipulation fingerprints
    for col, new in [("temp_reported","temp_var_5"), ("humidity_reported","hum_var_5")]:
        df[new] = df.groupby("Device_ID")[col].transform(lambda x: x.rolling(5, min_periods=2).var()).fillna(0)

    # zcr on deltas
    def _zcr_roll(series, win=5):
        s = series.fillna(0).values
        sign = np.sign(s)
        zc = (np.roll(sign, 1) != sign).astype(int); zc[0] = 0
        return pd.Series(zc).rolling(win, min_periods=2).mean().values
    df["zcr_temp_5"] = df.groupby("Device_ID")["delta_temp"].transform(_zcr_roll)
    df["zcr_hum_5"]  = df.groupby("Device_ID")["delta_hum"].transform(_zcr_roll)

    # ratios
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

# ----------------------------- Model helpers -----------------------------
def _make_calibrated(est, method="isotonic"):
    cv_obj = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    try:
        return CalibratedClassifierCV(base_estimator=est, method=method, cv=cv_obj)
    except TypeError:
        return CalibratedClassifierCV(estimator=est, method=method, cv=cv_obj)

def fit_calibrated_gbt(X, y, sample_weight=None, max_iter=300, calib_method="isotonic"):
    base = HistGradientBoostingClassifier(max_iter=max_iter, random_state=RANDOM_STATE)
    clf = _make_calibrated(base, method=calib_method)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf

def sweep_best_threshold(y_true, scores, beta=1.0):
    p, r, t = precision_recall_curve(y_true, scores)
    if len(t)==0: return 0.5
    f = (1+beta**2)*(p*r)/np.clip(beta**2*p + r, 1e-9, None)
    best = np.nanargmax(f); thr = t[min(best, len(t)-1)]
    return float(thr)

def apply_persistence(scores, thr, n):
    bin_raw = (scores >= thr).astype(int)
    if n <= 1: return bin_raw
    out, run = np.zeros_like(bin_raw), 0
    for i, b in enumerate(bin_raw):
        run = run+1 if b else 0
        out[i] = 1 if run >= n else 0
    return out

def apply_persistence_variable(scores, thrs, n):
    bin_raw = (scores >= thrs).astype(int)
    if n <= 1: return bin_raw
    out, run = np.zeros_like(bin_raw), 0
    for i, b in enumerate(bin_raw):
        run = run+1 if b else 0
        out[i] = 1 if run >= n else 0
    return out

# ----------------------------- Main -----------------------------
def main():
    # Prefer augmented train if present
    train_path = pick_first_existing([
        "synthetic_train_better_journey_dev.csv"
    ])
    test_path = pick_first_existing([
        "synthetic_test_better_journey.csv"
    ])

    train_raw = read_df(train_path)
    test_raw  = read_df(test_path)
    report_feature_availability(train_raw)

    # Merge VAPT
    train_raw, test_raw = merge_vapt(train_raw, test_raw, "VAPT.csv")

    # Features + targets
    train = engineer_features(train_raw)
    test  = engineer_features(test_raw)
    train = add_binary_targets(train)
    test  = add_binary_targets(test)

    # carry augmentation flag through pipeline
    train["is_augmented"] = train_raw["is_augmented"].values if "is_augmented" in train_raw.columns else 0

    # ---- Feature sets ----
    temp_feats   = ["temp_reported","temp_roll_mean_3","temp_roll_std_3","temp_roll_std_5","delta_temp","temp_resid"]
    geo_feats    = ["distance_from_route_reported","dist_roll_max_5","delta_dist","gps_repeat_count"]
    tamper_feats = ["tamper_reported","delta_temp","temp_roll_std_3","distance_from_route_reported"]

    attack_feats = [
        # time & delays
        "reported_ts_diff_min","rpt_gap_min","rcv_gap_min","gap_err_min",
        "rpt_gap_cv_5","rcv_gap_cv_5","delay_min","delay_z","delay_big",
        "burst_count_5","delay_spike","replay_sig","rpt_ooo_burst",
        # gps freeze
        "gps_repeat_count","gps_repeat_cap10",
        # smoothing/variance suppression
        "temp_var_5","hum_var_5","zcr_temp_5","zcr_hum_5",
        "temp_resid_std_ratio","hum_resid_std_ratio",
        # clock consistency
        "rpt_ooo","clock_resid"
    ]

    # Time-based split
    train_det, val_det = time_based_val_split(train, val_frac=0.2)
    # CLEAN VALIDATION: exclude augmented rows from the validation set
    if "is_augmented" in val_det.columns:
        val_det = val_det[val_det["is_augmented"] == 0].copy()

    # ----------------- Operational detectors (supervised, calibrated) -----------------
    dets = {}
    X = train_det[temp_feats].fillna(0).values; y = train_det["y_temp"].values
    temp_clf = fit_calibrated_gbt(X, y, None, 300, "isotonic"); dets["temp"] = (temp_clf, temp_feats)
    X = train_det[geo_feats].fillna(0).values;  y = train_det["y_geo"].values
    geo_clf  = fit_calibrated_gbt(X, y, None, 300, "isotonic"); dets["geo"] = (geo_clf, geo_feats)
    X = train_det[tamper_feats].fillna(0).values; y = train_det["y_tamper"].values
    tamper_clf = fit_calibrated_gbt(X, y, None, 300, "isotonic"); dets["tamper"] = (tamper_clf, tamper_feats)

    # thresholds for ops (recall-leaning); tamper even more recall-leaning
    val = val_det.copy()
    for name, (clf, feats) in dets.items():
        val[f"score_{name}"] = clf.predict_proba(val[feats].fillna(0).values)[:,1]
    thr = {}
    thr["temp"]   = sweep_best_threshold(val["y_temp"].values,   val["score_temp"].values,   beta=1.3)
    thr["geo"]    = sweep_best_threshold(val["y_geo"].values,    val["score_geo"].values,    beta=1.3)
    thr["tamper"] = sweep_best_threshold(val["y_tamper"].values, val["score_tamper"].values, beta=2.0)

    # ----------------- Cyber: supervised head (sigmoid) -----------------
    cy_train_mask = train_det["primary_label_reported"].isin(["normal","Cyberattack_anomaly"])
    cy_val_mask   = val_det["primary_label_reported"].isin(["normal","Cyberattack_anomaly"])
    train_attack  = train_det[cy_train_mask].copy()
    val_attack    = val_det[cy_val_mask].copy()
    train_attack["y_attack_bin"] = (train_attack["primary_label_reported"]=="Cyberattack_anomaly").astype(int)
    val_attack["y_attack_bin"]   = (val_attack["primary_label_reported"]=="Cyberattack_anomaly").astype(int)

    X_sup = train_attack[attack_feats].fillna(0).values
    y_sup = train_attack["y_attack_bin"].values
    sw_sup= np.where(y_sup==1, 3.0, 1.0)
    sup_attack_clf = fit_calibrated_gbt(X_sup, y_sup, sample_weight=sw_sup, max_iter=400, calib_method="sigmoid")
    val_sup_scores = sup_attack_clf.predict_proba(val_attack[attack_feats].fillna(0).values)[:,1]
    thr["attack_sup_base"] = sweep_best_threshold(val_attack["y_attack_bin"].values, val_sup_scores, beta=1.0)

    # ----------------- Cyber: Isolation Forest (unsupervised) w/ RobustScaler -----------------
    scaler_if = RobustScaler()
    X_if_train_norm = scaler_if.fit_transform(train_attack.loc[train_attack["y_attack_bin"]==0, attack_feats].fillna(0).values)
    if_clf = IsolationForest(n_estimators=300, contamination=0.02, random_state=RANDOM_STATE)
    if_clf.fit(X_if_train_norm)

    def iforest_prob(model, X):
        s = -model.score_samples(X)  # higher → more anomalous
        s_min, s_max = np.min(s), np.max(s)
        if s_max - s_min < 1e-9: return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)

    X_if_val  = scaler_if.transform(val_attack[attack_feats].fillna(0).values)
    val_if_scores = iforest_prob(if_clf, X_if_val)
    thr["attack_if_base"] = sweep_best_threshold(val_attack["y_attack_bin"].values, val_if_scores, beta=1.0)

    # ----------------- Cyber ensemble (tune alpha on PR-AUC) -----------------
    alphas = [0.2, 0.35, 0.5, 0.65, 0.8]
    best_alpha, best_prauc = 0.5, -1.0
    for a in alphas:
        sc = a*val_sup_scores + (1.0-a)*val_if_scores
        p, r, _ = precision_recall_curve(val_attack["y_attack_bin"].values, sc)
        pr = auc(r, p)
        if pr > best_prauc: best_alpha, best_prauc = a, pr
    alpha_ens = best_alpha
    thr["attack_ens_base"] = sweep_best_threshold(
        val_attack["y_attack_bin"].values,
        alpha_ens*val_sup_scores + (1.0-alpha_ens)*val_if_scores,
        beta=1.0
    )
    print(f"Cyber ensemble tuned: alpha={alpha_ens:.2f} | VAL PR-AUC={best_prauc:.3f}")

    # ----------------- Inference (TEST) -----------------
    test_eval = test.copy()

    # Operational → scores + persistence + tamper failsafe → operational_label
    for name, (clf, feats) in dets.items():
        test_eval[f"score_{name}"] = clf.predict_proba(test_eval[feats].fillna(0).values)[:,1]
        test_eval[f"pred_{name}_bin"] = 0

    for dev, g in test_eval.groupby("Device_ID"):
        idx = g.index
        for k in ["temp","geo","tamper"]:
            n_req = PERSISTENCE.get(k, 5)
            test_eval.loc[idx, f"pred_{k}_bin"] = apply_persistence(g[f"score_{k}"].values, thr[k], n=n_req)

    # Tamper raw-sensor failsafe (boost recall)
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

    # Cyber → supervised + IF + ensemble
    X_test_attack = test_eval[attack_feats].fillna(0).values
    test_eval["score_attack_sup"] = sup_attack_clf.predict_proba(X_test_attack)[:,1]
    X_if_test = scaler_if.transform(test_eval[attack_feats].fillna(0).values)
    test_eval["score_attack_if"]  = iforest_prob(if_clf, X_if_test)
    test_eval["score_attack_ens"] = alpha_ens*test_eval["score_attack_sup"] + (1.0-alpha_ens)*test_eval["score_attack_if"]

    # Risk-aware thresholding (stronger weights) + dynamic persistence on high-risk devices
    thr_base = thr["attack_ens_base"]
    epss_w, cvss_w, dvd_w = 0.35, 0.15, 0.15
    risk_term = (
        epss_w * test_eval.get("epss",0.0) +
        cvss_w * (test_eval.get("cvss",0.0)/10.0) +
        dvd_w  * test_eval.get("dvd",0.0)
    ).fillna(0.0)
    risk_term = np.clip(risk_term, 0.0, 0.7)
    thr_attack_vec = np.clip(thr_base * (1.0 - risk_term), 0.03, 0.95)

    # Persistence with variable thresholds (+ n=1 on very high risk)
    test_eval["pred_attack_bin"] = 0
    high_risk = ((test_eval.get("epss",0.0) >= 0.60) | (test_eval.get("cvss",0.0) >= 8.0)).astype(bool)

    for dev, g in test_eval.groupby("Device_ID"):
        idx = g.index
        n_default = PERSISTENCE.get("attack", 2)
        n_vec = np.where(high_risk.loc[idx].values, 1, n_default)
        # apply persistence row-wise with per-row n is tricky; approximate: two passes
        # pass 1: n=1 for all rows where n_vec==1
        bin_base = (g["score_attack_ens"].values >= thr_attack_vec.loc[idx].values).astype(int)
        # build final with runs where n_default applies, but force any n=1 positives
        bin_persist = apply_persistence_variable(g["score_attack_ens"].values, thr_attack_vec.loc[idx].values, n_default)
        bin_persist = np.where(n_vec==1, bin_base, bin_persist)
        test_eval.loc[idx, "pred_attack_bin"] = bin_persist

    # Independent outputs (no masking)
    test_eval["cyber_prob"] = test_eval["score_attack_ens"].astype(float)
    test_eval["cyber_flag"] = test_eval["pred_attack_bin"].astype(int)
    # Single fused label for legacy consumers = operational only (no override)
    test_eval["final_label"] = test_eval["operational_label"]

    # ----------------- Reports -----------------
    y_true = test_eval["primary_label_reported"].astype(str).values
    def to_operational(lbl):
        return lbl if lbl in ("Geofence_anomaly","Loss_of_storage_condition","Tamper_Damage_anomaly") else "normal"
    op_true = np.array([to_operational(x) for x in y_true])

    print("\nOperational report (multi-class, cyber ignored):")
    print(classification_report(op_true, test_eval["operational_label"].values, zero_division=0, digits=3))

    y_cy_true = (test_eval["primary_label_reported"]=="Cyberattack_anomaly").astype(int).values
    for name in ["sup","if","ens"]:
        sc = test_eval[f"score_attack_{'ens' if name=='ens' else name}"].values
        try:
            p,r,_ = precision_recall_curve(y_cy_true, sc)
            pr = auc(r,p); roc = roc_auc_score(y_cy_true, sc) if len(np.unique(y_cy_true))>1 else float("nan")
            print(f"Cyber {name.upper()} → PR-AUC: {pr:.3f} | ROC-AUC: {roc:.3f}")
        except Exception as e:
            print(f"Cyber {name} metrics error:", e)

    # ----------------- Risk dashboard (per device-day) -----------------
    def action(row):
        if row["cyber_flag"]==1 and row["op_best_score"]<0.5: return "ISOLATE_DEVICE"
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
            has_cyber=("cyber_flag","max"),
            op_label_top=("operational_label", lambda s: s.value_counts().idxmax()),
            max_op_score=("op_best_score","max"),
            risk_mean=("risk_score_combined","mean"),
            cvss=("cvss","max"), epss=("epss","max"), dvd=("dvd","max"),
            rec_action=("recommended_action", lambda s: s.value_counts().idxmax())
        ).reset_index())

    # ----------------- Save -----------------
    out_csv  = "test_predictions_independent_hybrid.csv"
    out_meta = "meta_independent_hybrid.json"
    dash_csv = "risk_dashboard_hybrid.csv"
    test_eval.to_csv(out_csv, index=False)
    dash.to_csv(dash_csv, index=False)
    meta = {
        "persistence": PERSISTENCE,
        "thresholds": {
            "temp": float(thr["temp"]), "geo": float(thr["geo"]), "tamper": float(thr["tamper"]),
            "attack_sup_base": float(thr["attack_sup_base"]),
            "attack_if_base": float(thr["attack_if_base"]),
            "attack_ens_base": float(thr["attack_ens_base"]),
        },
        "ensemble_alpha": float(alpha_ens),
        "risk_threshold_weights": {"epss_w":0.35, "cvss_w":0.15, "dvd_w":0.15, "cap":0.7},
        "features": {
            "temp_feats": temp_feats, "geo_feats": geo_feats, "tamper_feats": tamper_feats,
            "attack_feats": attack_feats
        },
        "notes": "Independent ops & cyber. Cyber = supervised(sigmoid) + IF(RobustScaler) with risk-aware thresholds and dynamic persistence. Validation excludes augmented rows."
    }
    with open(out_meta, "w") as f: json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_csv}, {out_meta}, {dash_csv}")
    print("Key columns: operational_label, op_best_score, cyber_prob, cyber_flag, final_label, recommended_action")

if __name__ == "__main__":
    raise SystemExit(main())
