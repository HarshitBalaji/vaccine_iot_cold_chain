import numpy as np, pandas as pd
from datetime import datetime, timedelta

def generate_rebalanced_train(devices=50, days=7, interval_min=5, seed=2025):
    np.random.seed(seed)
    samples_per_device = int(days*24*60/interval_min)
    rows = []
    center_lat, center_lon = 12.9716, 77.5946

    for d in range(devices):
        dev_id = f"D{d+1:03d}"
        cvss = float(np.clip(np.random.beta(1.5,6.0)*10 + np.random.normal(0,0.5), 0, 10))
        epss = float(np.clip(np.random.beta(1.2,6.0)*0.6 + np.random.normal(0,0.02), 0, 1))
        dvd = int(max(1, np.random.poisson(3)))

        # base signals
        lat0 = center_lat + np.random.normal(0, 0.01)
        lon0 = center_lon + np.random.normal(0, 0.01)
        lats = lat0 + np.cumsum(np.random.normal(0, 0.0002, samples_per_device))
        lons = lon0 + np.cumsum(np.random.normal(0, 0.0002, samples_per_device))
        temp_true = np.random.normal(5.0, 0.6, samples_per_device)
        hum_true = np.clip(np.random.normal(55, 5, samples_per_device), 20, 95)
        dist_true = np.abs(np.random.normal(10, 6, samples_per_device))
        accel_mag = np.abs(np.random.normal(0.2, 0.15, samples_per_device))
        tamper_true = np.zeros(samples_per_device, dtype=int)

        # operational anomalies: temp (~17%)
        for _ in range(max(1, int(0.17 * samples_per_device / 10))):
            start = np.random.randint(0, samples_per_device-50)
            length = np.random.randint(5, 30)
            temp_true[start:start+length] += np.random.uniform(6.0,10.0)

        # tamper (~10%)
        tamper_idxs = np.random.choice(samples_per_device, size=int(0.10*samples_per_device), replace=False)
        tamper_true[tamper_idxs] = 1

        # cyber (~10%)
        cyber_flags = np.zeros(samples_per_device, dtype=int)
        cyber_idxs = np.random.choice(samples_per_device, size=int(0.10*samples_per_device), replace=False)
        cyber_flags[cyber_idxs] = 1

        # apply cyber transformations
        temp_rep, hum_rep, lat_rep, lon_rep, dist_rep = temp_true.copy(), hum_true.copy(), lats.copy(), lons.copy(), dist_true.copy()
        tamper_rep = tamper_true.copy()
        for idx in cyber_idxs:
            mode = idx % 4
            if mode == 0 and idx > 30:  # replay
                src = idx - np.random.randint(5, 30)
                temp_rep[idx] = temp_rep[src]
                hum_rep[idx] = hum_rep[src]
                lat_rep[idx] = lat_rep[src]
                lon_rep[idx] = lon_rep[src]
                dist_rep[idx] = dist_rep[src]
            elif mode == 1:  # smoothing
                lo, hi = max(0, idx-2), min(samples_per_device-1, idx+2)
                temp_rep[idx] = temp_rep[lo:hi+1].mean()
                hum_rep[idx] = hum_rep[lo:hi+1].mean()
            elif mode == 2 and idx > 0:  # suppression
                temp_rep[idx] = temp_rep[idx-1]
                hum_rep[idx] = hum_rep[idx-1]
                dist_rep[idx] = dist_rep[idx-1]
            else:  # gps spoof
                lat_rep[idx] = center_lat + np.random.normal(0,0.00005)
                lon_rep[idx] = center_lon + np.random.normal(0,0.00005)
                dist_rep[idx] = max(1.0, dist_rep[idx]*0.1)
            if tamper_rep[idx] == 1 and np.random.rand() < 0.5:
                tamper_rep[idx] = 0

        base_time = datetime(2025,1,1)
        timestamps = [base_time + timedelta(minutes=interval_min*i) for i in range(samples_per_device)]
        for i in range(samples_per_device):
            temp_breach = int((temp_true[i] < 2.0) or (temp_true[i] > 8.0))
            loss_flag = temp_breach
            geofence = int(dist_rep[i] > 300)
            label = "normal"
            if tamper_rep[i] == 1:
                label = "Tamper_Damage_anomaly"
            elif cyber_flags[i] == 1:
                label = "Cyberattack_anomaly"
            elif loss_flag == 1:
                label = "Loss_of_storage_condition"
            elif geofence == 1:
                label = "Geofence_anomaly"
            rows.append({
                "Device_ID": dev_id,
                "timestamp": timestamps[i].isoformat(),
                "timestamp_reported": timestamps[i].isoformat(),
                "temp_true": round(float(temp_true[i]),3),
                "temp_reported": round(float(temp_rep[i]),3),
                "humidity_reported": round(float(hum_rep[i]),3),
                "lat_reported": round(float(lat_rep[i]),6),
                "lon_reported": round(float(lon_rep[i]),6),
                "distance_from_route_reported": round(float(dist_rep[i]),3),
                "accel_mag_reported": round(float(accel_mag[i]),3),
                "tamper_reported": int(tamper_rep[i]),
                "cyberattack_anomaly": int(cyber_flags[i]),
                "temperature_anomaly_true": int((temp_true[i] < 2.0) | (temp_true[i] > 8.0)),
                "primary_label_reported": label,
                "cvss": round(cvss,3),
                "epss": round(epss,4),
                "dvd": int(dvd)
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_rebalanced_train()
    df.to_csv("synthetic_train_rebalanced.csv", index=False)
    print("Saved synthetic_train_rebalanced.csv with", len(df), "rows")
    print(df["primary_label_reported"].value_counts())
