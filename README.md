# Unified Operational–Cyber Anomaly Intelligence for Risk-Aware IoT Cold-Chain Monitoring

This repository contains the reference implementation and experimental artifacts for the paper:

**Unified Operational–Cyber Anomaly Intelligence for Risk-Aware IoT Blood Bank Cold-Chain Monitoring**

The code is provided to support reproducibility and to serve as a reference implementation corresponding to the conference submission.

---

## Overview

This work proposes a unified anomaly intelligence framework for IoT-based cold-chain monitoring systems, jointly addressing:

- **Operational anomalies** (e.g., temperature excursions, sensor drift, route deviations)
- **Cyber anomalies** (e.g., replay attacks, spoofing, timing manipulation)

A risk-aware decision layer fuses operational and cyber anomaly signals to generate context-sensitive alerts suitable for safety-critical cold-chain applications such as vaccine and blood transport.

---

## Key Contributions

- Dual-branch anomaly detection architecture (operational + cyber)
- Persistence-aware and posture-aware anomaly reasoning
- Risk-weighted alert generation for cold-chain monitoring
- Experimental evaluation on a realistic IoT cold-chain dataset

---

## Running the Code

The repository provides Streamlit-based interfaces for both baseline methods and the proposed unified anomaly intelligence model.

### Baseline Models
To run the baseline anomaly detection methods:

```bash
python -m streamlit run baselines.py
```

### Proposed Model
To run the proposed unified operational–cyber anomaly intelligence framework:
``` bash
python -m streamlit run Streamlit.py
```

The Streamlit interface enables interactive visualization of anomaly scores, alerts, and risk-aware decisions.

### Notes
The implementation and experimental configuration are aligned with the accompanying paper.

Additional experimental details, ablation studies, and extended results are intended to support reproducibility.

This repository is provided as a reference implementation for research and academic use.