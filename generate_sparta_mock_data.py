# -*- coding: utf-8 -*-
"""
###################################################
###################################################
The active research areas at TAN's Laboratory: 
                 Time-sensitive networking (T),
                   AI-driven cybersecurity (A),
            NextG communication networking (N),
Time-series Analysis via Network science (TAN)
###################################################
###################################################
@author:
    Van Le - Tan Le
generate_sparta_mock_data.py

SPARTA / Aerospace Threat-Based synthetic dataset generator.
"""

import numpy as np
import pandas as pd

def generate_mock_sparta_events_csv(path, n_samples=1000, seed=42):
    np.random.seed(seed)
    segments = ["space", "link", "ground"]
    link_statuses = ["nominal", "degraded", "lost"]
    labels = [
        "benign",
        "uplink_jamming",
        "spoofing",
        "unauthorized_command",
        "payload_data_manipulation",
        "ground_segment_compromise",
    ]

    rows = []
    for i in range(n_samples):
        label = np.random.choice(labels, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])
        segment = np.random.choice(segments)
        link_status = np.random.choice(link_statuses, p=[0.7, 0.2, 0.1])

        # RF + BER patterns
        if label == "uplink_jamming":
            rf_power_mean = np.random.normal(10, 2)
            rf_power_var = np.random.normal(5, 1)
            bit_error_rate = np.random.uniform(0.05, 0.2)
        elif label == "spoofing":
            rf_power_mean = np.random.normal(3, 1)
            rf_power_var = np.random.normal(1, 0.3)
            bit_error_rate = np.random.uniform(0.01, 0.05)
        elif label == "unauthorized_command":
            rf_power_mean = np.random.normal(1, 0.5)
            rf_power_var = np.random.normal(0.5, 0.2)
            bit_error_rate = np.random.uniform(0.0, 0.02)
        else:
            rf_power_mean = np.random.normal(1, 0.5)
            rf_power_var = np.random.normal(0.5, 0.2)
            bit_error_rate = np.random.uniform(0.0, 0.01)

        # Command behavior
        cmd_rate = np.random.poisson(5 if label != "unauthorized_command" else 15)
        cmd_invalid_ratio = np.clip(
            np.random.beta(1, 20) if label == "benign" else np.random.beta(2, 5),
            0, 1
        )

        # Mode changes
        mode_change_flag = int(np.random.rand() < (0.05 if label == "benign" else 0.3))

        # Aerospace threat-based physical inconsistencies
        attitude_error = (
            np.random.normal(0.1, 0.05) if label == "benign"
            else np.random.normal(2.0, 0.5)
        )
        thermal_drift = (
            np.random.normal(0.05, 0.02) if label == "benign"
            else np.random.normal(0.5, 0.1)
        )
        payload_data_integrity = (
            np.random.uniform(0.8, 1.0) if label == "benign"
            else np.random.uniform(0.0, 0.6)
        )

        # Generic anomaly score
        anomaly_score = (
            np.random.uniform(0, 0.3) if label == "benign"
            else np.random.uniform(0.4, 1.0)
        )

        rows.append({
            "timestamp": i,
            "segment": segment,
            "rf_power_mean": rf_power_mean,
            "rf_power_var": rf_power_var,
            "cmd_rate": cmd_rate,
            "cmd_invalid_ratio": cmd_invalid_ratio,
            "mode_change_flag": mode_change_flag,
            "bit_error_rate": bit_error_rate,
            "link_status": link_status,
            "attitude_error": attitude_error,
            "thermal_drift": thermal_drift,
            "payload_data_integrity": payload_data_integrity,
            "anomaly_score": anomaly_score,
            "label": label,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Mock SPARTA events saved to {path}")
if __name__ == "__main__":
    output_path = "sparta_events.csv"
    generate_mock_sparta_events_csv(output_path)
    print(f"Dataset generated: {output_path}")
