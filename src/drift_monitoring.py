import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def simulate_production_drift(df_current: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Simulates a macroeconomic stress scenario on incoming Bank Muscat loan applications
    (higher interest rates, lower borrower incomes, higher loan-to-income ratios).
    """
    rng = np.random.default_rng(random_state)
    drifted = df_current.copy()

    # Simulate macroeconomic shift in production batch
    drifted["loan_int_rate"] = np.clip(drifted["loan_int_rate"] + rng.normal(2.8, 0.8, len(drifted)), 5.0, 28.0)
    drifted["person_income"] = np.maximum(drifted["person_income"] * rng.uniform(0.78, 0.92, len(drifted)), 4000.0)
    drifted["loan_amnt"] = drifted["loan_amnt"] * rng.uniform(1.10, 1.25, len(drifted))

    # Recompute affected underwriting ratios
    drifted["loan_to_income_ratio"] = drifted["loan_amnt"] / (drifted["person_income"] + 1.0)
    drifted["interest_burden_ratio"] = (
        (drifted["loan_amnt"] * (drifted["loan_int_rate"] / 100.0)) / (drifted["person_income"] + 1.0)
    )
    return drifted

def generate_drift_reports(reference_df: pd.DataFrame, current_df: pd.DataFrame, project_root: str):
    """
    Generates an EvidentlyAI HTML drift report and a structured JSON drift summary
    for the FastAPI /monitoring/drift endpoint and Streamlit dashboard.
    """
    monitored_cols = [
        "person_age", "person_income", "person_emp_length", "loan_amnt",
        "loan_int_rate", "loan_to_income_ratio", "interest_burden_ratio",
        "credit_maturity_ratio"
    ]

    ref_sub = reference_df[monitored_cols].sample(n=min(2000, len(reference_df)), random_state=42)
    cur_sub = current_df[monitored_cols].sample(n=min(2000, len(current_df)), random_state=42)

    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    html_report_path = os.path.join(reports_dir, "evidently_drift_report.html")
    evidently_used = False

    # 1. Try EvidentlyAI Report (supports both 0.4.x/0.5.x and 0.6+ legacy namespace)
    try:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_sub, current_data=cur_sub)
        report.save_html(html_report_path)
        evidently_used = True
    except Exception as e:
        print(f"ℹ️ Using built-in HTML drift renderer ({e})")

    # 2. Compute exact feature-level KS-test drift metrics for API & Streamlit UI
    feature_drift_list = []
    drifted_count = 0

    for col in monitored_cols:
        ks_stat, p_val = ks_2samp(ref_sub[col].dropna(), cur_sub[col].dropna())
        is_drifted = bool(p_val < 0.01 and ks_stat > 0.08)
        if is_drifted:
            drifted_count += 1

        feature_drift_list.append({
            "feature": col,
            "reference_mean": round(float(ref_sub[col].mean()), 3),
            "current_mean": round(float(cur_sub[col].mean()), 3),
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 6),
            "drift_detected": is_drifted
        })

    drift_share = round((drifted_count / len(monitored_cols)) * 100.0, 2)
    dataset_drift = bool(drift_share >= 25.0)

    drift_summary = {
        "dataset_drift_detected": dataset_drift,
        "drifted_features_count": drifted_count,
        "total_features_monitored": len(monitored_cols),
        "drift_share_pct": drift_share,
        "status": "⚠️ DRIFT ALERT: Retraining Recommended" if dataset_drift else "✅ HEALTHY: No Significant Drift",
        "html_report_path": "reports/evidently_drift_report.html",
        "features": feature_drift_list
    }

    # Save fallback HTML if Evidently HTML was not written
    if not evidently_used:
        df_html = pd.DataFrame(feature_drift_list).to_html(index=False)
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h2>Bank Muscat EvidentlyAI Drift Report</h2>{df_html}</body></html>")

    summary_path = os.path.join(reports_dir, "drift_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(drift_summary, f, indent=4)

    return drift_summary

