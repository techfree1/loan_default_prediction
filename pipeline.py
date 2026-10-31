#!/usr/bin/env python3
"""
Bank Muscat Loan Default Prediction & MLOps Pipeline Runner
Executes the full machine learning lifecycle:
1. Data cleaning & banking ratio feature engineering
2. Stratified train/test dataset generation
3. Preprocessing pipeline construction (ColumnTransformer)
4. MLflow-tracked multi-model training (Logistic Regression, Random Forest, XGBoost)
5. Model selection, evaluation metrics, and retail NPA impact calculation
6. SHAP explainability generation (global beeswarm & local waterfall plots)
7. Macroeconomic production drift simulation & EvidentlyAI drift reporting
"""

import os
import sys
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Ensure repository root is on sys.path
PROJECT_ROOT = os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_and_engineer_features, build_preprocessing_pipeline
from src.train_and_evaluate import train_all_models
from src.drift_monitoring import simulate_production_drift, generate_drift_reports

def run_pipeline(project_root: str = PROJECT_ROOT, raw_data_path: str = None):
    print("=" * 70)
    print("🚀 STARTING BANK MUSCAT ML & MLOPS RETRAINING PIPELINE (PROJECT P_137)")
    print(f"📁 Project Root: {project_root}")
    print("=" * 70)

    # 1. Paths verification
    raw_csv = raw_data_path or os.path.join(project_root, "data", "raw_bank_muscat_loans.csv")
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv}")

    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 2. Data Cleaning & Feature Engineering
    print(f"\n[1/6] 📥 Ingesting and engineering banking underwriting ratios from {raw_csv}...")
    df_raw = pd.read_csv(raw_csv)
    df_clean = clean_and_engineer_features(df_raw)
    print(f"       Total processed records: {len(df_clean):,} (outlier sanitization applied)")

    target_col = "loan_status"
    feature_cols = [c for c in df_clean.columns if c != target_col]
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # 3. Stratified Train / Test Split
    print("\n[2/6] ✂️ Creating 80/20 stratified reference train and test splits...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train_export = pd.concat([X_train, y_train], axis=1)
    test_export = pd.concat([X_test, y_test], axis=1)
    train_export.to_csv(os.path.join(data_dir, "reference_train.csv"), index=False)
    test_export.to_csv(os.path.join(data_dir, "current_test.csv"), index=False)
    print(f"       Reference Train set: {len(train_export):,} rows -> data/reference_train.csv")
    print(f"       Current Test set:    {len(test_export):,} rows -> data/current_test.csv")

    # 4. Preprocessing Pipeline
    print("\n[3/6] ⚙️ Fitting ColumnTransformer (StandardScaler + OneHotEncoder)...")
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_names

    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    joblib.dump(all_feature_names, os.path.join(models_dir, "feature_names.pkl"))
    print(f"       Preprocessor saved -> models/preprocessor.pkl")
    print(f"       Features count: {len(all_feature_names)} -> models/feature_names.pkl")

    # 5. Model Training & MLflow Logging
    print("\n[4/6] 🧠 Training candidate algorithms with MLflow tracking...")
    leaderboard_df, trained_models, metadata = train_all_models(
        X_train_proc, y_train, X_test_proc, y_test, project_root=project_root
    )

    print("\n🏆 MODEL EVALUATION LEADERBOARD:")
    print(leaderboard_df.to_string(index=False))

    # Generate ROC Curve Plot
    print("\n[5/6] 📈 Generating evaluation plots and SHAP explainability artifacts...")
    plt.figure(figsize=(9, 7))
    for model_name, model_obj in trained_models.items():
        y_prob = model_obj.predict_proba(X_test_proc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    plt.axvline(0.35, color="red", linestyle=":", alpha=0.7, label="Bank Muscat Cutoff (0.35)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Bank Muscat Retail Credit Risk — Candidate Models ROC Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_plot_path = os.path.join(reports_dir, "roc_comparison.png")
    plt.savefig(roc_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"       ROC curve saved -> {roc_plot_path}")

    # Generate SHAP explanations
    champion_name = metadata.get("best_model_name", "XGBoost")
    champion_model = trained_models.get(champion_name)
    try:
        import shap
        print(f"       Fitting SHAP TreeExplainer on champion ({champion_name})...")
        explainer = shap.TreeExplainer(champion_model)
        joblib.dump(explainer, os.path.join(models_dir, "shap_explainer.pkl"))

        sample_subset = X_test_proc[:500]
        shap_values = explainer(sample_subset)

        # Global Beeswarm Plot
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.title(f"Bank Muscat Credit Risk — Global SHAP Drivers ({champion_name})", fontsize=12)
        beeswarm_path = os.path.join(reports_dir, "shap_global_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Local Waterfall Plot (Single Applicant)
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.title(f"Bank Muscat — Local Underwriting SHAP Explanation (Sample Applicant #0)", fontsize=11)
        waterfall_path = os.path.join(reports_dir, "shap_local_waterfall.png")
        plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"       SHAP explainer saved -> models/shap_explainer.pkl")
        print(f"       SHAP beeswarm saved  -> {beeswarm_path}")
        print(f"       SHAP waterfall saved -> {waterfall_path}")
    except Exception as e:
        print(f"       ⚠️ SHAP plotting warning: {e}")

    # 6. Macroeconomic Stress Simulation & EvidentlyAI Drift Monitoring
    print("\n[6/6] 📡 Simulating macroeconomic production stress & computing drift...")
    drifted_df = simulate_production_drift(test_export)
    drifted_csv_path = os.path.join(data_dir, "production_drifted_batch.csv")
    drifted_df.to_csv(drifted_csv_path, index=False)
    print(f"       Production drifted batch saved -> {drifted_csv_path}")

    drift_summary = generate_drift_reports(
        reference_df=train_export,
        current_df=drifted_df,
        project_root=project_root
    )
    print(f"       Monitoring status: {drift_summary.get('status')}")
    print(f"       Drift share:       {drift_summary.get('drift_share_pct')}% ({drift_summary.get('drifted_features_count')}/{drift_summary.get('total_features_monitored')} features)")
    print(f"       Drift summary      -> reports/drift_summary.json")
    print(f"       Evidently report   -> reports/evidently_drift_report.html")

    print("\n" + "=" * 70)
    print("✅ RETRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Champion Model:      {metadata.get('best_model_name')}")
    print(f"   Champion ROC-AUC:    {metadata.get('roc_auc_pct')}%")
    print(f"   Relative NPA Red.:   {metadata.get('npa_metrics', {}).get('relative_npa_reduction_pct')}%")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank Muscat Credit Risk MLOps Pipeline Runner")
    parser.add_argument("--project-root", default=PROJECT_ROOT, help="Absolute path to project directory")
    parser.add_argument("--data-path", default=None, help="Path to raw CSV dataset")
    args = parser.parse_args()
    run_pipeline(project_root=args.project_root, raw_data_path=args.data_path)
