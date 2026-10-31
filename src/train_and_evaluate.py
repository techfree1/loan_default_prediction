import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)

def calculate_npa_metrics(y_true, y_prob, threshold: float = 0.35):
    y_true_arr = np.asarray(y_true)
    baseline_npa_rate = float(np.mean(y_true_arr))
    total_loans = len(y_true_arr)
    total_defaults = int(np.sum(y_true_arr))

    approved_mask = y_prob < threshold
    approved_loans = int(np.sum(approved_mask))
    
    if approved_loans == 0:
        return {
            "threshold": threshold,
            "baseline_npa_pct": round(baseline_npa_rate * 100, 2),
            "post_model_npa_pct": 0.0,
            "relative_npa_reduction_pct": 100.0,
            "defaults_prevented_pct": 100.0,
            "approval_rate_pct": 0.0
        }

    approved_outcomes = np.compress(approved_mask, y_true_arr)
    post_model_npa_rate = float(np.mean(approved_outcomes))
    relative_npa_reduction = ((baseline_npa_rate - post_model_npa_rate) / baseline_npa_rate) * 100.0
    defaults_prevented = (1.0 - (np.sum(approved_outcomes) / total_defaults)) * 100.0
    approval_rate = (approved_loans / total_loans) * 100.0

    return {
        "threshold": round(float(threshold), 2),
        "baseline_npa_pct": round(baseline_npa_rate * 100, 2),
        "post_model_npa_pct": round(post_model_npa_rate * 100, 2),
        "relative_npa_reduction_pct": round(relative_npa_reduction, 2),
        "defaults_prevented_pct": round(defaults_prevented, 2),
        "approval_rate_pct": round(approval_rate, 2)
    }

def train_all_models(X_train, y_train, X_test, y_test, project_root: str):
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    mlflow_db_path = f"sqlite:///{os.path.join(project_root, 'mlflow.db')}"
    mlflow.set_tracking_uri(mlflow_db_path)
    mlflow.set_experiment("Bank_Muscat_Loan_Default_P137")

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, base_score=0.5,
            eval_metric="auc", random_state=42, n_jobs=-1
        )
    }

    leaderboard = list()
    trained_models = dict()
    auc_scores = dict()
    best_auc = -1.0
    best_model_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_prob = np.take(model.predict_proba(X_test), 1, axis=1)
            y_pred = (y_prob >= 0.35).astype(int)

            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            npa_stats = calculate_npa_metrics(y_test, y_prob, threshold=0.35)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("relative_npa_reduction_pct", npa_stats.get("relative_npa_reduction_pct"))
            mlflow.log_metric("defaults_prevented_pct", npa_stats.get("defaults_prevented_pct"))

            model_path = os.path.join(models_dir, f"{name.lower()}.pkl")
            joblib.dump(model, model_path)
            trained_models.update({name: model})
            auc_scores.update({name: round(roc_auc * 100, 2)})

            leaderboard.append({
                "Model": name,
                "ROC-AUC (%)": round(roc_auc * 100, 2),
                "PR-AUC (%)": round(pr_auc * 100, 2),
                "F1-Score": round(f1, 3),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "Baseline NPA (%)": npa_stats.get("baseline_npa_pct"),
                "Post-Model NPA (%)": npa_stats.get("post_model_npa_pct"),
                "NPA Reduction (%)": npa_stats.get("relative_npa_reduction_pct"),
                "Defaults Caught (%)": npa_stats.get("defaults_prevented_pct"),
                "Target >=85% AUC Met": "YES" if roc_auc >= 0.85 else "NO",
                "Target >=30% NPA Red. Met": "YES" if npa_stats.get("relative_npa_reduction_pct") >= 30.0 else "NO"
            })

            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model_name = name

    best_model = trained_models.get(best_model_name)
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))

    best_probs = np.take(best_model.predict_proba(X_test), 1, axis=1)
    best_npa = calculate_npa_metrics(y_test, best_probs, threshold=0.35)
    metadata = {
        "best_model_name": best_model_name,
        "roc_auc_pct": round(best_auc * 100, 2),
        "optimal_threshold": 0.35,
        "auc_scores": auc_scores,
        "npa_metrics": best_npa,
        "leaderboard": leaderboard
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return pd.DataFrame(leaderboard), trained_models, metadata
