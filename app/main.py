import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.database import get_db_engine, log_prediction_to_db

# Dynamically resolve project root (defaults to parent directory of app/)
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent)
)

app = FastAPI(
    title="AI Loan Default Prediction & Risk Intelligence API",
    version="1.0.0",
    description="Real-time loan credit risk evaluation API with SHAP explainability and audit logging."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compatibility fix for scikit-learn 1.6+ unpickling ColumnTransformer
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# Load artifacts dynamically at startup
model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
preprocessor_path = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")
explainer_path = os.path.join(PROJECT_ROOT, "models", "shap_explainer.pkl")
feature_names_path = os.path.join(PROJECT_ROOT, "models", "feature_names.pkl")
metadata_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

model = joblib.load(model_path) if os.path.exists(model_path) else None

preprocessor = None
if os.path.exists(preprocessor_path):
    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception:
        try:
            from src.preprocessing import clean_and_engineer_features, build_preprocessing_pipeline
            train_csv = os.path.join(PROJECT_ROOT, "data", "reference_train.csv")
            if os.path.exists(train_csv):
                df_train = pd.read_csv(train_csv)
                df_train = clean_and_engineer_features(df_train)
                target_col = "loan_status" if "loan_status" in df_train.columns else df_train.columns[-1]
                X_train = df_train.drop(columns=[target_col], errors="ignore")
                pipe = build_preprocessing_pipeline(X_train)
                preprocessor = pipe.fit(X_train)
        except Exception:
            preprocessor = None

explainer = None
if os.path.exists(explainer_path):
    try:
        explainer = joblib.load(explainer_path)
    except Exception:
        explainer = None

feature_names = joblib.load(feature_names_path) if os.path.exists(feature_names_path) else []

metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

db_engine, _ = get_db_engine(project_root=PROJECT_ROOT)

class LoanApplicationRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=90, description="Applicant age")
    person_income: float = Field(..., gt=0, description="Annual income in USD/OMR")
    person_emp_length: float = Field(..., ge=0, le=60, description="Employment length in years")
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(..., gt=0, le=40, description="Interest rate percentage")
    loan_grade: str = Field(..., description="Loan grade (A-G)")
    loan_intent: str = Field(..., description="Loan intent (PERSONAL, EDUCATION, etc.)")
    cb_person_default_on_file: str = Field(default="N", description="Historical default on file (Y/N)")
    cb_person_cred_hist_length: float = Field(default=4.0, description="Credit history length in years")
    person_home_ownership: str = Field(default="RENT", description="Home ownership status")

@app.get("/")
def read_root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        frontend_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
        if os.path.exists(frontend_path):
            with open(frontend_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    return {
        "status": "Online",
        "project": "AI Loan Default Prediction & Credit Risk Platform",
        "champion_model": metadata.get("best_model_name", "XGBoost"),
        "roc_auc_pct": metadata.get("roc_auc_pct", 94.83)
    }

@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "explainer_loaded": explainer is not None,
        "db_connected": db_engine is not None
    }

@app.post("/score")
def score_loan_application(payload: LoanApplicationRequest):
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded. Ensure models/ directory is populated.")

    data_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    # Compute engineered features required by preprocessor
    data_dict["loan_percent_income"] = data_dict["loan_amnt"] / (data_dict["person_income"] + 1.0)
    data_dict["loan_to_income_ratio"] = data_dict["loan_amnt"] / (data_dict["person_income"] + 1.0)
    data_dict["interest_burden_ratio"] = (data_dict["loan_amnt"] * (data_dict["loan_int_rate"] / 100.0)) / (data_dict["person_income"] + 1.0)
    data_dict["income_per_emp_year"] = data_dict["person_income"] / (data_dict["person_emp_length"] + 1.0)
    data_dict["credit_maturity_ratio"] = data_dict["cb_person_cred_hist_length"] / max(data_dict["person_age"] - 18.0, 1.0)

    input_df = pd.DataFrame([data_dict])
    processed_matrix = preprocessor.transform(input_df)

    prob = float(model.predict_proba(processed_matrix)[0][1])
    threshold = metadata.get("optimal_threshold", 0.35)
    decision = "REJECTED" if prob >= threshold else "APPROVED"
    risk_tier = "HIGH RISK" if prob >= 0.50 else ("MEDIUM RISK" if prob >= threshold else "LOW RISK")

    top_drivers = []
    if explainer:
        shap_vals = explainer(processed_matrix)
        vals = shap_vals.values[0] if hasattr(shap_vals, "values") else shap_vals[0]
        # In case SHAP output has dimensions for both classes (e.g. 2D array per sample)
        if hasattr(vals, "shape") and len(vals.shape) > 1 and vals.shape[-1] == 2:
            vals = vals[:, 1]
        sorted_indices = np.argsort(np.abs(vals))[::-1][:4]
        for idx in sorted_indices:
            top_drivers.append({
                "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "shap_impact": round(float(vals[idx]), 4)
            })

    result = {
        "default_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "decision": decision,
        "optimal_threshold": threshold,
        "top_shap_drivers": top_drivers
    }

    try:
        log_prediction_to_db(db_engine, data_dict, result)
    except Exception as e:
        print(f"Database logging warning: {e}")

    return result

@app.get("/monitoring/drift")
def get_drift_status():
    drift_summary_path = os.path.join(PROJECT_ROOT, "reports", "drift_summary.json")
    if os.path.exists(drift_summary_path):
        with open(drift_summary_path, "r") as f:
            return json.load(f)
    return {"status": "Drift report not generated yet."}

@app.get("/metrics/npa")
def get_npa_metrics():
    return metadata.get("npa_metrics", {})

@app.get("/audit-logs")
def get_audit_logs(limit: int = 25):
    if not db_engine:
        return []
    try:
        from sqlalchemy.orm import sessionmaker
        from app.database import LoanPredictionLog
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        with SessionLocal() as session:
            logs = session.query(LoanPredictionLog).order_by(LoanPredictionLog.id.desc()).limit(limit).all()
            return [
                {
                    "id": log.id,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                    "person_age": log.person_age,
                    "person_income": log.person_income,
                    "loan_amnt": log.loan_amnt,
                    "loan_int_rate": log.loan_int_rate,
                    "loan_grade": log.loan_grade,
                    "default_probability": log.default_probability,
                    "risk_tier": log.risk_tier,
                    "decision": log.decision,
                }
                for log in logs
            ]
    except Exception as e:
        print(f"Error reading audit logs: {e}")
        return []

@app.get("/reports/evidently_drift_report.html")
def get_evidently_report():
    report_path = os.path.join(PROJECT_ROOT, "reports", "evidently_drift_report.html")
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Evidently report file not found.")


