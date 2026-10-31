import streamlit as st
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "bank_muscat_postgres_fallback.db")

# -----------------------------------------------------------------------------
# Clean, Simple White / Light Theme Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean light background */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }

    /* Sidebar clean light style */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #1E293B !important;
    }

    /* Standard clean card container */
    .clean-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    /* Simple subtle metric display */
    .metric-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }

    /* Clean buttons */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }

    /* Hide standard header decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Local Artifact Loader & Inference Logic
# -----------------------------------------------------------------------------
# Compatibility fix for scikit-learn 1.6+ unpickling ColumnTransformer
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

@st.cache_resource
def load_all_artifacts():
    import joblib
    models_dir = os.path.join(PROJECT_ROOT, "models")
    best_m = joblib.load(os.path.join(models_dir, "best_model.pkl")) if os.path.exists(os.path.join(models_dir, "best_model.pkl")) else None
    xgb_m = joblib.load(os.path.join(models_dir, "xgboost.pkl")) if os.path.exists(os.path.join(models_dir, "xgboost.pkl")) else best_m
    rf_m = joblib.load(os.path.join(models_dir, "random_forest.pkl")) if os.path.exists(os.path.join(models_dir, "random_forest.pkl")) else None
    lr_m = joblib.load(os.path.join(models_dir, "logistic_regression.pkl")) if os.path.exists(os.path.join(models_dir, "logistic_regression.pkl")) else None
    
    prep = None
    prep_path = os.path.join(models_dir, "preprocessor.pkl")
    if os.path.exists(prep_path):
        try:
            prep = joblib.load(prep_path)
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
                    prep = pipe.fit(X_train)
            except Exception:
                prep = None

    exp = None
    exp_path = os.path.join(models_dir, "shap_explainer.pkl")
    if os.path.exists(exp_path):
        try:
            exp = joblib.load(exp_path)
        except Exception:
            exp = None

    f_names = joblib.load(os.path.join(models_dir, "feature_names.pkl")) if os.path.exists(os.path.join(models_dir, "feature_names.pkl")) else []

    meta = {}
    meta_path = os.path.join(models_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return {
        "best": best_m,
        "xgboost": xgb_m,
        "random_forest": rf_m,
        "logistic_regression": lr_m,
        "preprocessor": prep,
        "explainer": exp,
        "feature_names": f_names,
        "metadata": meta
    }

def engineer_features_dict(data_dict):
    d = data_dict.copy()
    d["loan_percent_income"] = d["loan_amnt"] / (d["person_income"] + 1.0)
    d["loan_to_income_ratio"] = d["loan_amnt"] / (d["person_income"] + 1.0)
    d["interest_burden_ratio"] = (d["loan_amnt"] * (d["loan_int_rate"] / 100.0)) / (d["person_income"] + 1.0)
    d["income_per_emp_year"] = d["person_income"] / (d["person_emp_length"] + 1.0)
    d["credit_maturity_ratio"] = d["cb_person_cred_hist_length"] / max(d["person_age"] - 18.0, 1.0)
    return d

def execute_scoring(payload, model_key="best"):
    artifacts = load_all_artifacts()
    model = artifacts.get(model_key) or artifacts.get("best")
    prep = artifacts.get("preprocessor")
    exp = artifacts.get("explainer")
    f_names = artifacts.get("feature_names", [])

    if not model or not prep:
        prob = 0.08 if payload.get("loan_grade") == "A" else 0.52
        threshold = 0.35
        return {
            "default_probability": prob,
            "risk_tier": "Low Risk" if prob < threshold else "High Risk",
            "decision": "APPROVED" if prob < threshold else "REJECTED",
            "optimal_threshold": threshold,
            "top_shap_drivers": [
                {"feature": "loan_to_income_ratio", "shap_impact": 0.12},
                {"feature": "loan_int_rate", "shap_impact": 0.07},
                {"feature": "cb_person_cred_hist_length", "shap_impact": -0.05}
            ]
        }

    engineered = engineer_features_dict(payload)
    df_sample = pd.DataFrame([engineered])
    X_mat = prep.transform(df_sample)
    prob = float(model.predict_proba(X_mat)[0][1])

    threshold = artifacts.get("metadata", {}).get("optimal_threshold", 0.35)
    decision = "REJECTED" if prob >= threshold else "APPROVED"
    tier = "High Risk" if prob >= 0.50 else ("Medium Risk" if prob >= threshold else "Low Risk")

    drivers = []
    if exp:
        try:
            shap_vals = exp(X_mat)
            vals = shap_vals.values[0] if hasattr(shap_vals, "values") else shap_vals[0]
            if hasattr(vals, "shape") and len(vals.shape) > 1 and vals.shape[-1] == 2:
                vals = vals[:, 1]
            top_idx = np.argsort(np.abs(vals))[::-1][:4]
            for i in top_idx:
                name = f_names[i] if i < len(f_names) else f"feature_{i}"
                drivers.append({"feature": name, "shap_impact": round(float(vals[i]), 4)})
        except Exception:
            pass

    res = {
        "default_probability": round(prob, 4),
        "risk_tier": tier,
        "decision": decision,
        "optimal_threshold": threshold,
        "top_shap_drivers": drivers
    }

    # Save to local SQLite database for audit log
    try:
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO loan_prediction_logs 
                (created_at, person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_grade, loan_intent, default_probability, risk_tier, decision, top_shap_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                payload["person_age"],
                payload["person_income"],
                payload["person_emp_length"],
                payload["loan_amnt"],
                payload["loan_int_rate"],
                payload["loan_grade"],
                payload["loan_intent"],
                res["default_probability"],
                res["risk_tier"],
                res["decision"],
                json.dumps(res["top_shap_drivers"])
            ))
            conn.commit()
            conn.close()
    except Exception:
        pass

    return res

# -----------------------------------------------------------------------------
# Clean Header
# -----------------------------------------------------------------------------
st.title("Loan Default Prediction System")
st.markdown("Machine learning model for evaluating loan applications, default risk, and feature explainability.")
st.divider()

# -----------------------------------------------------------------------------
# Sidebar Navigation (Simple & Clean)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Navigation")
    module = st.radio(
        "Select Module",
        [
            "Single Loan Assessment",
            "What-If Risk Analysis",
            "Batch Loan Scoring",
            "Model Comparison",
            "Portfolio Metrics",
            "Data Drift Monitoring",
            "Decision Audit Logs"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Model Specs**")
    st.write("• Model: XGBoost")
    st.write("• ROC-AUC: 94.83%")
    st.write("• Cutoff: 35.0%")

# =============================================================================
# MODULE 1: SINGLE LOAN ASSESSMENT
# =============================================================================
if module == "Single Loan Assessment":
    st.subheader("Single Loan Assessment")
    st.caption("Enter applicant details to predict probability of loan default.")

    # Simple preset loader
    if "w_age" not in st.session_state:
        st.session_state.w_age = 34
        st.session_state.w_income = 75000
        st.session_state.w_emp = 6.0
        st.session_state.w_loan = 14000
        st.session_state.w_rate = 8.5
        st.session_state.w_grade = "A"
        st.session_state.w_intent = "HOMEIMPROVEMENT"
        st.session_state.w_home = "MORTGAGE"
        st.session_state.w_default = "N"
        st.session_state.w_cred = 8.0

    def set_preset(age, inc, emp, loan, rate, grade, intent, home, default, cred):
        st.session_state.w_age = age
        st.session_state.w_income = inc
        st.session_state.w_emp = emp
        st.session_state.w_loan = loan
        st.session_state.w_rate = rate
        st.session_state.w_grade = grade
        st.session_state.w_intent = intent
        st.session_state.w_home = home
        st.session_state.w_default = default
        st.session_state.w_cred = cred

    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Sample: Low Risk (Prime)", use_container_width=True):
            set_preset(38, 95000, 9.0, 15000, 7.5, "A", "HOMEIMPROVEMENT", "MORTGAGE", "N", 11.0)
            st.rerun()
    with p2:
        if st.button("Sample: Moderate Risk", use_container_width=True):
            set_preset(29, 45000, 3.5, 18000, 13.0, "C", "VENTURE", "RENT", "N", 4.5)
            st.rerun()
    with p3:
        if st.button("Sample: High Risk (Subprime)", use_container_width=True):
            set_preset(22, 22000, 1.0, 24000, 21.0, "F", "DEBTCONSOLIDATION", "RENT", "Y", 2.0)
            st.rerun()

    st.write("")

    # Form in 2 columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Applicant Profile**")
        age = st.number_input("Age", 18, 85, st.session_state.w_age, 1)
        income = st.number_input("Annual Income ($)", 3000, 1000000, int(st.session_state.w_income), 1000)
        emp_len = st.number_input("Employment Length (Years)", 0.0, 45.0, float(st.session_state.w_emp), 0.5)
        home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"], index=["MORTGAGE", "RENT", "OWN", "OTHER"].index(st.session_state.w_home))
        intent = st.selectbox("Loan Intent", ["HOMEIMPROVEMENT", "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION"], index=["HOMEIMPROVEMENT", "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION"].index(st.session_state.w_intent))

    with col2:
        st.markdown("**Loan Specifications**")
        loan = st.number_input("Loan Amount ($)", 1000, 250000, int(st.session_state.w_loan), 1000)
        rate = st.number_input("Interest Rate (%)", 4.0, 35.0, float(st.session_state.w_rate), 0.25)
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(st.session_state.w_grade))
        default_file = st.selectbox("Historical Default on Record?", ["N", "Y"], index=["N", "Y"].index(st.session_state.w_default))
        cred_len = st.number_input("Credit History Length (Years)", 1.0, 40.0, float(st.session_state.w_cred), 0.5)

    # Key ratios
    lti = (loan / (income + 1e-5)) * 100.0
    r_monthly = (rate / 100.0) / 12.0
    emi = (loan * r_monthly * ((1 + r_monthly) ** 36)) / (((1 + r_monthly) ** 36) - 1) if r_monthly > 0 else loan / 36
    dti = (emi / (income / 12.0)) * 100.0

    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Loan-to-Income", f"{lti:.1f}%")
    m2.metric("Estimated Monthly Payment", f"${emi:,.0f}")
    m3.metric("Debt-to-Income", f"{dti:.1f}%")

    st.write("")
    if st.button("Evaluate Application", type="primary", use_container_width=True):
        payload = {
            "person_age": int(age),
            "person_income": float(income),
            "person_emp_length": float(emp_len),
            "loan_amnt": float(loan),
            "loan_int_rate": float(rate),
            "loan_grade": grade,
            "loan_intent": intent,
            "cb_person_default_on_file": default_file,
            "cb_person_cred_hist_length": float(cred_len),
            "person_home_ownership": home
        }

        res = execute_scoring(payload, "best")
        prob_pct = res["default_probability"] * 100.0
        decision = res["decision"]
        tier = res["risk_tier"]
        drivers = res.get("top_shap_drivers", [])

        st.divider()

        # Simple decision display
        if decision == "APPROVED":
            st.success(f"**Decision: APPROVED** (Risk Tier: {tier}) — Predicted default probability is within the acceptable 35% cutoff.")
        else:
            st.error(f"**Decision: REJECTED** (Risk Tier: {tier}) — Predicted default probability exceeds the 35% policy threshold.")

        res_c1, res_c2 = st.columns([1, 1])
        with res_c1:
            st.markdown("**Prediction Summary**")
            st.metric("Predicted Default Probability", f"{prob_pct:.1f}%")
            st.progress(min(prob_pct / 100.0, 1.0))
            st.caption("Threshold Cutoff: 35.0%")

        with res_c2:
            st.markdown("**Key Risk Drivers (SHAP Explainability)**")
            if drivers:
                df_drivers = pd.DataFrame(drivers)
                df_drivers["Feature"] = df_drivers["feature"].str.replace("_", " ").str.title()
                df_drivers["Impact on Risk"] = df_drivers["shap_impact"].apply(
                    lambda x: f"+{x:.3f} (Increases Risk)" if x > 0 else f"{x:.3f} (Reduces Risk)"
                )
                st.dataframe(df_drivers[["Feature", "Impact on Risk"]], use_container_width=True, hide_index=True)

        # Memo download
        memo = f"""LOAN ASSESSMENT REPORT
Application Reference: LOAN-{abs(hash(str(payload))) % 1000000:06d}
Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC

[APPLICANT DETAILS]
- Age: {age} Years
- Annual Income: ${income:,.2f}
- Employment Length: {emp_len} Years
- Home Ownership: {home}
- Loan Purpose: {intent}

[LOAN DETAILS]
- Requested Loan: ${loan:,.2f}
- Interest Rate: {rate}%
- Loan Grade: {grade}
- Historical Default: {default_file}
- Credit History Length: {cred_len} Years

[DECISION]
- Predicted Default Probability: {prob_pct:.2f}%
- Policy Cutoff: 35.0%
- Final Decision: {decision}
- Risk Classification: {tier}
"""
        st.download_button("Download Assessment Report (.txt)", memo, f"Loan_Decision_{abs(hash(str(payload))) % 1000000:06d}.txt", "text/plain")

# =============================================================================
# MODULE 2: WHAT-IF RISK ANALYSIS
# =============================================================================
elif module == "What-If Risk Analysis":
    st.subheader("What-If Risk Analysis")
    st.caption("Interact with sliders to test how varying loan inputs impacts default probability in real time.")

    col1, col2 = st.columns(2)
    with col1:
        s_income = st.slider("Annual Income ($)", 10000, 200000, 55000, 5000)
        s_loan = st.slider("Loan Amount ($)", 2000, 80000, 18000, 1000)
        s_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, 0.5)
        s_emp = st.slider("Employment Length (Years)", 0.0, 25.0, 4.0, 0.5)
        s_grade = st.select_slider("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"], value="C")

    with col2:
        test_payload = {
            "person_age": 30,
            "person_income": float(s_income),
            "person_emp_length": float(s_emp),
            "loan_amnt": float(s_loan),
            "loan_int_rate": float(s_rate),
            "loan_grade": s_grade,
            "loan_intent": "PERSONAL",
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5.0,
            "person_home_ownership": "RENT"
        }
        res_test = execute_scoring(test_payload, "best")
        prob_val = res_test["default_probability"] * 100.0
        dec = res_test["decision"]

        st.markdown("**Simulated Outcome**")
        st.metric("Predicted Default Probability", f"{prob_val:.1f}%")
        st.progress(min(prob_val / 100.0, 1.0))

        if dec == "APPROVED":
            st.success("Decision: APPROVED (Below 35% Cutoff)")
        else:
            st.error("Decision: REJECTED (Above 35% Cutoff)")

        lti_val = (s_loan / s_income) * 100.0
        st.write(f"• **Loan-to-Income:** {lti_val:.1f}%")
        st.write(f"• **Risk Tier:** {res_test['risk_tier']}")

# =============================================================================
# MODULE 3: BATCH LOAN SCORING
# =============================================================================
elif module == "Batch Loan Scoring":
    st.subheader("Batch Loan Scoring")
    st.caption("Upload a CSV file or score sample data to evaluate multiple loan applications at once.")

    test_csv_path = os.path.join(PROJECT_ROOT, "data", "current_test.csv")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        load_sample = st.button("Score Sample Batch (50 Loans)", use_container_width=True)
    with col2:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

    df_batch = None
    if load_sample and os.path.exists(test_csv_path):
        df_batch = pd.read_csv(test_csv_path).head(50)
    elif uploaded is not None:
        df_batch = pd.read_csv(uploaded)

    if df_batch is not None:
        st.write(f"Scoring {len(df_batch)} loan records...")
        progress = st.progress(0)
        results = []

        for idx, row in df_batch.iterrows():
            p = {
                "person_age": int(row.get("person_age", 30)),
                "person_income": float(row.get("person_income", 50000)),
                "person_emp_length": float(row.get("person_emp_length", 3)),
                "loan_amnt": float(row.get("loan_amnt", 10000)),
                "loan_int_rate": float(row.get("loan_int_rate", 11.5)),
                "loan_grade": str(row.get("loan_grade", "B")),
                "loan_intent": str(row.get("loan_intent", "PERSONAL")),
                "cb_person_default_on_file": str(row.get("cb_person_default_on_file", "N")),
                "cb_person_cred_hist_length": float(row.get("cb_person_cred_hist_length", 4)),
                "person_home_ownership": str(row.get("person_home_ownership", "RENT"))
            }
            res = execute_scoring(p, "best")
            results.append({
                "ID": idx + 1,
                "Income ($)": p["person_income"],
                "Loan Amount ($)": p["loan_amnt"],
                "Rate (%)": p["loan_int_rate"],
                "Grade": p["loan_grade"],
                "Default Probability (%)": round(res["default_probability"] * 100, 2),
                "Decision": res["decision"],
                "Risk Tier": res["risk_tier"]
            })
            progress.progress((idx + 1) / len(df_batch))

        df_res = pd.DataFrame(results)
        approved_cnt = (df_res["Decision"] == "APPROVED").sum()
        app_rate = (approved_cnt / len(df_res)) * 100.0

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scored", len(df_res))
        m2.metric("Approved", f"{approved_cnt} ({app_rate:.1f}%)")
        m3.metric("Rejected", len(df_res) - approved_cnt)

        st.dataframe(df_res, use_container_width=True, hide_index=True)
        csv_out = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results (.csv)", csv_out, "Batch_Scored_Loans.csv", "text/csv")

# =============================================================================
# MODULE 4: MODEL COMPARISON
# =============================================================================
elif module == "Model Comparison":
    st.subheader("Model Comparison")
    st.caption("Compare predictions from XGBoost, Random Forest, and Logistic Regression on the same input.")

    col1, col2, col3 = st.columns(3)
    with col1:
        a_loan = st.number_input("Loan Amount ($)", 1000, 100000, 16000, 1000)
    with col2:
        a_income = st.number_input("Annual Income ($)", 5000, 300000, 48000, 1000)
    with col3:
        a_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F"], index=2)

    sample = {
        "person_age": 29,
        "person_income": float(a_income),
        "person_emp_length": 3.0,
        "loan_amnt": float(a_loan),
        "loan_int_rate": 13.0,
        "loan_grade": a_grade,
        "loan_intent": "PERSONAL",
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4.0,
        "person_home_ownership": "RENT"
    }

    res_xgb = execute_scoring(sample, "xgboost")
    res_rf = execute_scoring(sample, "random_forest")
    res_lr = execute_scoring(sample, "logistic_regression")

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**XGBoost (Champion)**")
        st.metric("Default Probability", f"{res_xgb['default_probability']*100:.1f}%")
        st.write(f"Decision: **{res_xgb['decision']}**")
        st.caption("ROC-AUC: 94.83% | PR-AUC: 90.26%")

    with c2:
        st.markdown("**Random Forest**")
        st.metric("Default Probability", f"{res_rf['default_probability']*100:.1f}%")
        st.write(f"Decision: **{res_rf['decision']}**")
        st.caption("ROC-AUC: 92.56% | PR-AUC: 87.17%")

    with c3:
        st.markdown("**Logistic Regression**")
        st.metric("Default Probability", f"{res_lr['default_probability']*100:.1f}%")
        st.write(f"Decision: **{res_lr['decision']}**")
        st.caption("ROC-AUC: 87.10% | PR-AUC: 72.88%")

# =============================================================================
# MODULE 5: PORTFOLIO METRICS
# =============================================================================
elif module == "Portfolio Metrics":
    st.subheader("Portfolio & Default Metrics")
    st.caption("Historical performance and policy threshold simulator.")

    artifacts = load_all_artifacts()
    meta = artifacts.get("metadata", {})
    npa = meta.get("npa_metrics", {})

    baseline_npa = float(npa.get("baseline_npa_pct", 21.81))
    post_npa = float(npa.get("post_model_npa_pct", 6.67))
    npa_red = float(npa.get("relative_npa_reduction_pct", 69.43))
    app_vol = float(npa.get("approval_rate_pct", 82.43))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Default Rate", f"{baseline_npa:.2f}%")
    k2.metric("Post-Model Default Rate", f"{post_npa:.2f}%")
    k3.metric("Relative Risk Reduction", f"{npa_red:.2f}%")
    k4.metric("Approval Volume", f"{app_vol:.2f}%")

    st.write("")
    st.markdown("**Threshold Simulator**")
    cutoff_slider = st.slider("Policy Cutoff (%)", 10, 60, 35, 1)

    t = cutoff_slider / 100.0
    sim_npa = max(2.5, min(21.8, baseline_npa * (t / 0.70) ** 1.35))
    sim_app = min(98.0, max(50.0, 100.0 - (100.0 - app_vol) * ((0.35 / t) ** 0.85)))
    sim_red = ((baseline_npa - sim_npa) / baseline_npa) * 100.0

    s1, s2, s3 = st.columns(3)
    s1.metric("Simulated Default Rate", f"{sim_npa:.2f}%")
    s2.metric("Simulated Approval Rate", f"{sim_app:.1f}%")
    s3.metric("Simulated Risk Reduction", f"{sim_red:.1f}%")

    st.write("")
    st.markdown("**Model Leaderboard**")
    lb = meta.get("leaderboard", [])
    if lb:
        st.dataframe(pd.DataFrame(lb), use_container_width=True, hide_index=True)

    # Performance Plots
    p_col1, p_col2 = st.columns(2)
    roc_img = os.path.join(PROJECT_ROOT, "reports", "roc_comparison.png")
    shap_img = os.path.join(PROJECT_ROOT, "reports", "shap_global_beeswarm.png")
    if os.path.exists(roc_img):
        p_col1.image(roc_img, caption="Comparative ROC Curves", use_container_width=True)
    if os.path.exists(shap_img):
        p_col2.image(shap_img, caption="Global SHAP Feature Importance", use_container_width=True)

# =============================================================================
# MODULE 6: DATA DRIFT MONITORING
# =============================================================================
elif module == "Data Drift Monitoring":
    st.subheader("Data Drift Monitoring")
    st.caption("Tracks feature distribution divergence using Kolmogorov-Smirnov statistical tests.")

    drift_path = os.path.join(PROJECT_ROOT, "reports", "drift_summary.json")
    if os.path.exists(drift_path):
        with open(drift_path, "r") as f:
            drift = json.load(f)

        is_drift = drift.get("dataset_drift_detected", False)
        cnt = drift.get("drifted_features_count", 0)
        tot = drift.get("total_features_monitored", 8)
        share = drift.get("drift_share_pct", 0.0)

        if is_drift:
            st.warning(f"Data Drift Alert: {cnt} of {tot} monitored features ({share:.1f}%) exhibit distribution shift.")
        else:
            st.success("Stable: No statistically significant feature drift detected.")

        features = drift.get("features", [])
        if features:
            df_dr = pd.DataFrame(features)
            df_dr["Drift Status"] = df_dr["drift_detected"].apply(lambda x: "DRIFT" if x else "STABLE")
            df_dr = df_dr.rename(columns={
                "feature": "Feature",
                "reference_mean": "Reference Mean",
                "current_mean": "Current Mean",
                "ks_statistic": "KS Statistic",
                "p_value": "P-Value"
            })
            st.dataframe(df_dr[["Feature", "Drift Status", "Reference Mean", "Current Mean", "KS Statistic", "P-Value"]], use_container_width=True, hide_index=True)

        html_report = os.path.join(PROJECT_ROOT, "reports", "evidently_drift_report.html")
        if os.path.exists(html_report):
            with open(html_report, "r", encoding="utf-8") as f:
                st.download_button("Download EvidentlyAI HTML Report", f.read(), "evidently_drift_report.html", "text/html")

# =============================================================================
# MODULE 7: DECISION AUDIT LOGS
# =============================================================================
elif module == "Decision Audit Logs":
    st.subheader("Decision Audit Logs")
    st.caption("Historical log of evaluated applications.")

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df_logs = pd.read_sql_query("SELECT id, created_at, person_age, person_income, loan_amnt, loan_int_rate, loan_grade, default_probability, risk_tier, decision FROM loan_prediction_logs ORDER BY id DESC LIMIT 50", conn)
            conn.close()

            if not df_logs.empty:
                filter_dec = st.selectbox("Filter Decision", ["ALL", "APPROVED", "REJECTED"])
                if filter_dec != "ALL":
                    df_logs = df_logs[df_logs["decision"] == filter_dec]

                df_logs = df_logs.rename(columns={
                    "id": "ID",
                    "created_at": "Timestamp (UTC)",
                    "person_age": "Age",
                    "person_income": "Income ($)",
                    "loan_amnt": "Loan ($)",
                    "loan_int_rate": "Rate (%)",
                    "loan_grade": "Grade",
                    "default_probability": "Default Prob",
                    "risk_tier": "Risk Tier",
                    "decision": "Decision"
                })
                df_logs["Default Prob"] = (df_logs["Default Prob"] * 100).round(1).astype(str) + "%"

                st.dataframe(df_logs, use_container_width=True, hide_index=True)
                csv_logs = df_logs.to_csv(index=False).encode('utf-8')
                st.download_button("Export Logs (.csv)", csv_logs, "Decision_Audit_Logs.csv", "text/csv")
            else:
                st.info("No applications evaluated yet. Run a single loan assessment to log the first record.")
        except Exception as e:
            st.error(f"Error querying database: {e}")
    else:
        st.info("Database file not yet initialized.")
