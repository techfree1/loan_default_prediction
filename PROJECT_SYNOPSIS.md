# PROJECT SYNOPSIS

## 1. Project Header & Identification

- **Project Title:** Bank Muscat Intelligent Retail Credit Risk & MLOps Platform
- **Project Code:** P_137
- **Target Organization / Domain:** Bank Muscat — Retail Banking Division / Credit Underwriting & Risk Intelligence
- **Primary Domain:** Financial Technology (FinTech), Applied Machine Learning, Explainable AI (XAI), MLOps
- **Author / Lead Engineer:** Raghav ([@raghav989](https://github.com/raghav989))
- **Document Version:** 1.0 (Production Release)
- **Date:** October 2026

---

## 2. Executive Summary

Credit risk assessment is central to commercial banking profitability, capital adequacy, and regulatory compliance under Basel II/III and Central Bank of Oman (CBO) directives. Traditional credit scoring systems rely heavily on static scorecard rule sets and linear regressions that fail to capture nonlinear default patterns or adapt dynamically to macroeconomic shifts.

**Project P_137** introduces an end-to-end, production-grade Machine Learning and MLOps platform engineered specifically for retail credit underwriting. The system evaluates loan default probabilities in sub-50 milliseconds, enforces an optimized risk-tiering decision threshold ($0.35$), provides instant mathematical explainability for every underwriting decision via SHAP (SHapley Additive exPlanations), persists full audit logs in PostgreSQL, and continuously detects macroeconomic data drift using Evidently AI and Kolmogorov-Smirnov statistical tests.

On a holdout test portfolio of 6,516 loans, the champion **XGBoost** model demonstrated an **ROC-AUC of 94.83%** and **PR-AUC of 90.26%**, slashing the portfolio Non-Performing Asset (NPA) default rate from a baseline of **21.81%** down to **6.67%** (a **69.43% relative NPA reduction**), successfully intercepting **74.81%** of all potential loan defaults while maintaining a commercially viable **82.43% loan approval rate**.

---

## 3. Problem Statement & Business Objectives

### 3.1 Problem Statement
Retail lending institutions face three critical operational bottlenecks:
1. **High Default and NPA Exposure:** Unscreened retail lending portfolios exhibit default rates exceeding 20%, resulting in heavy loan impairment provisions and balance sheet drag.
2. **Regulatory & "Black Box" Compliance Barrier:** Banking regulators mandate that automated credit rejections provide explicit, adverse-action explanations. Black-box algorithms without explainability cannot be legally deployed in production credit underwriting.
3. **Macroeconomic Vulnerability & Silent Model Degradation:** Inflation spikes, benchmark interest rate hikes, and wage contractions alter borrower behavior post-deployment, causing models trained on historical data to degrade silently without automated drift telemetry.

### 3.2 Target Benchmarks vs. Achieved Results

| Key Metric | Target Requirement | Value Achieved | Status | Business Consequence |
| :--- | :---: | :---: | :---: | :--- |
| **Model Discrimination (ROC-AUC)** | $\ge 85.0\%$ | **94.83%** | **Exceeded (+9.83%)** | Flawless separation between healthy borrowers and defaulters. |
| **Precision-Recall AUC (PR-AUC)** | $\ge 75.0\%$ | **90.26%** | **Exceeded (+15.26%)** | High confidence in high-risk rankings despite class imbalance. |
| **Relative Portfolio NPA Reduction** | $\ge 30.0\%$ | **69.43%** | **Exceeded (+39.43%)** | More than double the initial risk reduction mandate. |
| **Portfolio Post-Model NPA Rate** | $< 10.0\%$ | **6.67%** | **Achieved** | Keeps retail loan book quality well above industry benchmarks. |
| **Default Interception Rate (Recall)** | $\ge 70.0\%$ | **74.81%** | **Achieved** | 3 out of every 4 defaults are successfully preempted. |
| **Underwriting Approval Rate** | $\ge 80.0\%$ | **82.43%** | **Achieved** | Safeguards bank interest margins and retail loan growth. |
| **Audit & Explainability Readiness** | 100% auditable | **100% (SHAP + DB)** | **Achieved** | Full compliance with regulatory Fair Lending & CBO audits. |

---

## 4. System Architecture & Component Design

The platform adopts a decoupled, microservices-ready architecture:

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    Client Touchpoints                  │
                               │  - Underwriter Streamlit Dashboard (Port 8501)        │
                               │  - Modern Web Frontend (Tailwind/HTML5) (Port 8000)    │
                               │  - Upstream Core Banking APIs (REST JSON)              │
                               └───────────────────────────┬────────────────────────────┘
                                                           │ HTTP POST /score
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                  FastAPI Serving Engine                │
                               │  - Input Schema Validation (Pydantic)                  │
                               │  - Real-time Feature Engineering                       │
                               │  - Fitted ColumnTransformer Pipeline                   │
                               │  - XGBoost Classifier Inference (Threshold = 0.35)      │
                               │  - SHAP TreeExplainer Attributions                     │
                               └───────────────┬────────────────────────┬───────────────┘
                                               │                        │
                        Prediction Log Payload │                        │ JSON Response
                                               ▼                        ▼
               ┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────┐
               │         Audit & Persistence Store               │   │ Client Response                     │
               │ - PostgreSQL (loan_prediction_logs)             │   │ - Probability of Default (PD)       │
               │ - Fallback: SQLite Local DB                     │   │ - Risk Tier (Low / Medium / High)   │
               │ - Stores: Inputs, Prob, Decision, Top SHAP      │   │ - Decision: APPROVED / REJECTED     │
               └─────────────────────────────────────────────────┘   │ - Top 4 Adverse / Positive Factors  │
                                                                     └─────────────────────────────────────┘
```

### Core Subsystems:
1. **ML Training & Retraining Engine (`run_pipeline.py`, `src/`):**
   - Implements automated end-to-end data processing, feature scaling, model training, MLflow experiment logging, champion selection, and drift scenario simulation.
2. **Serving Backend (`app/main.py`):**
   - High-throughput asynchronous FastAPI server exposing `/score`, `/health`, `/metrics/npa`, `/monitoring/drift`, and `/audit-logs`.
3. **Regulatory Explainability Module (`shap_explainer.pkl`, `reports/`):**
   - Utilizes TreeSHAP to calculate exact additive feature importances per applicant, extracting the primary four risk drivers for immediate loan officer review.
4. **Interactive Underwriting & Risk Portal (`app/streamlit_app.py`):**
   - Three-tab executive dashboard providing single-loan scoring, portfolio sensitivity simulation, and macroeconomic drift monitoring.
5. **Continuous Telemetry & Drift Monitoring (`src/drift_monitoring.py`):**
   - Performs two-sample Kolmogorov-Smirnov (KS) hypothesis tests and generates interactive Evidently AI HTML reports comparing reference distributions against production inputs.

---

## 5. Dataset & Feature Engineering

### 5.1 Dataset Demographics
- **Source Dataset:** Historical retail loan records (`data/raw_bank_muscat_loans.csv`) comprising **32,582 records**.
- **Data Partitions:**
  - **Reference Training Set:** 26,059 applications (80% stratified split).
  - **Holdout Test Set:** 6,516 applications (20% stratified split).
  - **Production Drift Stress Set:** 6,516 synthetically shocked applications.
- **Target Variable:** `loan_status` (0 = Fully Paid / Non-Default, 1 = Default / 90+ Days Past Due).
- **Baseline Default Rate:** **21.81%** (class imbalance: ~78.2% non-default, ~21.8% default).

### 5.2 Banking Ratio Feature Engineering
To maximize model discrimination beyond raw demographic fields, five domain-specific financial ratios were engineered:

$$\text{Loan-to-Income Ratio} = \frac{\text{loan\_amnt}}{\text{person\_income} + 1.0}$$

$$\text{Annual Interest Burden Ratio} = \frac{\text{loan\_amnt} \times \left(\frac{\text{loan\_int\_rate}}{100}\right)}{\text{person\_income} + 1.0}$$

$$\text{Income Stability Index} = \frac{\text{person\_income}}{\text{person\_emp\_length} + 1.0}$$

$$\text{Credit Bureau Maturity Ratio} = \frac{\text{cb\_person\_cred\_hist\_length}}{\max(\text{person\_age} - 18.0, 1.0)}$$

$$\text{Loan Percent Income} = \frac{\text{loan\_amnt}}{\text{person\_income} + 1.0}$$

### 5.3 Preprocessing & Column Transformation
- **Numerical Pipeline:** Handled via `SimpleImputer(strategy='median')` followed by `StandardScaler()`. Outliers in age ($>90$ years) and employment length ($>60$ years) are sanitized.
- **Categorical Pipeline:** Handled via `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- All transformations are bundled into a serialized `ColumnTransformer` object (`models/preprocessor.pkl`) guaranteeing parity between training and inference.

---

## 6. Model Training, Selection & Evaluation

Three candidate algorithms were evaluated across identical cross-validation splits and logged to an SQLite-backed **MLflow** registry:

### 6.1 Multi-Model Performance Leaderboard

| Model Candidate | Hyperparameter Highlights | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Post-Model NPA | NPA Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $C=1.0$, `class_weight='balanced'`, L-BFGS | 87.10% | 72.88% | 0.576 | 0.437 | **84.38%** | 5.89% | 73.00% |
| **Random Forest** | $N=300$, depth $=14$, `balanced_subsample` | 92.56% | 87.17% | 0.748 | 0.714 | 78.61% | 6.14% | 71.84% |
| **XGBoost (Champion)** | $N=400$, depth $=6$, $\eta=0.05$, sub $=0.85$ | **94.83%** | **90.26%** | **0.829** | **0.928** | 74.81% | **6.67%** | **69.43%** |

### 6.2 Decision Threshold Optimization Rationale
In retail banking, false positives (approving a defaulter) incur significantly higher financial losses (full loss-given-default) than false negatives (rejecting an applicant who would have repaid).
- The conventional $0.50$ probability threshold lets high-risk marginal applicants pass through.
- Logistic Regression achieved high recall ($84.38\%$) but low precision ($43.7\%$), meaning it falsely flagged over $56\%$ of creditworthy borrowers as defaulters, severely harming revenue.
- **XGBoost tuned at threshold 0.35** provided the optimal utility frontier:
  - **Precision:** 92.8% (approved borrowers are overwhelmingly reliable).
  - **Recall:** 74.81% (neutralizes ~3 of every 4 defaults).
  - **Approval Rate:** 82.43% (retains standard retail banking loan velocity).

---

## 7. Explainable AI (XAI) & Regulatory Compliance

To adhere to Central Bank guidelines and fair lending mandates, the platform incorporates **TreeSHAP**:

1. **Global Portfolio Risk Attribution:**
   - Saved in `reports/shap_global_beeswarm.png`.
   - Top portfolio predictors identified:
     1. `loan_to_income_ratio` (strongest positive driver of default).
     2. `loan_int_rate` (higher interest burden exponentially elevates default propensity).
     3. `interest_burden_ratio` (composite pressure on liquid borrower income).
     4. `person_income` (protective factor mitigating default risk).
2. **Local Adverse-Action Attribution:**
   - Every inference response from `POST /score` dynamically computes SHAP values for the applicant and returns the top 4 contributing factors.
   - Example response explanation:
     - $+0.312$ impact from `loan_to_income_ratio`
     - $+0.186$ impact from `loan_int_rate`
     - $-0.095$ mitigating impact from stable `person_emp_length`
3. **Audit Trail Persistence:**
   - Every API request, engineered vector, model prediction, risk tier, and SHAP explanation is recorded in the `loan_prediction_logs` table (PostgreSQL/SQLite) with UTC timestamps.

---

## 8. Macroeconomic Drift Simulation & MLOps Governance

To address model degradation during economic cycles (e.g., inflation, regional recessions, monetary tightening), the system integrates automated drift detection:

### 8.1 Macroeconomic Stress Scenario
A synthetic production stress batch (`data/production_drifted_batch.csv`) simulates:
- **Interest Rate Spike:** $+2.80\%$ shift ($\sigma = 0.8\%$).
- **Borrower Income Contraction:** $-15.0\%$ uniform shock.
- **Credit Demand Surge:** $+15.0\%$ higher requested loan amounts.

### 8.2 Drift Detection Protocol
- **Kolmogorov-Smirnov (KS) Two-Sample Test:** Tests cumulative distribution functions between training baseline and production batches.
- **Significance Level:** $\alpha = 0.05$.
- **Automated Alerting:** When 5 of 8 key monitored features show $p < 0.05$, the system logs a `⚠️ DRIFT ALERT: Retraining Recommended` warning and serves the comprehensive Evidently AI HTML report at `/reports/evidently_drift_report.html`.

---

## 9. Technology Stack & Deployment Specifications

### 9.1 Software & Frameworks

| Tier | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Core runtime environment |
| **Machine Learning** | XGBoost, Scikit-Learn, NumPy, Pandas | Model training, preprocessing, ratio engineering |
| **Explainable AI** | SHAP (TreeSHAP) | Game-theoretic local and global feature attributions |
| **MLOps & Monitoring** | MLflow, Evidently AI, SciPy | Experiment tracking, KS-test statistical drift evaluation |
| **API Backend** | FastAPI, Uvicorn, Pydantic | High-performance asynchronous REST service |
| **Database ORM** | SQLAlchemy, PostgreSQL 15, SQLite3 | Persistent underwriting audit logs & fallback store |
| **Frontend UI** | Streamlit, Tailwind CSS, HTML5, Chart.js | Executive dashboards & responsive applicant scoring portals |
| **Containerization** | Docker, Docker Compose | Multi-container production deployment |

### 9.2 Container & Deployment Options
1. **Zero-Setup Cloud Pull:**
   ```bash
   docker run -d -p 8501:8501 -p 8000:8000 raghav989/loan-default-prediction:latest
   ```
2. **Production Multi-Container Stack (PostgreSQL + API + Web):**
   ```bash
   docker compose up --build
   ```
3. **Single CLI Retraining Trigger:**
   ```bash
   python run_pipeline.py
   ```

---

## 10. Conclusion & Future Roadmap

**Project P_137** successfully transitions Bank Muscat's retail credit risk assessment from traditional heuristic scorecards to a fully automated, explainable, and production-hardened AI ecosystem. By coupling state-of-the-art predictive accuracy (**94.83% ROC-AUC**) with a cost-optimal **0.35** decision threshold, the platform achieves a **69.43% relative reduction in portfolio NPAs** while keeping retail loan flow uninterrupted (**82.43% approval rate**).

### Future Roadmap:
1. **Automated Continuous Training (CT) Pipeline:** Integrating Airflow or GitHub Actions to automatically trigger pipeline execution when drift is flagged by Evidently AI.
2. **Alternative Credit Data Integration:** Incorporating telecom utility bills, salary direct deposits, and rent payment history for thin-file credit applicants.
3. **Federated Learning & Real-time Bureau Webhooks:** Direct integrations with the Central Bank of Oman (CBO) credit bureau portal via secure mTLS REST webhooks.
