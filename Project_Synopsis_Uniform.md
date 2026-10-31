---
title: "PROJECT SYNOPSIS"
subtitle: "Loan Default Prediction Using Explainable Machine Learning"
author: "Project Code: P_137 | Domain: Data Analytics / Machine Learning / MLOps"
date: "October 2026"
---

# 1. Project at a Glance

| Item | Summary |
| :--- | :--- |
| **What it does** | Predicts the probability that a borrower will fail to repay a loan, and explains the reasons behind each individual prediction. |
| **Who uses it** | Loan officers and risk analysts at lending institutions (as a quantitative decision-support tool, preserving human-in-the-loop oversight). |
| **Input Data** | Borrower profile and loan details: annual income, loan amount, interest rate, employment length, home ownership, credit history, and past defaults. |
| **Output** | Default probability (0% to 100%), risk category (Low / Medium / High), underwriting decision (Approved / Rejected), and key SHAP risk drivers. |
| **Core Technologies** | Python, Scikit-Learn, XGBoost, SHAP, FastAPI, Streamlit, PostgreSQL/SQLite, MLflow, Evidently AI, Docker. |
| **Target Performance** | ROC-AUC $\ge 0.85$ (Achieved: **94.83%** with XGBoost Champion on holdout test partition). |

---

# 2. Problem Statement

When financial institutions lend money, borrower defaults result in Non-Performing Assets (NPAs), leading to direct capital loss and destabilised balance sheets. Conventional credit underwriting heavily relies on static heuristic scorecards and manual assessments, which fail to capture non-linear, multi-variable interactions present in historical credit data.

Existing machine-learning solutions often exhibit two significant operational shortcomings:
1. **The "Black-Box" Opacity Problem:** Models generate a risk score without explainability, preventing loan underwriters from legally justifying adverse decisions to customers and regulatory auditors.
2. **Post-Deployment Model Degradation:** Models are rarely monitored in real time after deployment; over time, macroeconomic shifts and evolving consumer behaviour cause undetected data drift and silent accuracy loss.

This project resolves both challenges by developing an end-to-end intelligent platform that predicts default risk with high precision, explains every prediction transparently using SHAP, continuously audits data drift via Evidently AI, and serves inferences via a production-grade FastAPI microservice and Streamlit portal.

---

# 3. Motivation

- **Business Capital Preservation:** Early identification of high-risk applicants directly reduces default rates and enables risk-based pricing and prudent loan structuring.
- **Regulatory Compliance & Transparency:** Adverse-action notices and fair lending regulations require institutional transparency. SHAP provides individual feature attributions for every decision.
- **Continuous Model Governance:** Macroeconomic volatility requires real-time drift detection to trigger automated recalibration before models degrade.
- **Enterprise Engineering Lifecycle:** Unlike typical academic experiments that terminate at offline model training, this project implements the full production MLOps lifecycle: data ingestion, feature engineering, model benchmarking, explainability, API serving, persistent audit logging, and Docker containerization.

---

# 4. Aim and Objectives

**Aim:** To engineer and deploy a containerized end-to-end Machine Learning system that predicts loan default probability, explains individual predictions via SHAP, monitors macroeconomic data drift, and provides a real-time serving API with an interactive dashboard.

### Specific Objectives:

| Area | Objective |
| :--- | :--- |
| **Data Engineering** | Acquire, clean, and impute historical credit data; treat class imbalance; and engineer key financial ratios (Loan-to-Income, Debt Burden Ratio, Credit Maturity). |
| **Model Selection** | Train, optimize, and benchmark Logistic Regression (baseline), Random Forest, and XGBoost; evaluate via ROC-AUC, PR-AUC, F1-Score, and NPA reduction. |
| **Explainability (XAI)** | Implement TreeSHAP to generate global feature importance rankings and local per-borrower risk attribution waterfalls. |
| **Microservice Serving** | Build an asynchronous FastAPI REST microservice (`POST /score`) returning default probability, risk tier, and top contributing factors. |
| **Database Audit Trail** | Log all incoming applications and predictions to a persistent PostgreSQL / SQLite database for compliance auditing. |
| **Experiment Tracking** | Record hyperparameters, evaluation metrics, and model artifacts using MLflow for full reproducibility. |
| **Drift Monitoring** | Perform two-sample Kolmogorov-Smirnov statistical tests using Evidently AI to detect distribution drift between training baselines and live batches. |
| **Production Deployment** | Package the entire service stack into an immutable Docker image hosted on Docker Hub (`raghav989/loan-default-prediction:latest`). |

---

# 5. Key Terms Explained

| Term | Technical Meaning & Project Context |
| :--- | :--- |
| **Loan Default** | Failure of a borrower to meet legal repayment obligations (binary target: 1 = defaulted, 0 = repaid). |
| **NPA (Non-Performing Asset)** | A loan on which repayments have ceased, causing write-offs and balance sheet distress. |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve (measures discrimination across all decision thresholds; 1.0 represents perfect separation). |
| **PR-AUC** | Precision-Recall AUC (critical for heavily imbalanced credit datasets where defaults represent a minority class). |
| **Class Imbalance** | Disproportionate target distribution where defaulters are far fewer than non-defaulters; treated using stratified sampling and threshold tuning. |
| **SHAP** | Shapley Additive exPlanations; game-theoretic approach allocating each feature's marginal contribution toward a prediction score. |
| **Data Drift** | Statistical divergence between production input feature distributions and training baselines, evaluated via Kolmogorov-Smirnov test ($p < 0.05$). |
| **Docker Container** | A lightweight, standalone package containing application code, runtime libraries, and environment dependencies for consistent execution. |

---

# 6. Proposed System Architecture & Workflow

The platform operates across two synchronized phases: an **Offline Training & Governance Pipeline** and an **Online Real-Time Inference Microservice**.

| # | Pipeline Stage | Description of Operation | Technology Used |
| :---: | :--- | :--- | :--- |
| **1** | **Data Ingestion** | Ingest historical credit records with ground-truth default labels. | Pandas |
| **2** | **Preprocessing & Cleaning** | Impute missing values, remove age/employment anomalies, and stratify splits. | Scikit-Learn |
| **3** | **Feature Engineering** | Compute banking ratios: Loan-to-Income (LTI), Interest Burden, and Maturity Ratio. | NumPy, Scikit-Learn |
| **4** | **Model Training** | Train Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning. | Scikit-Learn, XGBoost |
| **5** | **Evaluation & Benchmarking** | Measure ROC-AUC, PR-AUC, F1-score, Precision, Recall, and portfolio default reduction. | Scikit-Learn |
| **6** | **Model Registry** | Log experiment runs, parameters, metrics, and serialized artifacts (`.pkl`). | MLflow |
| **7** | **Explainability Generation** | Compute TreeSHAP values for global feature rankings and applicant-level waterfalls. | SHAP |
| **8** | **REST Microservice** | Serve high-throughput asynchronous scoring endpoint (`POST /score`). | FastAPI, Uvicorn |
| **9** | **Audit Logging** | Persist application features, probabilities, and SHAP drivers for audit compliance. | SQLAlchemy, SQLite/PostgreSQL |
| **10** | **Drift Monitoring** | Execute two-sample Kolmogorov-Smirnov drift tests and generate interactive audit reports. | Evidently AI |
| **11** | **Container Deployment** | Multi-service Docker container deployment with unified startup orchestration. | Docker, Docker Hub |

### Practical Scenario (System in Action):
1. A loan applicant applies for a $\$15,000$ personal loan with an annual income of $\$55,000$.
2. The underwriting application sends the payload to the FastAPI scoring microservice.
3. The pipeline engineers underwriting ratios and passes the transformed vector to the champion XGBoost model.
4. The system returns:
   - **Default Probability:** $14.2\%$ (Risk Tier: Low Risk)
   - **Decision:** APPROVED (Threshold Cutoff: $\le 35.0\%$)
   - **Key Mitigating Factors:** High employment stability (+9 years), clean credit bureau record.
   - **Key Risk Drivers:** Loan amount to income ratio ($27.2\%$).
5. The decision is permanently logged in the audit database, allowing regulatory inspection at any time.

---

# 7. Dataset and Input Features

A structured credit dataset comprising **32,582 loan application records** is utilized, featuring a diverse cross-section of retail credit profiles.

| Feature Group | Specific Attributes | Relevance to Credit Risk |
| :--- | :--- | :--- |
| **Applicant Profile** | `person_age`, `person_emp_length`, `person_home_ownership` | Reflects demographic stability and tenure. |
| **Financial Capacity** | `person_income`, `loan_to_income_ratio`, `interest_burden_ratio` | Measures debt-service capacity and leverage. |
| **Credit Facility** | `loan_amnt`, `loan_int_rate`, `loan_grade`, `loan_intent` | Defines contractual loan exposure and risk pricing. |
| **Credit Bureau History** | `cb_person_default_on_file`, `cb_person_cred_hist_length` | Captures historical delinquency and credit maturity. |
| **Target Variable** | `loan_status` (Binary: 1 = Defaulted, 0 = Non-Defaulted) | Supervised classification ground truth. |

---

# 8. Methodology

1. **Exploratory Data Analysis (EDA):** Analyze distributions, correlate credit grades with default rates, and isolate anomalous outliers.
2. **Preprocessing Pipeline:** Implement a reproducible `ColumnTransformer` with `SimpleImputer` (median strategy) and `StandardScaler` for continuous features, alongside `OneHotEncoder` for categoricals.
3. **Financial Ratio Engineering:** Synthesize four banking ratios:
   - **Loan-to-Income Ratio (LTI):** $\frac{\text{Loan Amount}}{\text{Annual Income} + 1}$
   - **Interest Burden Ratio:** $\frac{\text{Loan Amount} \times (\text{Interest Rate} / 100)}{\text{Annual Income} + 1}$
   - **Income per Employment Year:** $\frac{\text{Annual Income}}{\text{Employment Length} + 1}$
   - **Credit Bureau Maturity Ratio:** $\frac{\text{Credit History Length}}{\max(\text{Age} - 18, 1)}$
4. **Stratified Data Splitting:** Partition into $80\%$ training and $20\%$ holdout test sets using stratification to maintain true default class ratios without data leakage.
5. **Model Training & Tuning:** Train Logistic Regression (L2 regularized baseline), Random Forest (100 estimators), and XGBoost (gradient boosting trees).
6. **Threshold Optimization:** Calibrate decision thresholds ($35\%$ optimal cutoff) to maximize default capture while maintaining retail approval volumes.
7. **Explainability Extraction:** Compute local Shapley values to identify top positive and negative risk contributors.
8. **Service Engineering & Containerization:** Develop REST endpoints, Streamlit workstations, persistent SQLite/PostgreSQL logging, and build multi-arch Docker images.

---

# 9. Technology Stack

| Technology | Category | Role in the Project |
| :--- | :--- | :--- |
| **Python 3.11** | Core Language | Core development environment and data science runtime. |
| **Pandas / NumPy** | Data Processing | Dataframe manipulation, matrix math, and ratio calculations. |
| **Scikit-Learn** | Machine Learning | Preprocessing pipelines, metrics evaluation, baseline models. |
| **XGBoost** | Machine Learning | High-performance gradient boosted decision tree classifier (Champion). |
| **SHAP** | Explainable AI | Feature attribution analysis and individual borrower waterfalls. |
| **FastAPI / Uvicorn** | Backend API | High-throughput asynchronous REST prediction service. |
| **Streamlit** | User Interface | Minimalist, clean-white interactive underwriting dashboard. |
| **SQLite / PostgreSQL** | Storage & Audit | Persistent relational logging of prediction transactions and SHAP drivers. |
| **MLflow** | MLOps Tracking | Experiment tracking, hyperparameter logging, and model artifact registry. |
| **Evidently AI** | Drift Monitoring | Two-sample Kolmogorov-Smirnov data drift tests and reporting. |
| **Docker & Docker Hub** | Containerization | Container packaging and public registry deployment (`raghav989/loan-default-prediction`). |

---

# 10. Evaluation and Success Criteria

| Evaluation Criterion | Evaluation Methodology | Benchmark / Target Achieved |
| :--- | :--- | :--- |
| **Predictive Discrimination** | ROC-AUC score on holdout test partition | **94.83%** (Exceeds target $\ge 85.0\%$) |
| **Precision-Recall Performance** | PR-AUC score for minority default class | **90.26%** |
| **Classification Quality** | F1-Score, Precision, and Recall | Precision: $0.928$ \| Recall: $0.748$ \| F1: $0.829$ |
| **Portfolio NPA Reduction** | Comparison of pre-model vs post-model default rate | Baseline NPA: $21.81\%$ $\rightarrow$ Post-Model NPA: $6.67\%$ (**$69.43\%$ Relative Reduction**) |
| **Explainability Validity** | Consistency of top-4 SHAP attributions per application | Successfully returns dynamic positive and negative risk contributions |
| **System Latency** | FastAPI inference turnaround time | $< 25\text{ ms}$ per loan application |
| **Container Portability** | Docker execution on any host environment | Fully verified via `docker run` on macOS, Linux, and Windows |

---

# 11. Expected Deliverables

- Fully trained, validated, and serialized machine learning estimators (`best_model.pkl`, `preprocessor.pkl`, `shap_explainer.pkl`).
- Comprehensive model tournament performance report and leaderboard.
- FastAPI REST microservice supporting `/score`, `/health`, and `/monitoring/drift` endpoints.
- Clean white Streamlit underwriting workstation with single scoring, what-if lab, batch CSV processing, and audit logs.
- Automated Evidently AI data drift reports (`reports/evidently_drift_report.html`).
- Relational audit schema storing historical loan evaluations.
- Production-ready Docker container published on Docker Hub (`raghav989/loan-default-prediction:latest`).
- Comprehensive documentation, source code repository, and project synopsis.

---

# 12. Scope, Limitations, and Responsible Use

### In-Scope:
- Full-lifecycle automated credit default scoring and explainability on structured tabular datasets.
- Interactive multi-module dashboard supporting underwriter decisioning, what-if sensitivity analysis, and batch scoring.
- Automated drift detection comparing inference batches against training baselines.
- Immutable Docker containerization enabling zero-dependency deployment.

### Out-of-Scope:
- Direct real-time clearinghouse API settlement with live central bank banking networks.
- Fully automated unattended loan disbursement (maintains human-in-the-loop oversight).

### Ethical & Responsible AI Considerations:
- **Fair Lending:** Demographic attributes (such as age) must be evaluated in compliance with non-discrimination guidelines; models should not rely on protected attributes unfairly.
- **Explainable Decisions:** Every rejected application provides verifiable, data-backed reasons via SHAP to eliminate algorithmic bias.
- **Human Oversight:** The system acts as an intelligent decision-support assistant; final approval authority remains with authorized underwriting officers.

---

# 13. Proposed Project Timeline (12-Week Schedule)

| Phase | Weeks | Core Milestones & Tasks | Deliverable Outputs |
| :---: | :---: | :--- | :--- |
| **1** | **Weeks 1 - 2** | Domain research, dataset acquisition, EDA, environment configuration. | Raw dataset, exploratory charts |
| **2** | **Weeks 3 - 4** | Data cleaning, imputation, and banking ratio feature engineering. | Preprocessing pipeline (`.pkl`) |
| **3** | **Weeks 5 - 6** | Model training (Logistic Regression, Random Forest, XGBoost) and tuning. | Model tournament benchmarks |
| **4** | **Week 7** | TreeSHAP explainer fitting, global importance, and local waterfall graphs. | Fitted explainer artifacts |
| **5** | **Weeks 8 - 9** | FastAPI REST endpoints, database schema design, and audit logging. | Working backend microservice |
| **6** | **Week 10** | Evidently AI macroeconomic drift monitoring and report generation. | Drift report (`drift_summary.json`) |
| **7** | **Week 11** | Streamlit workstation development, Docker build, and Docker Hub push. | Docker image (`raghav989/...`) |
| **8** | **Week 12** | End-to-end integration testing, performance validation, and final report. | Complete project synopsis & code |

---

# 14. Future Enhancements

- **Automated Retraining Pipelines:** Event-driven retraining workflows automatically triggered when Evidently AI detects statistical drift beyond defined alpha thresholds.
- **Cost-Sensitive Loss Functions:** Custom asymmetric loss weighting accounting for the fact that false negatives (missed defaulters) are economically costlier than false positives.
- **Open Banking API Connectors:** Automated ingestion and OCR extraction from live borrower bank statements and tax filings.
- **Alternative Credit Scoring:** Inclusion of alternative payment histories (utility bills, rental payments) to responsibly evaluate unbanked borrowers.

---

# 15. Conclusion

This project delivers a comprehensive, enterprise-ready credit underwriting platform that transcends simple model training. By uniting high predictive precision (**94.83% ROC-AUC** with XGBoost) with **transparent SHAP factor explainability**, **continuous data drift monitoring**, and **Docker containerization**, the platform demonstrates how modern machine learning and MLOps practices can transform institutional credit risk management into an auditable, transparent, and resilient operational workflow.

---

# 16. Keywords

Loan Default Prediction, Credit Risk Underwriting, Explainable AI (XAI), SHAP, XGBoost, Random Forest, Logistic Regression, Data Drift, Evidently AI, FastAPI, Streamlit, Docker Containerization, MLOps, Non-Performing Assets (NPAs).
