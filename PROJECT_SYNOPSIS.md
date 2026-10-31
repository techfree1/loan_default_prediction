---
title: "PROJECT SYNOPSIS"
subtitle: "AI-Driven Retail Loan Default Prediction, SHAP Explainability & Containerized MLOps Platform"
author: "Raghav"
date: "October 2026"
geometry: "margin=1in"
output: docx_document
---

# 1. Project Title
**Intelligent Retail Loan Default Prediction and Risk Analytics Platform using Machine Learning, Explainable AI (XAI), and Docker Containerization**

---

# 2. Executive Summary / Abstract
Credit risk assessment is one of the most critical challenges facing the global retail banking and financial technology (FinTech) industry. Traditional credit scorecard models rely predominantly on manual rule-based heuristics and static FICO/bureau scores, which fail to capture complex non-linear borrower risk patterns and macroeconomic shifts. 

This project presents an end-to-end, production-grade Machine Learning and MLOps system that automates retail loan default underwriting. Benchmarking three distinct algorithm families—**Logistic Regression (L2 Baseline)**, **Random Forest**, and an optimized **XGBoost (Champion)**—the champion model achieves a superior **ROC-AUC of 94.83%** and **PR-AUC of 90.26%**, reducing portfolio default rates from **21.81%** down to **6.67%** (a **69.43% relative reduction in Non-Performing Assets**). 

To ensure institutional compliance and regulatory transparency, individual risk decisions are explained in real time via **SHAP (Shapley Additive exPlanations)**. The entire system is engineered with automated Kolmogorov-Smirnov data drift monitoring (**EvidentlyAI**), a **FastAPI** REST microservice, a clean **Streamlit** user workstation, and is fully packaged and deployed as an industry-standard **Docker Container** published on Docker Hub (`raghav989/loan-default-prediction:latest`).

---

# 3. Problem Statement & Motivation
Financial institutions lose billions annually to credit defaults and Non-Performing Assets (NPAs). Key limitations in conventional credit evaluation include:
1. **Inability to Model Non-Linear Interactions:** Static thresholds overlook interdependent financial factors such as debt-to-income spikes combined with short credit history.
2. **Lack of Explainability in Black-Box Models:** Regulatory standards mandate that lenders provide clear, auditable adverse-action reasons whenever an application is declined.
3. **Environment Discrepancies ("Works on My Machine"):** Data science models often struggle in transition from local research notebooks to scalable production environments.
4. **Data Drift & Concept Drift Vulnerability:** Shifting macroeconomic conditions degrade model accuracy over time if distribution changes are unmonitored.

---

# 4. Objectives of the Project
- **High-Precision Risk Discrimination:** Train and optimize tree-based ensemble estimators achieving $\ge 90\%$ ROC-AUC on holdout test partitions.
- **Measurable NPA Reduction:** Establish optimal policy rejection cutoffs ($35\%$ threshold) slashing portfolio defaults by $\ge 30\%$ while preserving healthy loan approval volumes ($\ge 80\%$).
- **Explainable AI (XAI):** Integrate tree SHAP explainers to surface the exact quantitative risk drivers behind every single loan decision.
- **Continuous Model Governance:** Implement automated statistical drift monitoring comparing production inference batches with reference baselines.
- **Enterprise Containerization:** Package the backend engine, model artifacts, and user interface into an immutable Docker image hosted on Docker Hub for zero-configuration, cross-platform execution.

---

# 5. System Architecture & Methodology

```
+-----------------------------------------------------------------------------------+
|                           END-TO-END SYSTEM PIPELINE                              |
+-----------------------------------------------------------------------------------+
                                          |
  [1] DATASET (32,582 Records)             v
  Raw Historical Loan Records ---> Data Cleaning & Outlier Filtering
                                          |
  [2] FEATURE ENGINEERING                 v
  Engineered Banking Ratios   ---> Loan-to-Income (LTI) | Debt-to-Income (DTI)
                              ---> Interest Burden Ratio | Credit Maturity Index
                                          |
  [3] MODEL SELECTION & TRAINING          v
  ColumnTransformer Pipeline  ---> Logistic Regression (87.10% AUC)
                              ---> Random Forest (92.56% AUC)
                              ---> XGBoost Champion (94.83% AUC)
                                          |
  [4] MODEL GOVERNANCE & XAI              v
  SHAP Attribution            ---> Local Waterfall / Bee-swarm Driver Plots
  EvidentlyAI Monitoring      ---> Two-Sample Kolmogorov-Smirnov Drift Tests
                                          |
  [5] CONTAINERIZED SERVING LAYER         v
  FastAPI Microservice (8000) ---> Real-time REST Scoring Endpoint (/score)
  Streamlit Portal (8501)     ---> Underwriting Dashboard, What-If Lab, Audit Log
  SQLite / PostgreSQL Engine  ---> Persistent Prediction Logging
                                          |
  [6] PRODUCTION DEPLOYMENT               v
  Docker Hub Public Registry  ---> raghav989/loan-default-prediction:latest
+-----------------------------------------------------------------------------------+
```

---

# 6. Machine Learning Benchmarks & Results

| Model Architecture | ROC-AUC (%) | PR-AUC (%) | F1-Score | Precision | Recall | Post-Model NPA (%) | Relative NPA Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 87.10% | 72.88% | 0.576 | 0.437 | 0.844 | 5.89% | 73.00% |
| **Random Forest Classifier** | 92.56% | 87.17% | 0.748 | 0.714 | 0.786 | 6.14% | 71.84% |
| **XGBoost Classifier (Champion)** | **94.83%** | **90.26%** | **0.829** | **0.928** | **0.748** | **6.67%** | **69.43%** |

### Key Business Metrics Achieved:
- **Baseline Historical NPA:** $21.81\%$
- **Post-Model Champion NPA:** $6.67\%$ (Approved portfolio default rate)
- **Target NPA Reduction Met:** $+69.43\%$ (exceeding strategic targets by $+39.43\%$)
- **Defaults Prevented:** $74.81\%$ (3 out of every 4 historical defaults blocked)
- **Preserved Loan Approval Velocity:** $82.43\%$

---

# 7. Docker Containerization & Deployment Architecture

### 7.1 Container Design
The platform is packaged into an optimized, secure Docker image based on `python:3.11-slim`:
- **Unified Multi-Service Entrypoint:** A shell entrypoint script (`deploy/entrypoint.sh`) orchestrates the simultaneous background execution of the FastAPI REST API (port `8000`) and Streamlit web application (port `8501`).
- **Cloud Architecture Support:** Built with `linux/amd64` architecture, ensuring native compatibility across cloud platforms (AWS, Render, Google Cloud, DigitalOcean) and Apple Silicon emulation.
- **Docker Hub Public Registry:** Published publicly for worldwide one-click deployment:
  $$\text{Image Repository: } \mathbf{raghav989/loan-default-prediction:latest}$$

### 7.2 Zero-Configuration Execution Command
Any evaluator or server administrator can deploy the entire platform instantly without installing Python, dependencies, or compilers:

```bash
docker run -d -p 8501:8501 -p 8000:8000 raghav989/loan-default-prediction:latest
```

- **Web Application Portal:** `http://localhost:8501`
- **Interactive REST Swagger Documentation:** `http://localhost:8000/docs`

---

# 8. Interactive Features of the Application

1. **Single Loan Underwriting Workstation:** Form for real-time applicant scoring with one-click quick-load borrower personas (Prime, Moderate, Subprime, Young Graduate).
2. **Real-time Financial Heads-Up Display (HUD):** Instant calculation of Loan-to-Income (LTI), Debt-to-Income (DTI), and Estimated Monthly Installments (EMI).
3. **SHAP Factor Attribution:** Dynamic risk-increasing and risk-mitigating visual bar charts explaining model predictions.
4. **What-If Risk Sensitivity Lab:** Real-time sliders allowing underwriters to adjust loan parameters and observe live threshold changes.
5. **Batch Portfolio Scorer:** CSV upload engine with real-time scoring progress and downloadable batch decision reports.
6. **Model Arena:** Side-by-side consensus comparison across all three trained estimators (XGBoost vs Random Forest vs Logistic Regression).
7. **Continuous Data Drift Monitor:** Kolmogorov-Smirnov statistical tests tracking feature distribution drift with downloadable EvidentlyAI HTML audit reports.
8. **Live Persistent Audit Trail:** Database-backed historical decision log with CSV export capability.

---

# 9. Hardware & Software Requirements

### Software Stack:
- **Operating System:** Linux / macOS / Windows (Dockerized)
- **Programming Language:** Python 3.11
- **Machine Learning & Modeling:** Scikit-Learn 1.5.2, XGBoost 2.1+, SHAP 0.46+
- **Backend Framework:** FastAPI, Uvicorn, Pydantic
- **Frontend & UI:** Streamlit
- **Monitoring & MLOps:** EvidentlyAI, MLflow
- **Database & ORM:** SQLite / PostgreSQL, SQLAlchemy
- **Containerization:** Docker Engine, Docker Hub

### Minimum Hardware Specifications:
- **RAM:** 2 GB minimum (4 GB recommended)
- **CPU:** Dual-core 2.0 GHz or higher (x86_64 / ARM64)
- **Disk Storage:** 3 GB free space

---

# 10. Conclusion & Future Enhancements
The AI-Driven Loan Default Prediction Platform bridges the gap between sophisticated machine learning research and enterprise lending operations. By coupling top-tier predictive accuracy (**94.83% ROC-AUC**) with **regulatory SHAP explainability**, **data drift governance**, and **Docker containerization**, the platform delivers an audit-ready, production-grade financial risk solution.

**Future Scope:**
- Integration with live open banking APIs for automated bank statement parsing.
- Implementation of automated continuous retraining pipelines triggered by data drift alerts.
- Incorporation of alternative data sources (utility payments, rent verification) for unbanked borrowers.
