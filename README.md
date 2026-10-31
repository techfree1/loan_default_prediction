# Bank Retail Credit Risk & MLOps Platform

## Project Overview
An end-to-end credit risk assessment system that predicts loan default probabilities using Machine Learning. The platform provides real-time loan scoring, explainable AI, model monitoring, and audit logging.

## Technologies Used
- **Programming Language:** Python 3.11
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Machine Learning:** XGBoost, Random Forest, Logistic Regression, Scikit-learn
- **Explainable AI:** SHAP
- **MLOps:** MLflow, Evidently AI
- **Database:** PostgreSQL, SQLite
- **Deployment:** Docker

## Key Features
- Real-time loan default risk prediction.
- Loan approval decisions and risk classification.
- Model explainability using SHAP.
- Data drift detection using statistical tests.
- Model performance tracking with MLflow.
- Audit logging for loan applications.
- Interactive dashboard for risk analysis.
- Docker-based deployment.

## Model Performance
The XGBoost model achieved a **94.83% ROC-AUC score** on the holdout test dataset.

## Project Structure
- `app/` – FastAPI backend, database, and Streamlit dashboard.
- `src/` – Data preprocessing, model training, and drift monitoring.
- `data/` – Dataset files.
- `models/` – Trained models and preprocessing artifacts.
- `reports/` – Model evaluation, explainability, and drift reports.
- `run_pipeline.py` – End-to-end ML pipeline.

## Installation and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/raghav989/loan-default-prediction.git
cd loan-default-prediction
```

### 2. Create a Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Backend
```bash
uvicorn app.main:app --reload
```

Access the API documentation at `http://localhost:8000/docs`.

### 5. Run the Dashboard
Open another terminal and execute:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`.

## Author
**Raghav**

GitHub: https://github.com/techfree1/loan_default_prediction/n
Deployed site:- https://github.com/techfree1/loan_default_prediction
