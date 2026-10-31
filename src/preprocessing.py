import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data anomalies and engineers Bank Muscat retail credit risk features.
    """
    df = df.copy()
    
    # Filter realistic age and employment length bounds (removes data entry errors > 100 yrs)
    if "person_age" in df.columns:
        df = df[df["person_age"] <= 90]
    if "person_emp_length" in df.columns:
        df = df[(df["person_emp_length"] <= 60) | (df["person_emp_length"].isna())]

    # Impute raw missing values before ratio calculation so engineered features have no NaNs
    emp_median = df["person_emp_length"].median() if "person_emp_length" in df.columns else 4.0
    rate_median = df["loan_int_rate"].median() if "loan_int_rate" in df.columns else 11.0
    df["person_emp_length"] = df["person_emp_length"].fillna(emp_median)
    df["loan_int_rate"] = df["loan_int_rate"].fillna(rate_median)

    # Engineered Banking Underwriting Ratios
    # 1. Exact Loan-to-Income Ratio
    df["loan_to_income_ratio"] = df["loan_amnt"] / (df["person_income"] + 1.0)
    
    # 2. Annual Interest Burden to Income Ratio
    df["interest_burden_ratio"] = (df["loan_amnt"] * (df["loan_int_rate"] / 100.0)) / (df["person_income"] + 1.0)
    
    # 3. Income Stability Index (Income per year of employment)
    df["income_per_emp_year"] = df["person_income"] / (df["person_emp_length"] + 1.0)
    
    # 4. Credit Bureau Maturity Ratio (Credit history length relative to adult age)
    df["credit_maturity_ratio"] = df["cb_person_cred_hist_length"] / np.maximum(df["person_age"] - 18.0, 1.0)

    return df

def build_preprocessing_pipeline(X: pd.DataFrame):
    """
    Builds a reproducible scikit-learn ColumnTransformer for numerical and categorical features.
    """
    numeric_features = [
        "person_age", "person_income", "person_emp_length", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "loan_to_income_ratio", "interest_burden_ratio",
        "income_per_emp_year", "credit_maturity_ratio"
    ]
    categorical_features = [
        "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features
