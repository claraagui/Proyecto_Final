# General Libraries
import os
import pandas as pd
import numpy as np

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Databricks Env
import pathlib
import pickle
from dotenv import load_dotenv

# Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optimization
import math
import optuna
from optuna.samplers import TPESampler

# MLFlow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
# safe_databricks_setup.py
from dotenv import load_dotenv
import os
import mlflow

from prefect import flow, task
from typing import Tuple, List, Optional


@task(name="Load Data")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@task(name="Train/Val/Test Split")
def make_splits(df: pd.DataFrame, test_size: float = 0.2, val_ratio_from_train: float = 0.25, random_state: int = 42, shuffle: bool = False):
    y = df["price"]
    X = df.drop(columns=["price"])
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio_from_train, random_state=random_state, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test

@task(name="Preprocessor")
def preprocessor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    save_data: bool = False,
    save_artifacts: bool = True,
    artifacts_dir: str = "../../artifacts/preprocessor"
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], OneHotEncoder, StandardScaler, List[str]]:

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_val = X_val.copy() if X_val is not None else None

    # Imputación simple
    if 'pets_allowed' in X_train.columns:
        for df in [X_train, X_test] + ([X_val] if X_val is not None else []):
            df['pets_allowed'] = df['pets_allowed'].fillna(0)

    numeric_cols = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'amenities_count']
    for col in numeric_cols:
        if col in X_train.columns:
            med = X_train[col].median()
            for df in [X_train, X_test] + ([X_val] if X_val is not None else []):
                df[col] = df[col].fillna(med)

    # One-hot encode
    cat_cols = [c for c in ["category", "has_photo", "pets_allowed", "cityname", "state"] if c in X_train.columns]
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    if len(cat_cols) > 0:
        encoder.fit(X_train[cat_cols])
        X_train_cat = encoder.transform(X_train[cat_cols])
        X_test_cat  = encoder.transform(X_test[cat_cols])
        X_val_cat   = encoder.transform(X_val[cat_cols]) if X_val is not None else None
        cat_feature_names = encoder.get_feature_names_out(cat_cols)

        X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train.index)
        X_test_cat_df  = pd.DataFrame(X_test_cat,  columns=cat_feature_names, index=X_test.index)
        X_val_cat_df   = pd.DataFrame(X_val_cat,   columns=cat_feature_names, index=X_val.index) if X_val is not None else None

        for df in [X_train, X_test] + ([X_val] if X_val is not None else []):
            df.drop(columns=cat_cols, inplace=True)

        X_train_final = pd.concat([X_train, X_train_cat_df], axis=1)
        X_test_final  = pd.concat([X_test,  X_test_cat_df],  axis=1)
        X_val_final   = pd.concat([X_val,   X_val_cat_df],   axis=1) if X_val is not None else None
    else:
        X_train_final = X_train
        X_test_final  = X_test
        X_val_final   = X_val

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled  = scaler.transform(X_test_final)
    X_val_scaled   = scaler.transform(X_val_final) if X_val is not None else None

    # Guardar artefactos
    if save_artifacts:
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, "encoder.pkl"), 'wb') as f:
            pickle.dump(encoder, f)
        with open(os.path.join(artifacts_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        try:
            mlflow.log_artifact(os.path.join(artifacts_dir, "encoder.pkl"), artifact_path="preprocessor")
            mlflow.log_artifact(os.path.join(artifacts_dir, "scaler.pkl"), artifact_path="preprocessor")
        except Exception as e:
            print("MLflow artifact logging skipped/failed:", e)

    return X_train_scaled, X_test_scaled, X_val_scaled, encoder, scaler, list(X_train_final.columns)


# ---------------------------
# ---- HYPERPARAM TUNING ----
# ---------------------------

@task(name="HP Tuning RF")
def hp_tuning_rf(X_train, X_test, y_train, y_test, X_val=None, y_val=None, n_trials=10):
    mlflow.sklearn.autolog()
    sampler = TPESampler(seed=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
            "n_jobs": -1
        }

        X_train_scaled, X_test_scaled, X_val_scaled, _, _, _ = preprocessor(X_train, X_test, X_val)

        eval_X = X_val_scaled if (X_val is not None and y_val is not None) else X_test_scaled
        eval_y = y_val if (X_val is not None and y_val is not None) else y_test

        model = RandomForestRegressor(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(eval_X)
        return float(np.sqrt(mean_squared_error(eval_y, y_pred)))

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# --- HP Tuning XGB ---
@task(name="HP Tuning XGB")
def hp_tuning_xgb(X_train, X_test, y_train, y_test, X_val=None, y_val=None, n_trials=10):
    mlflow.xgboost.autolog()
    sampler = TPESampler(seed=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }

        X_train_scaled, X_test_scaled, X_val_scaled, _, _, _ = preprocessor(X_train, X_test, X_val)
        eval_X = X_val_scaled if (X_val is not None and y_val is not None) else X_test_scaled
        eval_y = y_val if (X_val is not None and y_val is not None) else y_test

        model = xgb.XGBRegressor(objective='reg:squarederror', **params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(eval_X)
        return float(np.sqrt(mean_squared_error(eval_y, y_pred)))

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# --- HP Tuning LGBM ---
@task(name="HP Tuning LGBM")
def hp_tuning_lgbm(X_train, X_test, y_train, y_test, X_val=None, y_val=None, n_trials=10):
    sampler = TPESampler(seed=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", -1, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }

        X_train_scaled, X_test_scaled, X_val_scaled, _, _, _ = preprocessor(X_train, X_test, X_val)
        eval_X = X_val_scaled if (X_val is not None and y_val is not None) else X_test_scaled
        eval_y = y_val if (X_val is not None and y_val is not None) else y_test

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(eval_X)
        return float(np.sqrt(mean_squared_error(eval_y, y_pred)))

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# --- TRAIN BEST MODELS ---
@task(name="Train Best Models")
def train_best_models(
    X_train, y_train, X_test, y_test,
    best_params_rf, best_params_xgb, best_params_lgbm
):
    run_ids = {}

    # Random Forest
    X_train_scaled, X_test_scaled, _, _, _, _ = preprocessor(X_train, X_test)
    with mlflow.start_run(run_name="Best Random Forest Regressor"):
        model = RandomForestRegressor(**best_params_rf)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mlflow.log_metric("rmse", float(np.sqrt(mean_squared_error(y_test, y_pred))))
        mlflow.log_metric("mae", float(mean_absolute_error(y_test, y_pred)))
        mlflow.log_metric("r2", float(r2_score(y_test, y_pred)))
        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
        run_ids['rf'] = mlflow.active_run().info.run_id

    # XGBoost
    X_train_scaled, X_test_scaled, _, _, _, _ = preprocessor(X_train, X_test)
    with mlflow.start_run(run_name="Best XGBoost Regressor"):
        model = xgb.XGBRegressor(objective='reg:squarederror', **best_params_xgb)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mlflow.log_metric("rmse", float(np.sqrt(mean_squared_error(y_test, y_pred))))
        mlflow.log_metric("mae", float(mean_absolute_error(y_test, y_pred)))
        mlflow.log_metric("r2", float(r2_score(y_test, y_pred)))
        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.xgboost.log_model(model, artifact_path="model", signature=signature)
        run_ids['xgb'] = mlflow.active_run().info.run_id

    # LightGBM
    X_train_scaled, X_test_scaled, _, _, _, _ = preprocessor(X_train, X_test)
    with mlflow.start_run(run_name="Best LightGBM Regressor"):
        model = lgb.LGBMRegressor(**best_params_lgbm)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mlflow.log_metric("rmse", float(np.sqrt(mean_squared_error(y_test, y_pred))))
        mlflow.log_metric("mae", float(mean_absolute_error(y_test, y_pred)))
        mlflow.log_metric("r2", float(r2_score(y_test, y_pred)))
        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.lightgbm.log_model(model, artifact_path="model", signature=signature)
        run_ids['lgbm'] = mlflow.active_run().info.run_id

    return run_ids['rf'], run_ids['xgb'], run_ids['lgbm']


# --- REGISTER MODELS ---
@task(name="Register Champion/Challenger")
def register_champion_challenger_reg(experiment_name: str, model_registry_name: str, metric: str = "r2"):
    client = MlflowClient()
    order = "DESC" if metric == "r2" else "ASC"

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.candidate = 'true'",
        order_by=[f"metrics.{metric} {order}"]
    )

    if runs.empty:
        print("No candidate runs found.")
        return

    champion = runs.iloc[0]
    challenger = runs.iloc[1] if len(runs) > 1 else None

    def register(run_row, alias):
        if run_row is None:
            print(f"No {alias} available.")
            return
        run_id = run_row["run_id"]
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_registry_name
        )
        client.set_registered_model_alias(name=model_registry_name, alias=alias, version=result.version)
        print(f"{alias} registrado: Run ID {run_id}")

    register(champion, "Champion")
    register(challenger, "Challenger")

# -------- MAIN FLOW -------
@flow(name="Proyecto-Final-Precios-Flow")
def main_flow():
    load_dotenv(override=True)
    mlflow.set_tracking_uri("databricks")
    EXPERIMENT_NAME = "/Users/aclarapao@gmail.com/proyecto-final-precios_prefect"
    mlflow.set_experiment(EXPERIMENT_NAME)
    MODEL_REGISTRY_NAME = "workspace.default.proyecto_final_precios_prefect"
    mlflow.sklearn.autolog()

    df = load_data("C:/Users/Clara/Documents/Semestre5/Proyecto_Final/Proyecto_Final/data/processed/df_clean.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)

    best_rf = hp_tuning_rf(X_train, X_test, y_train, y_test, X_val=X_val, y_val=y_val, n_trials=10)
    best_xgb = hp_tuning_xgb(X_train, X_test, y_train, y_test, X_val=X_val, y_val=y_val, n_trials=10)
    best_lgbm = hp_tuning_lgbm(X_train, X_test, y_train, y_test, X_val=X_val, y_val=y_val, n_trials=10)

    rf_run_id, xgb_run_id, lgbm_run_id = train_best_models(
        X_train, y_train, X_test, y_test,
        best_params_rf=best_rf,
        best_params_xgb=best_xgb,
        best_params_lgbm=best_lgbm
    )

    register_champion_challenger_reg(EXPERIMENT_NAME, MODEL_REGISTRY_NAME, metric="r2")


if __name__ == "__main__":
    main_flow()