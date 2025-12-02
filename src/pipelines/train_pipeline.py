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

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

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
def preprocessor(X_train, X_test, X_val=None, save_artifacts=True):
    # Make copies so we don't mutate outside variables
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_val   = X_val.copy() if X_val is not None else None

    # 1) Impute missing values
    # Fill pets_allowed with 0 (assumption: NaN => no pets allowed)
    if 'pets_allowed' in X_train.columns:
        X_train['pets_allowed'] = X_train['pets_allowed'].fillna(0)
        X_test['pets_allowed']  = X_test['pets_allowed'].fillna(0)
        if X_val is not None:
            X_val['pets_allowed'] = X_val['pets_allowed'].fillna(0)

    # For other numeric columns use train median
    numeric_cols = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'amenities_count']
    for col in numeric_cols:
        if col in X_train.columns:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col]  = X_test[col].fillna(med)
            if X_val is not None:
                X_val[col] = X_val[col].fillna(med)

    # 2) One-Hot encode cityname and state together
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

        # drop original cat cols and concat encoded
        X_train = X_train.drop(columns=cat_cols)
        X_test  = X_test.drop(columns=cat_cols)
        X_val   = X_val.drop(columns=cat_cols) if X_val is not None else None

        X_train_final = pd.concat([X_train, X_train_cat_df], axis=1)
        X_test_final  = pd.concat([X_test,  X_test_cat_df],  axis=1)
        X_val_final   = pd.concat([X_val,   X_val_cat_df],   axis=1) if X_val is not None else None
    else:
        # no categorical cols found
        X_train_final = X_train
        X_test_final  = X_test
        X_val_final   = X_val

    # 3) Scale (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled  = scaler.transform(X_test_final)
    X_val_scaled   = scaler.transform(X_val_final) if X_val is not None else None

    feature_cols = list(X_train_final.columns)

    # 5) Save artifacts
    if save_artifacts:
        os.makedirs("artifacts/preprocessor", exist_ok=True)
        with open('artifacts/preprocessor/encoder.pkl', 'wb') as f_out:
            pickle.dump(encoder, f_out)
        with open('artifacts/preprocessor/scaler.pkl', 'wb') as f_out:
            pickle.dump(scaler, f_out)
        with open('artifacts/preprocessor/feature_columns.pkl', 'wb') as f_out:
            pickle.dump(feature_cols, f_out)


    # Return scaled arrays + artifacts + feature names so user can reconstruct dfs
    return X_train_scaled, X_test_scaled, X_val_scaled, encoder, scaler, list(X_train_final.columns)

# ---- HYPERPARAM TUNING ----

@task(name="HP Tuning RF")
def hp_tuning_rf_reg(X_train_scaled, X_test_scaled, y_train, y_test, X_val=None, y_val=None, n_trials=5):
    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
            "n_jobs": -1
        }

        with mlflow.start_run(nested=True):

            mlflow.set_tag("model_family", "random_forest_regressor")
            mlflow.log_params(params)
            mlflow.set_tags({'best': 'true'})

            mlflow.log_artifact("artifacts/preprocessor/encoder.pkl", artifact_path="preprocessor")
            mlflow.log_artifact("artifacts/preprocessor/scaler.pkl", artifact_path="preprocessor")
            mlflow.log_artifact("artifacts/preprocessor/feature_columns.pkl", artifact_path="preprocessor")
            
            model = RandomForestRegressor(**params)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            rms = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            mlflow.log_metric("rmse", rms)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            signature = infer_signature(X_test_scaled, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="rf_regressor", signature=signature)

        return rms  # Optuna will minimize RMSE

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    with mlflow.start_run(run_name="RF Regression (Optuna)", nested=True):
        study.optimize(objective_rf, n_trials=n_trials)

    return study.best_params

# --- HP Tuning XGB ---
@task(name="HP Tuning XGB")
def hp_tuning_xgb_reg(X_train_scaled, X_test_scaled, y_train, y_test, X_val=None, y_val=None, n_trials=5):
    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }

        with mlflow.start_run(nested=True):

            mlflow.set_tag("model_family", "xgboost_regressor")
            mlflow.log_params(params)
            mlflow.set_tags({'best': 'true'})
            
            mlflow.log_artifact("artifacts/preprocessor/encoder.pkl", artifact_path="preprocessor")
            mlflow.log_artifact("artifacts/preprocessor/scaler.pkl", artifact_path="preprocessor")
            mlflow.log_artifact("artifacts/preprocessor/feature_columns.pkl", artifact_path="preprocessor")

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            rms = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            mlflow.log_metric("rmse", rms)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            signature = infer_signature(X_test_scaled, y_pred)
            mlflow.xgboost.log_model(model, artifact_path="xgb_regressor", signature=signature)

        return rms

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    with mlflow.start_run(run_name="XGB Regression (Optuna)", nested=True):
        study.optimize(objective_xgb, n_trials=n_trials)

    return study.best_params

# --- HP Tuning LGBM ---
@task(name="HP Tuning LGBM")
def hp_tuning_lgbm_reg(X_train_scaled, X_test_scaled, y_train, y_test, X_val=None, y_val=None, n_trials=5):
    def objective_lgbm(trial):
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

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "lightgbm_regressor")
            mlflow.log_params(params)
            mlflow.set_tags({'best': 'true'})
            
            mlflow.log_artifact("artifacts/preprocessor/encoder.pkl", artifact_path="preprocessor")
            mlflow.log_artifact("artifacts/preprocessor/scaler.pkl", artifact_path="preprocessor")
            mlflow.log_artifact("artifacts/preprocessor/feature_columns.pkl", artifact_path="preprocessor")

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            rms = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            mlflow.log_metric("rmse", rms)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            signature = infer_signature(X_test_scaled, y_pred)
            mlflow.lightgbm.log_model(model, artifact_path="lgbm_regressor", signature=signature)

        return rms

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    with mlflow.start_run(run_name="LightGBM Regression (Optuna)", nested=True):
        study.optimize(objective_lgbm, n_trials=n_trials)

    return study.best_params


# --- TRAIN BEST MODELS ---
@task(name="Train Best Models")
def train_best_models(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    best_params_rf,
    best_params_xgb,
    best_params_lgbm
):
    # 0) FILTRADO DE HIPERPARÁMETROS POR MODELO
    RF_VALID = {
        "n_estimators", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features", "bootstrap",
        "criterion", "random_state"
    }

    XGB_VALID = {
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree", "gamma",
        "lambda", "alpha"
    }

    LGBM_VALID = {
        "num_leaves", "learning_rate", "n_estimators",
        "min_child_samples", "subsample", "colsample_bytree",
        "reg_lambda", "reg_alpha"
    }

    def filter_params(params, valid):
        return {k: v for k, v in params.items() if k in valid}

    best_params_rf   = filter_params(best_params_rf, RF_VALID)
    best_params_xgb  = filter_params(best_params_xgb, XGB_VALID)
    best_params_lgbm = filter_params(best_params_lgbm, LGBM_VALID)

    print("RF params usados:", best_params_rf)
    print("XGB params usados:", best_params_xgb)
    print("LGBM params usados:", best_params_lgbm)

    # 1) RANDOM FOREST REGRESSOR
    mlflow.end_run()
    with mlflow.start_run(run_name='Best Random Forest Regressor', nested=True):
        
        mlflow.log_artifact("artifacts/preprocessor/encoder.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("artifacts/preprocessor/scaler.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("artifacts/preprocessor/feature_columns.pkl", artifact_path="preprocessor")
        mlflow.log_params(best_params_rf)

        model = RandomForestRegressor(**best_params_rf)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mlflow.log_metric("rmse", root_mean_squared_error(y_test, y_pred))
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.sklearn.log_model(model, "model", signature=signature)

    # 2) XGBOOST REGRESSOR
    mlflow.end_run()
    with mlflow.start_run(run_name='Best XGBoost Regressor', nested=True):
        
        mlflow.log_artifact("artifacts/preprocessor/encoder.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("artifacts/preprocessor/scaler.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("artifacts/preprocessor/feature_columns.pkl", artifact_path="preprocessor")

        mlflow.log_params(best_params_xgb)

        model = xgb.XGBRegressor(objective='reg:squarederror', **best_params_xgb)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mlflow.log_metric("rmse", root_mean_squared_error(y_test, y_pred))
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.xgboost.log_model(model, "model", signature=signature)

    # 3) LIGHTGBM REGRESSOR
    mlflow.end_run()
    with mlflow.start_run(run_name='Best LightGBM Regressor', nested=True):
        
        mlflow.log_artifact("artifacts/preprocessor/encoder.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("artifacts/preprocessor/scaler.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("artifacts/preprocessor/feature_columns.pkl", artifact_path="preprocessor")

        mlflow.log_params(best_params_lgbm)

        model = lgb.LGBMRegressor(**best_params_lgbm)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mlflow.log_metric("rmse", root_mean_squared_error(y_test, y_pred))
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.lightgbm.log_model(model, "model", signature=signature)


# --- REGISTER MODELS ---
@task(name="Register Champion/Challenger")
def register_champion_challenger_reg(experiment_name: str, model_registry_name: str, metric: str = "r2"):
    client = MlflowClient()
    order = "DESC" if metric == "r2" else "ASC"

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.best = 'true'",
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
    EXPERIMENT_NAME = "/Users/aclarapao@gmail.com/proyecto_final_precios_prefect3"
    mlflow.set_experiment(EXPERIMENT_NAME)
    MODEL_REGISTRY_NAME = "workspace.default.proyecto_final_precios_prefect3"

    df = load_data("../data/processed/df_clean.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)
    X_train_scaled, X_test_scaled, X_val_scaled, encoder, scaler, feature_cols = preprocessor(
    X_train, X_test, X_val, save_data=True, save_artifacts=True)

    mlflow.sklearn.autolog()
    best_rf = hp_tuning_rf_reg(X_train_scaled, X_test_scaled, y_train, y_test, X_val=X_val, y_val=y_val, n_trials=2)
    mlflow.sklearn.autolog()
    best_xgb = hp_tuning_xgb_reg(X_train_scaled, X_test_scaled, y_train, y_test, X_val=X_val, y_val=y_val, n_trials=2)
    mlflow.sklearn.autolog()
    best_lgbm = hp_tuning_lgbm_reg(X_train_scaled, X_test_scaled, y_train, y_test, X_val=X_val, y_val=y_val, n_trials=2)

    train_best_models(
        X_train, y_train, X_test, y_test,
        best_params_rf=best_rf,
        best_params_xgb=best_xgb,
        best_params_lgbm=best_lgbm
    )

    register_champion_challenger_reg(EXPERIMENT_NAME, MODEL_REGISTRY_NAME, metric="r2")


if __name__ == "__main__":
    main_flow()