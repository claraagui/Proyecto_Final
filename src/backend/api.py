import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
from dotenv import load_dotenv
import os
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

load_dotenv(override=True)  # Carga las variables del archivo .env

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

EXPERIMENT_NAME = "/Users/aclarapao@gmail.com/proyecto_final_precios_3"

run_ = mlflow.search_runs(order_by=['metrics.r2 ASC'],
                          output_format="list",
                          experiment_names=[EXPERIMENT_NAME]
                          )[0]

run_id = run_.info.run_id

client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

with open("preprocessor/encoder.pkl", "rb") as f1_in:
    encoder = pickle.load(f1_in)
with open("preprocessor/scaler.pkl", "rb") as f2_in:
    scaler = pickle.load(f2_in)
with open("feature_columns.pkl", "rb") as f3_in:
    feature_columns = pickle.load(f3_in)

model_name = "workspace.default.proyecto_final_precios_3"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

def preprocesamiento(input_data):
    cat_cols = ["category", "has_photo", "pets_allowed", "cityname", "state"]

    # Convertir input a df
    df = pd.DataFrame([input_data.dict()])

    # --- 1) OneHot ---
    X_cat = encoder.transform(df[cat_cols])
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names, index=df.index)

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, X_cat_df], axis=1)

    # --- 2) Asegurar TODAS las columnas del entrenamiento ---
    df = df.reindex(columns=feature_columns, fill_value=0)

    # --- 3) Escalar ---
    df_scaled = scaler.transform(df)

    print("\n=== COLUMNAS API ===")
    print(len(df.columns))
    print(sorted(df.columns))

    print("\n=== COLUMNAS MODELO ===")
    print(len(feature_columns))
    print(sorted(feature_columns))

    return pd.DataFrame(df_scaled, columns=feature_columns)

# Realizar predicciones
def hacer_predic(input_data):
    X = preprocesamiento(input_data)
    pred = champion_model.predict(X)
    return int(pred[0])

app = FastAPI()

class InputData(BaseModel):
    category: str
    bathrooms: float
    bedrooms: int
    has_photo: str
    pets_allowed: float
    square_feet: int
    cityname: int
    state: int
    latitude: float
    longitude: float
    amenities_count: int


@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = hacer_predic(input_data)
    return {"prediction": result}