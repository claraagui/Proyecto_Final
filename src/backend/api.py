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

EXPERIMENT_NAME = "/Users/aclarapao@gmail.com/proyecto_final_precios"

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

model_name = "workspace.default.proyecto_final_precios"
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

# Preprocess de entrada
def preprocesamiento(input_data):
    df = pd.DataFrame([input_data.dict()])
    texto = dv.transform(df['cleaned_text'])
    return texto

# Realizar predicciones
def hacer_predic(input_data):
    X = preprocesamiento(input_data)
    pred = champion_model.predict(X)
    return int(pred[0])

app = FastAPI()

class InputData(BaseModel):
    cleaned_text: str


@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = hacer_predic(input_data)
    return {"prediction": result}