import fastapi
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.pipeline.inference_pipeline import PredictPipeline
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Defina os campos de entrada conforme o modelo espera
class PatientData(BaseModel):
	age: float
	sex: int
	cp: int
	trestbps: float
	chol: float
	fbs: int
	restecg: int
	thalach: float
	exang: int
	oldpeak: float
	slope: int
	ca: int
	thal: int

# Caminhos dos artefatos
#MODEL_PATH = os.path.join("artifacts", "heart_attack_model.pkl")
#PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

# Instancie o pipeline de predição
predict_pipeline = PredictPipeline()
predict_pipeline._load_artifacts()

@app.post("/predict")
async def predict(patient: PatientData):
	# Converta os dados recebidos para DataFrame
	df = pd.DataFrame([patient.model_dump()])
	pred, proba = predict_pipeline.predict(df)
	return {"prediction": int(pred), "probability": float(proba)}