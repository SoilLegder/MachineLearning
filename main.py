from fastapi import FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel
import joblib  # For saving and loading ML models

app = FastAPI()

# Sample model input schema
class SoilData(BaseModel):
    moisture: float
    temperature: float
    ph: float
    nitrogen: float
    phosphorus: float
    potassium: float

# Dummy ML model function
def predict_soil_quality(data: SoilData):
    score = (data.moisture * 0.3) + (data.temperature * 0.2) + (data.ph * 0.1) \
            + (data.nitrogen * 0.2) + (data.phosphorus * 0.1) + (data.potassium * 0.1)
    return {"soil_quality_score": round(score, 2)}

@app.get("/")
def home():
    return {"message": "Soil Ledger ML Service Running"}

@app.post("/predict")
def predict(data: SoilData):
    prediction = predict_soil_quality(data)
    return {"prediction": prediction}
