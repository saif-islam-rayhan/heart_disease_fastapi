from fastapi import FastAPI
import joblib
from app.schemas import HeartInput

# Load model
model = joblib.load("model/heart_model.joblib")

app = FastAPI(title="Heart Disease Prediction API")

@app.get("/health")
def health():
    return {"status": "API is running"}

@app.get("/info")
def info():
    return {
        "model": "Random Forest / Logistic Regression",
        "features": [
            "age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"
        ]
    }

@app.post("/predict")
def predict(data: HeartInput):
    input_data = [[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]]
    prediction = model.predict(input_data)[0]
    return {"heart_disease": bool(prediction)}
