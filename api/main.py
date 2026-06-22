from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "XAI Cloud Resource Management API"}

@app.post("/predict")
def predict(memory: float, disk: float, network: float):

    sample = pd.DataFrame({
        "memory":[memory],
        "disk":[disk],
        "network":[network]
    })

    prediction = model.predict(sample)[0]

    return {
        "predicted_cpu": round(float(prediction),2)
    }