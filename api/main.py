
from fastapi import FastAPI
import pandas as pd
import joblib

app=FastAPI()
model=joblib.load('model.pkl')

@app.get('/')
def home():
    return {'status':'running'}
