import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/cloud_metrics.csv")

X = df[['memory','disk','network']]
y = df['cpu']

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X,y)

joblib.dump(model,"model.pkl")

print("Model trained and saved")