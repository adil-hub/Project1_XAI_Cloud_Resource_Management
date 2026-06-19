import pandas as pd
import joblib
import shap

model = joblib.load("model.pkl")

sample = pd.DataFrame({
    "memory": [80],
    "disk": [75],
    "network": [40]
})

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(sample)

print("Prediction Explanation")

for feature, value in zip(sample.columns, shap_values[0]):
    print(f"{feature}: {value}")