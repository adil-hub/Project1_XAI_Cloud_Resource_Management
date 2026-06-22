import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

df = pd.read_csv("data/cloud_metrics.csv")

X = df[["memory", "disk", "network"]]

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, show=False)

plt.savefig("docs/shap_summary.png")

print("SHAP chart saved successfully")