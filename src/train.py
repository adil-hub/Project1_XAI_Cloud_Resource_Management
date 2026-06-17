
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.DataFrame({
 'memory':[60,80,40,90],
 'disk':[50,70,30,80],
 'network':[20,40,10,60],
 'cpu':[55,78,35,90]
})

X=df[['memory','disk','network']]
y=df['cpu']

model=RandomForestRegressor()
model.fit(X,y)

joblib.dump(model,'model.pkl')
print('Model saved')
