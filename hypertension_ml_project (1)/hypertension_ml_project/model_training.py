
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("patient_data.csv")

X = data.drop("stage", axis=1)
y = data["stage"]

X_encoded = pd.get_dummies(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_encoded, y)

bundle = {
    "model": model,
    "columns": list(X_encoded.columns)
}

pickle.dump(bundle, open("logreg_model.pkl", "wb"))

print("Model trained and saved as logreg_model.pkl")
