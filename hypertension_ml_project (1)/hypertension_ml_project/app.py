
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

bundle = pickle.load(open("logreg_model.pkl","rb"))
model = bundle["model"]
columns = bundle["columns"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    data = {
        "gender": request.form["gender"],
        "age_group": request.form["age_group"],
        "family_history": request.form["family_history"],
        "patient_status": request.form["patient_status"],
        "medication": request.form["medication"],
        "time_since_diagnosis": request.form["time_since_diagnosis"],
        "symptom_severity": request.form["symptom_severity"],
        "shortness_breath": request.form["shortness_breath"],
        "visual_changes": request.form["visual_changes"],
        "nosebleeds": request.form["nosebleeds"],
        "systolic_range": request.form["systolic_range"],
        "diastolic_range": request.form["diastolic_range"],
        "diet_control": request.form["diet_control"]
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in columns:
        if col not in df:
            df[col] = 0

    df = df[columns]

    prediction = model.predict(df)[0]

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
