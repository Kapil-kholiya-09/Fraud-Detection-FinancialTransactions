from flask import Flask, render_template, request
import joblib
import pandas as pd
from model_class import GaussianAnomalyDetector

app = Flask(__name__)

# Load model
model = joblib.load("fraud_model.pkl")
features = joblib.load("features.pkl")

EPSILON = 1e-18   # Use your best value


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""

    if request.method == "POST":

        data = {
            "V4": float(request.form["V4"]),
            "V11": float(request.form["V11"]),
            "V12": float(request.form["V12"]),
            "V14": float(request.form["V14"]),
            "V16": float(request.form["V16"]),
            "V17": float(request.form["V17"]),
            "V18": float(request.form["V18"]),
            "V19": float(request.form["V19"]),
            "V15": float(request.form["V15"]),
        }

        df = pd.DataFrame([data])
        df = df[features]

        pred = model.predict(df, EPSILON)

        prediction_text = "Fraud" if pred[0] == 1 else "Not Fraud"

    return render_template("index.html", prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
