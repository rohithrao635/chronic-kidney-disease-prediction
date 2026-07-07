from flask import Flask, render_template, request
import numpy as np
import joblib
import csv
import os

app = Flask(__name__)

# load model
model = joblib.load("kidney_model.pkl")
scaler = joblib.load("scaler.pkl")


# ------------------------------
# FUNCTION TO SAVE PATIENT DATA
# ------------------------------
def save_to_csv(sg, hemo, al, rc, htn, dm, appet, pc, result):

    file_name = "patient_records.csv"

    header = ["sg","hemo","al","rc","htn","dm","appet","pc","prediction"]

    row = [sg, hemo, al, rc, htn, dm, appet, pc, result]

    file_exists = os.path.isfile(file_name)

    with open(file_name, "a", newline="") as file:
        writer = csv.writer(file)

        # write header only once
        if not file_exists:
            writer.writerow(header)

        writer.writerow(row)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    sg = float(request.form["sg"])
    hemo = float(request.form["hemo"])
    al = float(request.form["al"])
    rc = float(request.form["rc"])
    htn = float(request.form["htn"])
    dm = float(request.form["dm"])
    appet = float(request.form["appet"])
    pc = float(request.form["pc"])

    data = np.array([[sg, hemo, al, rc, htn, dm, appet, pc]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "CKD"
    else:
        result = "NOTCKD"

    # SAVE DATA
    save_to_csv(sg, hemo, al, rc, htn, dm, appet, pc, result)

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)