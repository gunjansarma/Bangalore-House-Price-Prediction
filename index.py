from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", "rb"))


@app.route("/")
def index():
    locations = sorted(data["location"].unique())
    return render_template("index.html", locations=locations)


@app.route("/predict", methods=["POST"])
def predict():
    location = request.form.get("location")
    bhk = request.form.get("bhk")
    bath = request.form.get("bath")
    sqft = request.form.get("total_sqft")

    try:
        bhk = float(bhk)
        bath = float(bath)
        sqft = float(sqft)
    except ValueError:
        return "Error: BHK, Bath, and Total Square Feet must be valid numbers."

    input_data = pd.DataFrame(
        [[location, bhk, bath, sqft]], columns=["location", "bhk", "bath", "total_sqft"]
    )

    try:
        prediction = pipe.predict(input_data)[0] * 1e5
        formatted_prediction = f"₹{np.round(prediction, 2)}"
        return f"Predicted Price: {formatted_prediction}"
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
