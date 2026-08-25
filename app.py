from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Model
model = joblib.load("models/crop_yield_model.pkl")


try:
    df = pd.read_csv("dataset/crop_production.csv")
    
    states_list = sorted(df['State'].dropna().unique().tolist())
    crops_list = sorted(df['Crop'].dropna().unique().tolist())
except Exception as e:
    print(f"Error loading dataset for dropdowns: {e}")
    states_list = []
    crops_list = []

@app.route("/")
def home():
    return render_template("index.html", states=states_list, crops=crops_list)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        state = request.form["state"]
        crop = request.form["crop"]
        rainfall = float(request.form["rainfall"])

        input_data = pd.DataFrame([{
            "State": state,
            "Crop": crop,
            "Rainfall": rainfall
        }])

        prediction = model.predict(input_data)
        final_prediction = max(0, round(float(prediction[0]),2))

        return render_template(
            "index.html",
            prediction=final_prediction,
            state=state,
            crop=crop,
            rainfall=rainfall,
            states=states_list,
            crops=crops_list
        )
    except Exception as e:
        return render_template("index.html", error=str(e), states=states_list, crops=crops_list)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
