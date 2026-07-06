from flask import Flask, render_template, request
import pickle
import numpy as np

# ==========================================================
# Backend Initialization
# ==========================================================

# Load the trained Machine Learning model
model = pickle.load(open("models/crop_model.pkl", "rb"))

# Create Flask application
app = Flask(__name__)

# ==========================================================
# Route Definitions
# ==========================================================

# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# About Page
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/find-your-crop')
def findyourcrop():
    return render_template('find_your_crop.html')

# ==========================================================
# Crop Prediction
# ==========================================================

@app.route('/predict', methods=['POST'])
def predict():

    # Read values from HTML form
    nitrogen = float(request.form['Nitrogen'])
    phosphorous = float(request.form['Phosphorous'])
    potassium = float(request.form['Potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Prepare input for prediction
    input_data = np.array([[nitrogen,
                            phosphorous,
                            potassium,
                            temperature,
                            humidity,
                            ph,
                            rainfall]])

    # Predict crop
    prediction = model.predict(input_data)

    crop = prediction[0]

    # Display result on HTML page
    return render_template(
    "find_your_crop.html",
    prediction_text=f"Recommended Crop: {crop}"
)


# ==========================================================
# Main Function
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)