import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/logistic_model.pkl")

print("=" * 50)
print("OptiCrop Crop Prediction System")
print("=" * 50)

# Take user input
N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter pH Value: "))
rainfall = float(input("Enter Rainfall (mm): "))

# Create DataFrame
input_data = pd.DataFrame({
    "N": [N],
    "P": [P],
    "K": [K],
    "temperature": [temperature],
    "humidity": [humidity],
    "ph": [ph],
    "rainfall": [rainfall]
})

# Predict
prediction = model.predict(input_data)

print("\n" + "=" * 50)
print("Recommended Crop:", prediction[0].upper())
print("=" * 50)
probabilities = model.predict_proba(input_data)

confidence = probabilities.max() * 100

print("\nRecommended Crop :", prediction[0])
print(f"Confidence : {confidence:.2f}%")
probabilities = model.predict_proba(input_data)
