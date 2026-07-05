import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/Crop_recommendation.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# -----------------------------
# Logistic Regression Model
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("\nTraining Logistic Regression Model...")

model.fit(X_train, y_train)

print("Training Completed Successfully")

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "models/logistic_model.pkl")

print("\nModel Saved Successfully")