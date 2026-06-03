import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/careerboost_dataset.csv")

# Encode categorical columns
job_encoder = LabelEncoder()
degree_encoder = LabelEncoder()

df["JobRole"] = job_encoder.fit_transform(df["JobRole"])
df["Degree"] = degree_encoder.fit_transform(df["Degree"])

# Features & Target
X = df.drop("Selected", axis=1)
y = df["Selected"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

# Save Model
joblib.dump(model, "models/model.pkl")
joblib.dump(job_encoder, "models/job_encoder.pkl")
joblib.dump(degree_encoder, "models/degree_encoder.pkl")

print("Model Saved Successfully!")