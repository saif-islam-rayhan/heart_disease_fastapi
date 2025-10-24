import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# ১. Dataset load
df = pd.read_csv("/home/rayhan/Desktop/Heart_disease_fastapi/heart_disease_fastapi/heart.csv")  # downloaded from Kaggle

# ২. Feature & target
X = df.drop("target", axis=1)
y = df["target"]

# ৩. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ৪. Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Save model
joblib.dump(model, "model/heart_model.joblib")

# Optional: accuracy check
print("Model accuracy:", model.score(X_test, y_test))
