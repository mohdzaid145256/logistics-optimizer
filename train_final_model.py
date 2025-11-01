import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n🚀 Loading and preparing data...")

# --- Load dataset ---
df = pd.read_csv("data/logistics_data.csv")

# --- Define target column explicitly ---
target = "delayed"

# --- Verify target exists ---
if target not in df.columns:
    raise KeyError(f"❌ Target column '{target}' not found in dataset. Available columns: {df.columns.tolist()}")

# --- Define features ---
features = ["distance_km", "estimated_travel_time", "vehicle_age_norm", "distance_efficiency"]
X = df[features]
y = df[target]

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully! Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Save model ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/final_model.pkl")

print("\n💾 Model saved as: models/final_model.pkl")
