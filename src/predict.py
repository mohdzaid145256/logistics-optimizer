import pandas as pd
import joblib
import os
from src.features import create_features

model_path = "models/final_model.pkl"
new_data_path = "data/new_orders.csv"

print("\n🔍 Loading model and new data...")

# --- Load model ---
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model not found. Run train_final_model.py first.")
model = joblib.load(model_path)

# --- Load new unseen data ---
if not os.path.exists(new_data_path):
    raise FileNotFoundError("❌ new_orders.csv not found inside /data folder.")
df_new = pd.read_csv(new_data_path)

# --- Apply same feature creation safely ---
try:
    df_new = create_features(df_new)
except KeyError:
    print("⚠️ Skipping delay feature — columns for actual/promise time not found.")

# --- Ensure necessary features exist ---
features = ["distance_km", "estimated_travel_time", "vehicle_age_norm", "distance_efficiency"]
missing = [f for f in features if f not in df_new.columns]
if missing:
    raise ValueError(f"❌ Missing required features: {missing}")

# --- Predict ---
X_new = df_new[features]
df_new["predicted_delay"] = model.predict(X_new)

# --- Save predictions ---
os.makedirs("predictions", exist_ok=True)
output_path = "predictions/new_predictions.csv"
df_new.to_cs
