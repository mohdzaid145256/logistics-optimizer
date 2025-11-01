import pandas as pd
import joblib
import os
from src.features import create_features

# --- File paths ---
model_path = "models/final_model.pkl"
new_data_path = "data/new_orders.csv"
output_path = "predictions/new_predictions.csv"

print("\n🔍 Loading model and new data...")

# --- Load model ---
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model not found. Run train_final_model.py first.")
model = joblib.load(model_path)

# --- Load new unseen data ---
if not os.path.exists(new_data_path):
    raise FileNotFoundError("❌ new_orders.csv not found inside /data folder.")
df_new = pd.read_csv(new_data_path)

# --- Create same features ---
df_new = create_features(df_new)

# --- Select features used during training ---
features = ["distance_km", "estimated_travel_time", "vehicle_age_norm", "distance_efficiency"]
X_new = df_new[features]

# --- Predict delay (classification) ---
df_new["predicted_delay"] = model.predict(X_new)

# --- Predict probability of delay ---
if hasattr(model, "predict_proba"):
    df_new["delay_probability"] = model.predict_proba(X_new)[:, 1]

# --- Save predictions ---
os.makedirs("predictions", exist_ok=True)
df_new.to_csv(output_path, index=False)

print("\n✅ Predictions complete!")
print(f"📁 Saved to: {output_path}")
print(df_new[["order_id", "predicted_delay", "delay_probability"]].head())

# --- Summary ---
total_orders = len(df_new)
delayed_orders = df_new["predicted_delay"].sum()
print(f"\n🚚 Total orders: {total_orders}, Predicted delayed: {delayed_orders}")
