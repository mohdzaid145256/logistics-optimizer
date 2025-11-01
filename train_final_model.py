import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.data_processing import load_and_merge
from src.features import create_features

# Start timer
start_time = time.time()

print("🚀 Loading and preparing data...")
df = load_and_merge("data/orders.csv", "data/delivery_performance.csv")
df = create_features(df)

# Define target and features
target = "delayed"
features = ["distance_km", "estimated_travel_time", "vehicle_age_norm", "distance_efficiency"]

X = df[features]
y = df[target]

print(f"\n✅ Dataset ready: {X.shape[0]} rows, {X.shape[1]} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\n🧩 Training Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
mean_acc = scores.mean()

print("\n📊 Cross-validation accuracies:", scores)
print(f"Mean accuracy: {mean_acc:.3f}")

# Test performance
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print("\n🎯 Test Accuracy:", round(test_acc, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/final_random_forest.pkl")

# Save predictions
pred_df = pd.DataFrame({
    "order_id": X_test.index,
    "actual": y_test,
    "predicted": y_pred
})
pred_df.to_csv("predictions/final_predictions.csv", index=False)

# Feature importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🔥 Feature importances:")
print(importances)

# End summary
runtime = time.time() - start_time
print("\n✅ Training completed successfully!")
print(f"⏱️ Total runtime: {runtime:.2f} seconds")
print(f"🏆 Final Mean Accuracy: {mean_acc:.3f}")
print(f"⭐ Top feature: {importances.idxmax()} ({importances.max():.3f} importance)")

