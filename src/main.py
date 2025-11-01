import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import resample

def train_model(df):
    # Use only numeric, varying columns
    features = [
        "distance_km",
        "estimated_travel_time",
        "vehicle_age_norm",
        "distance_efficiency"
    ]
    target = "delayed"

    # Drop any constant columns
    df = df.loc[:, df.nunique() > 1]

    # Balance dataset
    df_major = df[df[target] == 0]
    df_minor = df[df[target] == 1]
    df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
    df_balanced = pd.concat([df_major, df_minor_up])

    print("✅ Balanced dataset:", df_balanced[target].value_counts().to_dict())

    X = df_balanced[features]
    y = df_balanced[target]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Random Forest (more stable for small data)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

    joblib.dump((model, scaler), "models/delay_predictor.joblib")
    print("✅ Model saved at models/delay_predictor.joblib")


def predict_proba(df):
    (model, scaler) = joblib.load("models/delay_predictor.joblib")
    features = [
        "distance_km",
        "estimated_travel_time",
        "vehicle_age_norm",
        "distance_efficiency"
    ]
    X_scaled = scaler.transform(df[features])
    return model.predict_proba(X_scaled)[:, 1]
