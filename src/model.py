import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def train_model(df):
    features = ["distance_km", "estimated_travel_time", "vehicle_age_norm", "priority_encoded", "distance_efficiency"]
    target = "delayed"

    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))

    # Avoid ROC-AUC crash if only one class present
    if len(set(y_test)) > 1:
        print("ROC-AUC:", roc_auc_score(y_test, proba))
    else:
        print("⚠️ ROC-AUC skipped: only one class in test set")

    joblib.dump(model, "models/delay_predictor.joblib")
    print("✅ Model saved at models/delay_predictor.joblib")


def predict_proba(df):
    model = joblib.load("models/delay_predictor.joblib")
    features = ["distance_km", "estimated_travel_time", "vehicle_age_norm", "priority_encoded", "distance_efficiency"]
    return model.predict_proba(df[features])[:, 1]
