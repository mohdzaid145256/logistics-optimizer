from src.data_processing import load_and_merge
from src.features import create_features
from src.model import train_model

def main():
    print("🚀 Starting model training...")
    df = load_and_merge("data/orders.csv", "data/delivery_performance.csv")
    print("✅ Data merged successfully")
    df = create_features(df)
    print("✅ Features created successfully")
    train_model(df)
    print("✅ Model trained and saved at models/delay_predictor.joblib")

if __name__ == "__main__":
    main()

