from src.data_processing import load_and_merge
from src.features import create_features
from src.model import train_model

if __name__ == "__main__":
    print("🚀 Loading and preparing data...")
    df = load_and_merge("data/orders.csv", "data/delivery_performance.csv")
    df = create_features(df)

    print("🧠 Training model...")
    train_model(df)


