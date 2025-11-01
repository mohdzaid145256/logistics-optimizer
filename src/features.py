import pandas as pd

def create_features(df):
    """
    Safely create engineered features for both training and new unseen data.
    Automatically skips missing columns like delivery times.
    """
    # --- Delay hours (only if both columns exist)
    if "actual_delivery_time" in df.columns and "promised_delivery_time" in df.columns:
        df["delay_hours"] = (
            pd.to_datetime(df["actual_delivery_time"]) -
            pd.to_datetime(df["promised_delivery_time"])
        ).dt.total_seconds() / 3600
    else:
        df["delay_hours"] = 0  # default for unseen data

    # --- Distance efficiency
    if "distance_km" in df.columns and "estimated_travel_time" in df.columns:
        df["distance_efficiency"] = df["distance_km"] / df["estimated_travel_time"]
    else:
        df["distance_efficiency"] = 0

    # --- Normalize vehicle age
    if "vehicle_age_norm" not in df.columns and "vehicle_age" in df.columns:
        df["vehicle_age_norm"] = df["vehicle_age"] / df["vehicle_age"].max()

    return df

