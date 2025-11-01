import pandas as pd

def summarize_kpis(df):
    total_orders = len(df)
    on_time_rate = (df["delayed"] == 0).mean() * 100
    avg_delay = df["delay_hours"].mean()
    return total_orders, on_time_rate, avg_delay
