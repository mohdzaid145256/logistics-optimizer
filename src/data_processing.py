import pandas as pd

def load_and_merge(order_fp, delivery_fp, route_fp=None, vehicle_fp=None):
    orders = pd.read_csv(order_fp)
    delivery = pd.read_csv(delivery_fp)

    df = pd.merge(orders, delivery, on="order_id", how="left")

    if route_fp:
        routes = pd.read_csv(route_fp)
        df = pd.merge(df, routes, on="route_id", how="left")

    if vehicle_fp:
        vehicles = pd.read_csv(vehicle_fp)
        df = pd.merge(df, vehicles, on="vehicle_id", how="left")

    # Basic cleanup
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    return df
