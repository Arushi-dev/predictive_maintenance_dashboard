# utils/feature_engineering.py

def add_rolling_features(df, window=5):
    """
    Adds rolling mean and std for each sensor per unit.
    """
    df = df.copy()
    sensor_cols = [col for col in df.columns if 'sensor_' in col]

    for col in sensor_cols:
        df[f'{col}_mean'] = df.groupby('unit')[col].rolling(window=window).mean().reset_index(level=0, drop=True)
        df[f'{col}_std'] = df.groupby('unit')[col].rolling(window=window).std().reset_index(level=0, drop=True)

    return df
