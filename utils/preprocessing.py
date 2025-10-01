# utils/preprocessing.py

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_features(df):
    feature_cols = df.columns.difference(['unit', 'cycle', 'RUL'])
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def save_dataset(df, filepath):
    # Windows-safe path handling with raw string
    df.to_csv(filepath, index=False)
