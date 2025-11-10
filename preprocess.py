# Load, clean, scale and split the dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC_TO_SCALE = ["Amount"]  # drop Time downstream per your plan

def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    # scale Amount
    scalers = {}
    for col in NUMERIC_TO_SCALE:
        sc = StandardScaler()
        df[col] = sc.fit_transform(df[col].values.reshape(-1,1))
        scalers[col] = sc

    X = df.drop(["Class","Time"], axis=1)  # you asked to drop Time
    y = df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return df, X_train, X_test, y_train, y_test, scalers, list(X.columns)
