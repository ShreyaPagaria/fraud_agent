#Train/load/run the 4 models
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, models

def train_supervised(X_train, y_train):
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return xgb, rf

def train_unsupervised(X_train, y_train, input_dim):
    Xn = X_train[y_train==0]
    iso = IsolationForest(n_estimators=300, contamination=y_train.mean(), random_state=42).fit(Xn)

    ae = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(8,  activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xn, Xn, epochs=15, batch_size=256, validation_split=0.1, verbose=0)
    return iso, ae

def save_models(xgb, rf, iso, ae, dirpath="models"):
    import os
    os.makedirs(dirpath, exist_ok=True)
    joblib.dump(xgb, f"{dirpath}/xgb.bin")
    joblib.dump(rf,  f"{dirpath}/rf.pkl")
    joblib.dump(iso, f"{dirpath}/iso.pkl")
    ae.save(f"{dirpath}/ae.keras")

def load_models(dirpath="models"):
    import os
    from tensorflow.keras.models import load_model
    xgb = joblib.load(f"{dirpath}/xgb.bin")
    rf  = joblib.load(f"{dirpath}/rf.pkl")
    iso = joblib.load(f"{dirpath}/iso.pkl")
    ae  = tf.keras.models.load_model(f"{dirpath}/ae.keras")
    return xgb, rf, iso, ae

def predict_all(xgb, rf, iso, ae, X_test):
    # supervised: probabilities for class=1
    xgb_proba = xgb.predict_proba(X_test)[:,1]
    rf_proba  = rf.predict_proba(X_test)[:,1]
    # unsupervised: anomaly scores
    iso_scores = -iso.decision_function(X_test)          # higher = more anomalous
    recon = ae.predict(X_test, verbose=0)
    ae_mse = np.mean(np.square(X_test - recon), axis=1)  # higher = more anomalous
    return xgb_proba, rf_proba, iso_scores, ae_mse
