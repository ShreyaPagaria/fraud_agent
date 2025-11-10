# Orchestrator: runs detectors, fuses, explains
import numpy as np
from preprocess import load_and_preprocess
from models import train_supervised, train_unsupervised, save_models, load_models, predict_all
from explainer import shap_top_features, llm_explain_or_fallback
from eval import metrics_from_scores, confusion, binarize_scores

def train_all(data_path):
    df, X_train, X_test, y_train, y_test, scalers, feature_names = load_and_preprocess(data_path)
    xgb, rf = train_supervised(X_train, y_train)
    iso, ae  = train_unsupervised(X_train, y_train, input_dim=X_train.shape[1])
    save_models(xgb, rf, iso, ae)
    return (df, X_train, X_test, y_train, y_test, feature_names)

def run_all(X_test, y_test, models, feature_names, thresholds):
    xgb, rf, iso, ae = models
    xgb_p, rf_p, iso_s, ae_s = predict_all(xgb, rf, iso, ae, X_test)

    # metrics per detector
    results = {}
    for name, scores in [("xgb", xgb_p), ("rf", rf_p)]:
        m = metrics_from_scores(y_test, scores)
        yb = binarize_scores(scores, thresholds.get(name, 0.5))
        cm, p, r, f1 = confusion(y_test, yb)
        m.update({"cm": cm, "precision": p, "recall": r, "f1": f1})
        results[name] = m

    for name, scores in [("iso", iso_s), ("ae", ae_s)]:
        # Use percentile thresholds for anomalies
        thr = thresholds.get(name, np.percentile(scores, 100*(1 - y_test.mean())))
        m = metrics_from_scores(y_test, scores)
        yb = binarize_scores(scores, thr)
        cm, p, r, f1 = confusion(y_test, yb)
        m.update({"cm": cm, "precision": p, "recall": r, "f1": f1, "threshold": thr})
        results[name] = m

    # ensemble
    # ens = ensemble_score(xgb_p, rf_p, iso_s, ae_s)
    # thr_e = thresholds.get("ensemble", 0.5)
    # m = metrics_from_scores(y_test, ens)
    # yb = binarize_scores(ens, thr_e)
    # cm, p, r, f1 = confusion(y_test, yb)
    # m.update({"cm": cm, "precision": p, "recall": r, "f1": f1, "threshold": thr_e})
    # results["ensemble"] = m

    # payload = {"scores": {"xgb": xgb_p, "rf": rf_p, "iso": iso_s, "ae": ae_s, "ensemble": ens}}
    # return results, payload
    payload = {"scores": {"xgb": xgb_p, "rf": rf_p, "iso": iso_s, "ae": ae_s}}
    return results, payload

# def explain_transaction(idx, X_test, models, feature_names, payload, llm_client=None):
#     xgb, rf, iso, ae = models
#     row = X_test.iloc[[idx]]
#     xgb_p = xgb.predict_proba(row)[:,1][0]
#     rf_p  = rf.predict_proba(row)[:,1][0]
#     iso_s = -iso.decision_function(row)[0]
#     recon = ae.predict(row, verbose=0)
#     ae_s  = float(np.mean(np.square(row - recon), axis=1)[0])

#     shap_dict = shap_top_features(xgb, row, feature_names, max_k=5)
#     proba_dict = {"xgb": float(xgb_p), "rf": float(rf_p)}
#     anomaly_dict = {"iso": float(iso_s), "ae": float(ae_s)}

#     text = llm_explain_or_fallback(idx, proba_dict, anomaly_dict, shap_dict, row, llm_client)
#     return text, shap_dict, proba_dict, anomaly_dict
def explain_transaction(idx, X_test, models, feature_names, payload, llm_client=None):
    xgb, rf, iso, ae = models

    # keep a 1-row DataFrame for SHAP, but use a NumPy view for math
    row_df = X_test.iloc[[idx]]
    row_np = row_df.to_numpy()              # shape: (1, n_features)

    # Supervised probs
    xgb_p = float(xgb.predict_proba(row_np)[:, 1][0])
    rf_p  = float(rf.predict_proba(row_np)[:, 1][0])

    # Unsupervised scores
    iso_s = float(-iso.decision_function(row_np)[0])

    recon = ae.predict(row_np, verbose=0)   # shape: (1, n_features)
    ae_s  = float(((row_np - recon) ** 2).mean())   # scalar; or .mean(axis=1).item()

    # SHAP can still take the DataFrame to preserve feature names
    shap_dict = shap_top_features(xgb, row_df, feature_names, max_k=5)

    proba_dict   = {"xgb": xgb_p, "rf": rf_p}
    anomaly_dict = {"iso": iso_s, "ae": ae_s}

    text = llm_explain_or_fallback(idx, proba_dict, anomaly_dict, shap_dict, row_df, llm_client)
    return text, shap_dict, proba_dict, anomaly_dict
