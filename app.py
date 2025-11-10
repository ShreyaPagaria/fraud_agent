#StreamlitUI
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from eval import plot_confusion_matrix
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 120

from preprocess import load_and_preprocess
from models import load_models, train_supervised, train_unsupervised, save_models, predict_all
from agent import train_all, run_all, explain_transaction
from explainer import make_ollama_client, is_ollama_up


st.set_page_config(page_title="Agentic Fraud Detector", layout="wide")

st.title("🔎 Agentic Financial Fraud Detector")
st.caption("Detectors: XGBoost, RandomForest, IsolationForest, Autoencoder • Self-explaining agent with SHAP + LLM")

uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"])
colA, colB = st.columns(2)
with colA:
    thr_xgb = st.slider("XGBoost threshold", 0.0, 1.0, 0.5, 0.01)
    thr_rf  = st.slider("RandomForest threshold", 0.0, 1.0, 0.5, 0.01)
with colB:
    thr_iso = st.slider("IsolationForest percentile threshold", 0.90, 0.999, 0.995, 0.001)
    thr_ae  = st.slider("Autoencoder percentile threshold", 0.90, 0.999, 0.995, 0.001)
# thr_ens = st.slider("Ensemble threshold", 0.0, 1.0, 0.5, 0.01)
st.markdown("---")
use_llm = st.checkbox(" Use Ollama for natural-language explanations", value=False)
ollama_model = st.text_input("Ollama model name", value="phi3:mini")


if uploaded is not None:
    # Save to temp and process
    path = r"C:\Users\pagar\OneDrive\Desktop\projects\fraud_agent\data\creditcard.csv"
    with open(path, "wb") as f: f.write(uploaded.getbuffer())
    df, X_train, X_test, y_train, y_test, scalers, feature_names = load_and_preprocess(path)

    # Load or train models
    try:
        models = load_models("models")
    except Exception:
        st.info("Training models for the first time…")
        from models import train_supervised, train_unsupervised, save_models
        xgb, rf = train_supervised(X_train, y_train)
        iso, ae  = train_unsupervised(X_train, y_train, input_dim=X_train.shape[1])
        save_models(xgb, rf, iso, ae)
        models = (xgb, rf, iso, ae)

    thresholds = {
        "xgb": thr_xgb, "rf": thr_rf,
        "iso": np.quantile(np.random.rand(len(X_test)), thr_iso),  # placeholder; eval module computes real one
        "ae":  np.quantile(np.random.rand(len(X_test)), thr_ae),
        # "ensemble": thr_ens
    }

    # Run detectors + ensemble
    from eval import metrics_from_scores, confusion, binarize_scores
    results, payload = run_all(X_test, y_test, models, feature_names, thresholds)

    st.subheader("Metrics summary (test set)")
    for key in ["xgb","rf","iso","ae"]:
        m = results[key]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{key} PR-AUC", f"{m['pr_auc']:.3f}")
        c2.metric(f"{key} ROC-AUC", f"{m['roc_auc']:.3f}")
        c3.metric(f"{key} Precision", f"{m['precision']:.3f}")
        c4.metric(f"{key} Recall", f"{m['recall']:.3f}")

        fig = plot_confusion_matrix(
            m["cm"],
            title=f"{key} – Confusion Matrix",
            figsize=(3.2, 2.6),  # <- small
            cmap="Blues",
            normalize=False,     # <- no percentages math
            show_percent=False   # <- do not print percentages
        )
        st.pyplot(fig, use_container_width=False)

    st.subheader("🔍 Drill-down: pick a transaction to explain")
    idx = st.number_input("Row index in test set", min_value=0, max_value=len(X_test)-1, value=0, step=1)

    if st.button("Explain this transaction"):
        llm_client = None
        if use_llm and is_ollama_up():
            llm_client = make_ollama_client(model=ollama_model)
        elif use_llm:
            st.warning("⚠️ Ollama not reachable at http://localhost:11434 — using fallback explanation.")

        text, shap_dict, proba_dict, anomaly_dict = explain_transaction(
            idx, X_test, models, feature_names, payload, llm_client=llm_client
        )
        st.code(text)
        st.write("Top features (SHAP, XGBoost):")
        st.table(pd.DataFrame(shap_dict, columns=["feature","shap_value","|shap|"]))
