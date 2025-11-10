# Agentic Financial Fraud Detector

An interactive **Streamlit web app** that detects and explains financial fraud using a combination of **machine learning**, **anomaly detection**, and **LLM-based interpretability**.

---

## Overview

This project demonstrates an **agentic AI pipeline** for fraud detection that integrates:
- **Supervised models:** `XGBoost`, `Random Forest`
- **Unsupervised models:** `Isolation Forest`, `Autoencoder`
- **Explainability:** SHAP feature attribution + optional natural-language explanations via **Ollama LLM**
- **UI:** Built with Streamlit for interactive exploration and analysis

---

## Key Features

✅ Detects fraudulent transactions from the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
✅ Visualizes confusion matrices and key metrics (Precision, Recall, ROC-AUC, PR-AUC)  
✅ Explains model reasoning using **SHAP** values  
✅ Provides natural-language transaction analysis powered by **Ollama** (e.g. `phi3:mini`)  
✅ Fallback explanations when LLM is offline  
✅ Modular structure — easily extendable to other datasets or models  

---

## 🏗️ Project Structure

