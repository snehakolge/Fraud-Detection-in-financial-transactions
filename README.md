# 💳 Fraud Detection in Financial Transactions

An end-to-end Machine Learning project that detects fraudulent financial transactions using a trained model, deployed via a Flask API and an interactive Streamlit dashboard.

---

## 🚀 Project Overview

This project simulates a real-world fraud detection system used in banking and fintech. It includes:

* Data preprocessing & feature engineering
* Machine learning model training
* Model deployment using Flask API
* Interactive UI using Streamlit
* End-to-end integration (Frontend ↔ Backend)

---

## 🧠 Features

* ✅ Detects fraudulent transactions in real-time
* ✅ REST API for predictions
* ✅ Interactive dashboard for user input
* ✅ Feature engineering for improved model performance
* ✅ Deployment-ready architecture

---

## 🏗️ Project Structure

```
Fraud_detection/
│
├── flask_api/               # Flask backend API
│   └── app.py
│
├── streamlit_app/           # Streamlit frontend UI
│   └── streamlit_app.py
│
├── model_training/          # Training scripts
│   └── train_model.py
│
├── predictor.py             # Prediction pipeline
│
├── fraud_model.pkl          # Trained model
├── fraud_scaler.pkl         # Feature scaler
├── fraud_le.pkl             # Label encoder
│
├── requirements.txt         # Dependencies
├── Procfile                 # Deployment config
├── runtime.txt              # Python version
│
└── README.md                # Documentation
```

---

## ⚙️ Tech Stack

* Python 🐍
* Scikit-learn
* Pandas & NumPy
* Flask (API)
* Streamlit (Dashboard)
* Gunicorn (Deployment)

---

## 📊 Model Details

* Algorithm: **Random Forest Classifier**
* Feature Engineering:

  * Amount Ratio
  * Balance Errors
  * Full Transfer Indicator
* Evaluation Metrics:

  * Accuracy
  * Precision / Recall
  * ROC-AUC Score

> ⚠️ Note: ROC-AUC = 1.0 may indicate small dataset or data leakage. Further tuning recommended.

---

## 🔌 API Endpoints

### Health Check

```
GET /health
```

Response:

```json
{
  "model": "RandomForestClassifier",
  "status": "ok"
}
```

---

### Predict Fraud

```
POST /predict
```

Sample Request:

```json
{
  "step": 1,
  "type": "PAYMENT",
  "amount": 1000,
  "oldbalanceOrg": 5000,
  "newbalanceOrig": 4000,
  "oldbalanceDest": 0,
  "newbalanceDest": 1000
}
```

Sample Response:

```json
{
  "prediction": "Fraud",
  "probability": 0.54
}
```

---

## 💻 Running Locally

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run Flask API

```
python flask_api/app.py
```

### 3️⃣ Run Streamlit App

```
streamlit run streamlit_app/streamlit_app.py
```

---

## 🌐 Deployment

### Backend (Flask API)

* Deployed using **Render**
* Uses **Gunicorn** as WSGI server

### Frontend (Streamlit)

* Deployed using **Streamlit Cloud**

---

## 🔍 Explainability

The dashboard provides insights into why a transaction is flagged as fraud:

* Sender balance becomes zero
* Receiver had no prior balance
* Suspicious transfer patterns
* High model confidence

---

## 📌 Future Improvements

* 🔥 Add SHAP explainability
* 🤖 Integrate LLM-based explanations
* 📈 Handle class imbalance better
* ☁️ Docker-based deployment
* 📊 Real-time streaming (Kafka)

---

## 👤 Author

**Sneha Kolge**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
