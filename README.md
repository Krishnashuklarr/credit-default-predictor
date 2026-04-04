# 💳 Credit Card Default Predictor

A machine learning web app that predicts whether a credit card holder will default on their payment next month — built with **LightGBM**, **scikit-learn**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square&logo=scikit-learn)

---

## 📁 Project Structure

```
credit_default/
├── data/
│   └── UCI_Credit_Card.csv      ← raw dataset (add manually)
├── models/                       ← auto-created by train.py
│   ├── preprocessor.pkl
│   ├── lgb_model.pkl
│   └── best_threshold.pkl
├── outputs/
│   ├── eda/                      ← EDA plots
│   └── predictions.csv           ← prediction results
├── scripts/
│   ├── eda.py                    ← exploratory data analysis
│   ├── train.py                  ← full training pipeline
│   └── predict.py                ← make predictions on new data
└── src/
    ├── config.py                 ← all paths & constants
    ├── feature_engineering.py    ← feature creation (shared by train & predict)
    └── preprocessing.py          ← sklearn ColumnTransformer helpers
```

---

## 🗃️ Dataset

**UCI Credit Card Default Dataset**
- 30,000 customer records from a Taiwan bank
- Target: `default.payment.next.month` (1 = default, 0 = no default)

Download from: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
Place the CSV inside the `data/` folder as `UCI_Credit_Card.csv`.

---

## ⚙️ Setup
```bash
git clone https://github.com/YOUR_USERNAME/credit-default-predictor.git
cd credit-default-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage
```bash
python scripts/eda.py        # EDA plots
python scripts/train.py      # Train model
streamlit run app.py         # Launch web app
```

---

## 🧠 Model Details

| Item | Detail |
|---|---|
| Algorithm | LightGBM |
| Class imbalance | `class_weight='balanced'` |
| Threshold | F1-optimal |
| ROC-AUC | ~0.77 |
| Accuracy | ~79% |

---

## 🔮 Future Improvements

- [ ] SHAP explainability
- [ ] Hyperparameter tuning with Optuna
- [ ] Deploy to Streamlit Cloud
- [ ] Batch prediction via CSV upload

---

## 👤 Author

Built by **Krish**