# 🏦 Bank GoodCredit — Credit Risk Scorecard (PR-0015)

End-to-end ML web application for credit risk prediction, built from the Bank GoodCredit notebook.

## Tech Stack
- **ML**: Python · Scikit-learn (GradientBoostingClassifier + Ensemble)
- **Backend**: Flask REST API
- **Frontend**: HTML5 · CSS3 · Vanilla JS

## Model Performance
| Model | Gini |
|---|---|
| Logistic Regression | 32.03 |
| Gradient Boosting | **32.75** ✓ Best |
| Random Forest | 32.68 |
| Ensemble (GB + RF) | 34.55 |
| 📌 Benchmark | 37.90 |

## Pipeline (follows notebook exactly)

1. **Load** 3 tables: `account.csv`, `enquiry.csv`, `demographics.csv`
2. **Parse dates** (`opened_dt`, `last_paymt_dt`, `enquiry_dt`, …)
3. **Account feature engineering**
   - Parse `paymenthistory1/2` → DPD codes per month
   - Bucket into 0-29, 30-59, 60-89, 90+ DPD counts
   - Aggregate per customer: max DPD, months to first 30+, utilisation ratios
4. **Enquiry feature engineering**
   - Recency windows: 90-day, 365-day enquiry counts
   - Total/mean/max enquiry amounts, unique purposes
5. **Master matrix**: left-join demographics ← account ← enquiry (23,896 customers × 111 features)
6. **Preprocessing**: drop >95% missing cols, label-encode, median impute
7. **Train/test split** 80/20 stratified (4.2% bad rate, highly imbalanced)
8. **Resampling** minority class oversampling (22.8× imbalance)
9. **Models**: Logistic Regression → GradientBoosting → Random Forest → Ensemble
10. **Evaluation**: Gini, AUC, decile rank-ordering table

## Project Structure
```
credit_fraud_app/
├── data/
│   ├── account.csv
│   ├── enquiry.csv
│   └── demographics.csv
├── model/
│   ├── model.pkl           ← Trained GradientBoosting model
│   ├── imputer.pkl         ← SimpleImputer (median)
│   ├── label_encoders.pkl  ← Per-column LabelEncoders
│   └── metadata.json       ← Features, metrics, rank table
├── templates/index.html
├── static/css/style.css
├── static/js/app.js
├── train_model.py          ← Full training pipeline
├── app.py                  ← Flask app
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
pip install -r requirements.txt
python train_model.py   # (skip if model/ already exists)
python app.py
# Open http://localhost:5000
```

## REST API

### `POST /api/predict`
```json
{
  "num_accounts": 3,
  "max_dpd_max": 0,
  "cnt_30_59_sum": 0,
  "cnt_90_plus_sum": 0,
  "cur_balance_amt_sum": 50000,
  "creditlimit_sum": 200000,
  "num_enquiries": 5,
  "count_enquiry_recency_90": 1,
  "feature_7": 0
}
```
**Response:**
```json
{
  "success": true,
  "prob_bad": 0.038,
  "bad_pct": 3.80,
  "credit_score": 962,
  "risk_tier": "Medium",
  "model_gini": 32.75
}
```

### `GET /api/metrics`  →  full model metadata + rank table
### `GET /api/health`   →  `{ "status": "ok" }`
