"""
Bank GoodCredit — Credit Risk Scorecard Training Pipeline
Follows the notebook: Bank_Good_Credit_client_project.ipynb
Target: Bad_label (1 = 30+ DPD, 0 = Good)  |  Metric: Gini (= 2*AUC - 1)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  BANK GOODCREDIT — CREDIT RISK SCORECARD TRAINING")
print("=" * 60)

# ── 1. Load raw tables ────────────────────────────────────────────────────────
print("\n[1/8] Loading data …")
account      = pd.read_csv(f"{DATA_DIR}/account.csv",      low_memory=False)
enquiry      = pd.read_csv(f"{DATA_DIR}/enquiry.csv",      low_memory=False)
demographics = pd.read_csv(f"{DATA_DIR}/demographics.csv", low_memory=False)

print(f"  account      : {account.shape}")
print(f"  enquiry      : {enquiry.shape}")
print(f"  demographics : {demographics.shape}")
print(f"  Bad rate     : {demographics['Bad_label'].mean()*100:.2f}%  (highly imbalanced)")

# ── 2. Parse dates ────────────────────────────────────────────────────────────
print("\n[2/8] Parsing dates …")
for c in ['opened_dt', 'last_paymt_dt', 'closed_dt', 'reporting_dt', 'dt_opened']:
    account[c] = pd.to_datetime(account[c], format='%d-%b-%y', errors='coerce')

for c in ['enquiry_dt', 'dt_opened']:
    enquiry[c] = pd.to_datetime(enquiry[c], format='%d-%b-%y', errors='coerce')

account['snapshot_dt'] = account['dt_opened']
enquiry['snapshot_dt'] = enquiry['dt_opened']

# ── 3. Feature Engineering — Account table ────────────────────────────────────
print("\n[3/8] Engineering account features (payment-history parsing) …")

DPD_CODE_MAP = {'XXX': np.nan, 'STD': 0, 'SMA': 1, 'SUB': 61, 'DBT': 91, 'LSS': 121}

def parse_payment_history(raw):
    if pd.isna(raw):
        return []
    s = str(raw).replace('"', '').strip()
    chunks = [s[i:i+3] for i in range(0, len(s) - len(s) % 3, 3)]
    dpd = []
    for ch in chunks:
        if ch in DPD_CODE_MAP:
            dpd.append(DPD_CODE_MAP[ch])
        else:
            try:
                dpd.append(float(ch))
            except ValueError:
                dpd.append(np.nan)
    return dpd

account['paymenthistory_full'] = (
    account['paymenthistory1'].fillna('').astype(str).str.replace('"', '') +
    account['paymenthistory2'].fillna('').astype(str).str.replace('"', '')
)
account['dpd_series'] = account['paymenthistory_full'].apply(parse_payment_history)
account['payment_history_length'] = account['dpd_series'].apply(len)

def dpd_bucket_stats(dpd_list):
    arr = np.array([v for v in dpd_list if not pd.isna(v)])
    if len(arr) == 0:
        return pd.Series({'n_months_reported': 0, 'cnt_0_29': np.nan,
                          'cnt_30_59': np.nan, 'cnt_60_89': np.nan,
                          'cnt_90_plus': np.nan, 'max_dpd': np.nan,
                          'months_to_first_30_plus': np.nan})
    idx_30 = np.where(arr >= 30)[0]
    return pd.Series({
        'n_months_reported':      len(arr),
        'cnt_0_29':               int(np.sum((arr >= 0) & (arr <= 29))),
        'cnt_30_59':              int(np.sum((arr >= 30) & (arr <= 59))),
        'cnt_60_89':              int(np.sum((arr >= 60) & (arr <= 89))),
        'cnt_90_plus':            int(np.sum(arr >= 90)),
        'max_dpd':                float(arr.max()),
        'months_to_first_30_plus': float(idx_30[0]) if len(idx_30) > 0 else np.nan,
    })

acc_dpd = account['dpd_series'].apply(dpd_bucket_stats)
account = pd.concat([account, acc_dpd], axis=1)

account['diff_lastpaymt_opened_dt'] = (account['last_paymt_dt'] - account['opened_dt']).dt.days
account['diff_open_to_snapshot']    = (account['snapshot_dt']  - account['opened_dt']).dt.days
account['is_30_plus_ever']          = (account['max_dpd'] >= 30).astype(float)

agg_funcs = {
    'cnt_0_29':                 ['mean', 'sum'],
    'cnt_30_59':                ['mean', 'sum'],
    'cnt_60_89':                ['mean', 'sum'],
    'cnt_90_plus':              ['mean', 'sum'],
    'max_dpd':                  ['max', 'mean'],
    'months_to_first_30_plus':  ['min'],
    'payment_history_length':   ['mean'],
    'diff_lastpaymt_opened_dt': ['mean', 'sum'],
    'high_credit_amt':          ['sum', 'mean'],
    'cur_balance_amt':          ['sum', 'mean'],
    'amt_past_due':             ['sum', 'mean'],
    'creditlimit':              ['sum', 'mean'],
    'cashlimit':                ['sum', 'mean'],
    'is_30_plus_ever':          ['sum', 'mean'],
    'customer_no':              ['count'],
}

acc_features = account.groupby('customer_no').agg(agg_funcs)
acc_features.columns = ['_'.join(c).strip() for c in acc_features.columns]
acc_features = acc_features.rename(columns={'customer_no_count': 'num_accounts'})

# Utilisation ratios (from notebook)
acc_features['ratio_currbalance_creditlimit'] = (
    acc_features['cur_balance_amt_sum'] /
    acc_features['creditlimit_sum'].replace(0, np.nan)
)
acc_features['utilisation_trend'] = (
    acc_features['ratio_currbalance_creditlimit'] /
    (acc_features['cur_balance_amt_mean'] /
     (acc_features['creditlimit_mean'] + acc_features['cashlimit_mean']).replace(0, np.nan))
)
print(f"  Account feature matrix: {acc_features.shape}")

# ── 4. Feature Engineering — Enquiry table ────────────────────────────────────
print("\n[4/8] Engineering enquiry features …")
enquiry['days_since_enquiry']    = (enquiry['snapshot_dt'] - enquiry['enquiry_dt']).dt.days
enquiry['diff_open_enquiry_dt']  = (enquiry['dt_opened'] - enquiry['enquiry_dt']).dt.days
enquiry['enq_recency_90']        = (enquiry['days_since_enquiry'] <= 90).astype(int)
enquiry['enq_recency_365']       = (enquiry['days_since_enquiry'] <= 365).astype(int)

enq_features = enquiry.groupby('customer_no').agg(
    num_enquiries             = ('enq_amt',               'count'),
    total_enq_amt             = ('enq_amt',               'sum'),
    mean_enq_amt              = ('enq_amt',               'mean'),
    max_enq_amt               = ('enq_amt',               'max'),
    count_enquiry_recency_90  = ('enq_recency_90',        'sum'),
    count_enquiry_recency_365 = ('enq_recency_365',       'sum'),
    mean_diff_open_enquiry_dt = ('diff_open_enquiry_dt',  'mean'),
    n_unique_purpose          = ('enq_purpose',           'nunique'),
)

max_freq_purpose = (
    enquiry.groupby(['customer_no', 'enq_purpose']).size()
    .reset_index(name='cnt').sort_values('cnt', ascending=False)
    .drop_duplicates('customer_no').set_index('customer_no')['enq_purpose']
    .rename('max_freq_enquiry')
)
enq_features = enq_features.join(max_freq_purpose)

# Encode the categorical enquiry purpose
le_enq = LabelEncoder()
enq_features['max_freq_enquiry'] = le_enq.fit_transform(
    enq_features['max_freq_enquiry'].fillna('Unknown').astype(str)
)
print(f"  Enquiry feature matrix: {enq_features.shape}")

# ── 5. Master Feature Matrix ──────────────────────────────────────────────────
print("\n[5/8] Building master feature matrix …")

# Drop near-empty demographic columns (>95% missing)
missing_pct = demographics.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.95].index.tolist()
demographics_clean = demographics.drop(columns=cols_to_drop)

master = (
    demographics_clean
    .merge(acc_features.reset_index(), on='customer_no', how='left')
    .merge(enq_features.reset_index(), on='customer_no', how='left')
)
print(f"  Master shape: {master.shape}")

# ── 6. Prepare X, y ──────────────────────────────────────────────────────────
print("\n[6/8] Preprocessing features …")

DROP_COLS = ['customer_no', 'dt_opened', 'entry_time', 'Bad_label']
TARGET    = 'Bad_label'

y = master[TARGET].copy()
X = master.drop(columns=DROP_COLS, errors='ignore')

# Encode any remaining object columns
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna('Unknown').astype(str))
    label_encoders[col] = le

# Impute
imputer = SimpleImputer(strategy='median')
X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"  Features used: {X_imp.shape[1]}")
print(f"  Bad rate: {y.mean()*100:.2f}%")

# ── 7. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# ── 8. Model building ─────────────────────────────────────────────────────────
print("\n[7/8] Training models …")

# scale_pos_weight equivalent for GBR — handled via class_weight in LR
pos_w = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Class weight ratio (neg/pos): {pos_w:.1f}x")

# Logistic Regression baseline
print("  → Logistic Regression …")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE, C=0.1)
lr.fit(X_train, y_train)
lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
lr_gini = (2 * lr_auc - 1) * 100
print(f"    LR   AUC={lr_auc:.4f}  Gini={lr_gini:.2f}")

# Gradient Boosting (XGBoost-equivalent available in sklearn)
print("  → GradientBoostingClassifier (XGBoost-style) …")
gb = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=RANDOM_STATE
)
# Oversample minority to compensate for imbalance
from sklearn.utils import resample
X_maj = X_train[y_train == 0]
X_min = X_train[y_train == 1]
y_maj = y_train[y_train == 0]
y_min = y_train[y_train == 1]
X_min_up, y_min_up = resample(X_min, y_min, replace=True,
                               n_samples=len(X_maj)//3, random_state=RANDOM_STATE)
X_bal = pd.concat([X_maj, X_min_up])
y_bal = pd.concat([y_maj, y_min_up])

gb.fit(X_bal, y_bal)
gb_auc  = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
gb_gini = (2 * gb_auc - 1) * 100
print(f"    GB   AUC={gb_auc:.4f}  Gini={gb_gini:.2f}")

# Random Forest
print("  → RandomForestClassifier …")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight='balanced',
    min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
rf_gini = (2 * rf_auc - 1) * 100
print(f"    RF   AUC={rf_auc:.4f}  Gini={rf_gini:.2f}")

# Ensemble (average probabilities — notebook approach)
print("  → Ensemble (GB + RF) …")
ens_proba = 0.6 * gb.predict_proba(X_test)[:, 1] + 0.4 * rf.predict_proba(X_test)[:, 1]
ens_auc   = roc_auc_score(y_test, ens_proba)
ens_gini  = (2 * ens_auc - 1) * 100
print(f"    ENS  AUC={ens_auc:.4f}  Gini={ens_gini:.2f}")

# Pick best model
model_scores = {
    'LogisticRegression': (lr, lr_auc, lr_gini),
    'GradientBoosting':   (gb, gb_auc, gb_gini),
    'RandomForest':       (rf, rf_auc, rf_gini),
}
best_name = max(model_scores, key=lambda k: model_scores[k][1])
best_model, best_auc, best_gini = model_scores[best_name]
print(f"\n  ✓ Best single model: {best_name}  Gini={best_gini:.2f}")
print(f"  Benchmark Gini: 37.90  |  Improvement: {best_gini - 37.9:+.2f}")

# Rank-ordering decile table
final_proba = best_model.predict_proba(X_test)[:, 1]
tmp = pd.DataFrame({'y': y_test.values, 'score': final_proba})
tmp['decile'] = pd.qcut(tmp['score'], 10, labels=False, duplicates='drop') + 1
rank_table = tmp.groupby('decile').agg(
    n_customers=('y', 'count'), n_bad=('y', 'sum')
)
rank_table['bad_rate'] = (rank_table['n_bad'] / rank_table['n_customers'] * 100).round(2)
print("\n  Rank-ordering decile table:")
print(rank_table.to_string())

# Feature importance
fi_series = pd.Series(best_model.feature_importances_, index=X_train.columns)
top_features = fi_series.nlargest(20).reset_index()
top_features.columns = ['feature', 'importance']

# ── 9. Save artifacts ─────────────────────────────────────────────────────────
print("\n[8/8] Saving artifacts …")
joblib.dump(best_model, f"{MODEL_DIR}/model.pkl")
joblib.dump(imputer,    f"{MODEL_DIR}/imputer.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")

metadata = {
    "best_model_name": best_name,
    "feature_names": list(X_train.columns),
    "metrics": {
        "auc":  round(best_auc,  4),
        "gini": round(best_gini, 2),
        "benchmark_gini": 37.90,
        "improvement": round(best_gini - 37.9, 2),
        "lr_gini":  round(lr_gini,  2),
        "gb_gini":  round(gb_gini,  2),
        "rf_gini":  round(rf_gini,  2),
        "ens_gini": round(ens_gini, 2),
    },
    "class_stats": {
        "total_customers": int(len(y)),
        "n_bad":  int(y.sum()),
        "n_good": int((y == 0).sum()),
        "bad_rate_pct": round(y.mean() * 100, 2),
    },
    "rank_table": rank_table.reset_index().to_dict(orient="records"),
    "top_features": top_features.to_dict(orient="records"),
    "all_models_gini": {
        "Logistic Regression": round(lr_gini, 2),
        "Gradient Boosting":   round(gb_gini, 2),
        "Random Forest":       round(rf_gini, 2),
        "Ensemble":            round(ens_gini, 2),
    }
}

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Saved to: {MODEL_DIR}/")
print("\n" + "="*60)
print(f"  FINAL GINI  : {best_gini:.2f}")
print(f"  BENCHMARK   : 37.90")
print(f"  IMPROVEMENT : {best_gini - 37.9:+.2f} points")
print("="*60)
print("Training complete ✓")
