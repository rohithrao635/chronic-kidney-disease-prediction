# ===============================
# Chronic Kidney Disease Trainer
# ===============================

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

# ---- Load dataset ----
df = pd.read_csv("Kidney_disease.csv")   # keep csv in same folder

print("Original Dataset Shape:", df.shape)

# remove id column if exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# remove spaces
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()

# -----------------------------
# TARGET VARIABLE CONVERSION
# -----------------------------
# Use map and explicit type conversion to avoid pandas downcasting warnings.
df['classification'] = df['classification'].map({
    'ckd': 1, 'ckd\t': 1,
    'notckd': 0, 'notckd\t': 0
}).astype('int64')

# -----------------------------
# CONVERT YES/NO FEATURES
# -----------------------------
binary_map = {
    'yes':1, 'no':0,
    'present':1, 'notpresent':0,
    'abnormal':1, 'normal':0,
    'poor':0, 'good':1
}

for col in df.columns:
    df[col] = df[col].replace(binary_map)

# convert all numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
for col in df.columns:
    if col != 'classification':
        df[col] = df[col].fillna(df[col].median())

# -----------------------------
# SELECT IMPORTANT FEATURES
# -----------------------------
features = ['sg','hemo','al','rc','htn','dm','appet','pc']
X = df[features]
y = df['classification']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\nFINAL MODEL EVALUATION")
print("-"*40)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "kidney_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as kidney_model.pkl")
print("Scaler saved as scaler.pkl")
