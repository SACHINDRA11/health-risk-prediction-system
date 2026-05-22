# import pandas as pd
# import numpy as np
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

# # ===============================
# # LOAD DATA
# # ===============================
# df = pd.read_csv("icu_data_new.csv")

# print("Columns:\n", df.columns)

# df.drop(columns=["RecordID"], inplace=True, errors="ignore")

# # ===============================
# # TARGET
# # ===============================
# target = "In-hospital_death"

# X = df.drop(columns=[target])
# y = df[target]

# print("\nClass Distribution BEFORE:\n", y.value_counts())

# # ===============================
# # HANDLE MISSING VALUES
# # ===============================
# X = X.dropna(thresh=int(0.6 * X.shape[1]))
# y = y.loc[X.index]

# X = X.fillna(X.median(numeric_only=True))
# X = X.fillna("missing")

# # ===============================
# # ENCODING
# # ===============================
# X = pd.get_dummies(X, drop_first=True)

# feature_names = X.columns

# # ===============================
# # SPLIT
# # ===============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # ===============================
# # SMOTE
# # ===============================
# sm = SMOTE(random_state=42)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# print("\nClass Distribution AFTER SMOTE:\n", pd.Series(y_train_res).value_counts())

# # ===============================
# # 🔥 MODEL COMPARISON (ADD THIS)
# # ===============================
# print("\n🔥 MODEL COMPARISON:\n")

# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=200),
#     "XGBoost": XGBClassifier(
#         n_estimators=200,
#         learning_rate=0.05,
#         max_depth=4,
#         eval_metric='logloss'
#     )
# }

# for name, m in models.items():
#     m.fit(X_train_res, y_train_res)
#     y_prob_temp = m.predict_proba(X_test)[:, 1]
#     y_pred_temp = (y_prob_temp > 0.3).astype(int)

#     print(f"\n{name}")
#     print("ROC AUC:", round(roc_auc_score(y_test, y_prob_temp), 4))
#     print(classification_report(y_test, y_pred_temp))

# # ===============================
# # 🔥 FINAL MODEL (YOUR ORIGINAL)
# # ===============================
# model = XGBClassifier(
#     n_estimators=400,
#     learning_rate=0.03,
#     max_depth=5,
#     scale_pos_weight=1.5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric='logloss'
# )

# # ===============================
# # TRAIN
# # ===============================
# model.fit(X_train_res, y_train_res)

# # ===============================
# # PREDICT
# # ===============================
# y_prob = model.predict_proba(X_test)[:, 1]

# # ===============================
# # THRESHOLD
# # ===============================
# best_threshold = 0.3

# print("\n🔥 Using Threshold:", best_threshold)

# y_pred = (y_prob > best_threshold).astype(int)

# # ===============================
# # FINAL EVALUATION
# # ===============================
# print("\n🔥 FINAL MODEL (XGBOOST):\n")

# print("Classification Report:\n")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:\n")
# print(confusion_matrix(y_test, y_pred))

# print("\nROC AUC:", roc_auc_score(y_test, y_prob))

# # ===============================
# # SAVE
# # ===============================
# joblib.dump(model, "icu_model.joblib")
# joblib.dump(best_threshold, "threshold.joblib")
# joblib.dump(feature_names, "feature_names.joblib")

# print("\n🔥 ICU model + threshold + features saved successfully!")




import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("icu_data_new.csv")
df.drop(columns=["RecordID"], inplace=True, errors="ignore")

target = "In-hospital_death"

X = df.drop(columns=[target])
y = df[target]

# Missing values
X = X.dropna(thresh=int(0.6 * X.shape[1]))
y = y.loc[X.index]

X = X.fillna(X.median(numeric_only=True))
X = X.fillna("missing")

# One-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)
feature_names = X_encoded.columns.tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Balance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# =========================
# MODEL COMPARISON
# =========================
print("\n===== MODEL COMPARISON =====\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2,
        random_state=42,
        eval_metric="logloss"
    )
}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.20).astype(int)

    print(f"\n{name}")
    print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
    print(classification_report(y_test, preds))

# =========================
# FINAL MODEL
# =========================
final_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=3,
    random_state=42,
    eval_metric="logloss"
)

final_model.fit(X_train_res, y_train_res)

probs = final_model.predict_proba(X_test)[:, 1]

# More sensitive for medical use
best_threshold = 0.08
preds = (probs > best_threshold).astype(int)

print("\n===== FINAL XGBOOST =====")
print("Threshold:", best_threshold)
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

joblib.dump(final_model, "icu_model.joblib")
joblib.dump(feature_names, "feature_names.joblib")

print("\nSaved icu_model.joblib and feature_names.joblib")