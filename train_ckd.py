# train_ckd.py
"""
CKD training script (updated).
Place 'ckd_data.csv' in the same folder and run:
    python train_ckd.py

This script:
 - loads and cleans the CSV (smart numeric coercion)
 - builds a preprocessor that imputes/scales numeric cols and imputes+one-hot encodes categorical cols
 - trains LogisticRegression, DecisionTree, RandomForest
 - prints metrics and saves RandomForest pipeline as ckd_model_rf.joblib
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

CKD_CSV_PATH = "ckd_data.csv"   # <-- put your dataset here (same folder)
RANDOM_STATE = 42

# ------------------ load & clean ------------------
def load_and_clean(path):
    """
    Load CSV, replace common missing markers, try numeric coercion columnwise,
    drop very-empty cols and return cleaned df + detected target column.
    """
    df = pd.read_csv(path)
    # common missing markers
    df.replace(['?','na','NA','', ' '], np.nan, inplace=True)

    # Try to coerce columns to numeric where possible (keep original if mostly non-numeric)
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        non_na = coerced.notna().sum()
        frac = non_na / len(df)
        # If >=50% of values can be numeric, convert to numeric
        if frac >= 0.5:
            df[col] = coerced

    # find target column (common names)
    target_candidates = [c for c in df.columns if c.lower() in ('class','ckd','status','label','target','classification')]
    if target_candidates:
        target_col = target_candidates[0]
    else:
        target_col = df.columns[-1]

    print("Columns found:", list(df.columns))
    print("Detected target column:", target_col)

    # If target is string, map to 0/1
    if df[target_col].dtype == object or df[target_col].dtype.name == 'category':
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
        uniq = [u for u in pd.unique(df[target_col]) if pd.notna(u)]
        print("Unique target values (sample):", uniq[:10])
        # common mapping for CKD dataset
        if set(df[target_col].unique()).issuperset({'ckd','notckd'}):
            mapping = {'ckd':1, 'notckd':0, 'not ckd':0}
        else:
            # fallback mapping: first two unique non-null values -> 0/1
            mapping = {}
            if len(uniq) >= 2:
                mapping = {uniq[0]:0, uniq[1]:1}
        if mapping:
            df[target_col] = df[target_col].map(mapping)
        else:
            # if mapping couldn't be built, try numeric coercion
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    # drop columns with >90% missing
    miss_frac = df.isna().mean()
    to_drop = miss_frac[miss_frac > 0.9].index.tolist()
    if to_drop:
        print("Dropping columns with >90% missing:", to_drop)
        df = df.drop(columns=to_drop)

    # drop any columns that became all-NaN
    df = df.dropna(axis=1, how='all')

    return df, target_col

# ------------------ preprocessor ------------------
def build_preprocessor(X):
    """
    Smartly infer numeric vs categorical based on dtype and content.
    Returns a ColumnTransformer that imputes+scales numeric,
    imputes+onehot encodes categorical.
    """
    # numeric columns: dtype numeric
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # categorical candidate columns
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    # debug prints (useful if mapping mismatch)
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # OneHotEncoder with handle_unknown to avoid errors at predict time
    cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ], remainder='drop')

    return preprocessor

# ------------------ training ------------------
def train_models(df, target_col):
    # drop id-like if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    preprocessor = build_preprocessor(X)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)
    }

    # train-test split (stratify if labels available)
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_arg)

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        roc = None
        try:
            roc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
        except Exception:
            pass
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC AUC:", roc)
        results[name] = {'pipeline': pipe, 'roc': roc}

    # Save RandomForest pipeline as default best model
    if 'RandomForest' in results:
        joblib.dump(results['RandomForest']['pipeline'], "ckd_model_rf.joblib")
        print("\nSaved model: ckd_model_rf.joblib")
    else:
        first_pipe = list(results.values())[0]['pipeline']
        joblib.dump(first_pipe, "ckd_model.joblib")
        print("\nSaved model: ckd_model.joblib")

    return results

# ------------------ main ------------------
if __name__ == "__main__":
    print("Loading dataset from:", CKD_CSV_PATH)
    df, target = load_and_clean(CKD_CSV_PATH)
    print("Dataset shape after cleaning:", df.shape)
    results = train_models(df, target)
    print("\nTraining finished.")