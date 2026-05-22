# train_ckd.py

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================================
# SETTINGS
# =========================================
CKD_CSV_PATH = "ckd_data.csv"
RANDOM_STATE = 42

# =========================================
# LOAD + CLEAN DATA
# =========================================
def load_and_clean(path):

    df = pd.read_csv(path)

    # replace missing markers
    df.replace(
        ['?', 'na', 'NA', '', ' '],
        np.nan,
        inplace=True
    )

    # numeric conversion
    for col in df.columns:

        converted = pd.to_numeric(
            df[col],
            errors='coerce'
        )

        success_ratio = converted.notna().mean()

        if success_ratio >= 0.5:
            df[col] = converted

    # target detection
    target_candidates = [
        c for c in df.columns
        if c.lower() in [
            'class',
            'classification',
            'target',
            'label',
            'ckd'
        ]
    ]

    if target_candidates:
        target_col = target_candidates[0]
    else:
        target_col = df.columns[-1]

    print("Detected target:", target_col)

    # target mapping
    if df[target_col].dtype == object:

        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapping = {
            'ckd': 1,
            'notckd': 0,
            'not ckd': 0
        }

        df[target_col] = df[target_col].map(mapping)

    # remove bad columns
    missing_ratio = df.isna().mean()

    drop_cols = (
        missing_ratio[missing_ratio > 0.9]
        .index
        .tolist()
    )

    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # remove fully null columns
    df.dropna(axis=1, how='all', inplace=True)

    return df, target_col

# =========================================
# PREPROCESSOR
# =========================================
def build_preprocessor(X):

    num_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    # numeric pipeline
    num_transformer = Pipeline([
        (
            'imputer',
            SimpleImputer(strategy='median')
        ),
        (
            'scaler',
            StandardScaler()
        )
    ])

    # categorical pipeline
    cat_transformer = Pipeline([
        (
            'imputer',
            SimpleImputer(strategy='most_frequent')
        ),
        (
            'onehot',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )
        )
    ])

    # final preprocessor
    preprocessor = ColumnTransformer([
        (
            'num',
            num_transformer,
            num_cols
        ),
        (
            'cat',
            cat_transformer,
            cat_cols
        )
    ])

    return preprocessor

# =========================================
# TRAIN MODEL
# =========================================
def train_model(df, target_col):

    # remove id column if exists
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    X = df.drop(columns=[target_col])

    y = df[target_col].astype(int)

    # preprocessing
    preprocessor = build_preprocessor(X)

    # model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    )

    # pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining model...\n")

    # train
    pipeline.fit(X_train, y_train)

    # predictions
    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # metrics
    print(classification_report(y_test, y_pred))

    print(
        "\nConfusion Matrix:\n",
        confusion_matrix(y_test, y_pred)
    )

    print(
        "\nROC AUC:",
        roc_auc_score(y_test, y_prob)
    )

    # save model
    joblib.dump(
        pipeline,
        "ckd_model_rf.joblib"
    )

    print("\n✅ Model Saved Successfully")

# =========================================
# MAIN
# =========================================
if __name__ == "__main__":

    print("Loading dataset...\n")

    df, target_col = load_and_clean(
        CKD_CSV_PATH
    )

    print(
        "\nDataset Shape:",
        df.shape
    )

    train_model(
        df,
        target_col
    )

    print("\n🔥 Training Completed")