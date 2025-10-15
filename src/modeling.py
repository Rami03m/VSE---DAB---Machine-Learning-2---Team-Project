from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

@dataclass
class TrainResult:
    model: Pipeline
    accuracy: float
    auc: float

def _make_pipeline(cat_cols) -> Pipeline:
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ], remainder='drop')
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    return pipe

def train_and_eval(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> TrainResult:
    pipe = _make_pipeline(cat_cols=X_train.columns.tolist())
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    proba = pipe.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, proba)
    return TrainResult(model=pipe, accuracy=acc, auc=auc)