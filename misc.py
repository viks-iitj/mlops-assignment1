"""
Contains:
- load_data(): downloads and constructs the DataFrame for Boston housing
- split_and_scale(): splits into train/test and scales features using StandardScaler
- fit_model(): trains any scikit-learn regressor passed to it
- compute_mse(): returns mean squared error on test data
- save_artifact(): saves model or scaler using joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
from typing import Tuple, Any

def load_data() -> pd.DataFrame:
    """
    Load the Boston Housing dataset from the public CMU repository and return a DataFrame.
    Columns: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, scale: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Split features and target into train and test sets. Optionally apply standard scaling.
    Returns: X_train, X_test, y_train, y_test, scaler_or_none
    """
    X = df.drop(columns=['MEDV']).values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def fit_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit the provided sklearn-like model on training data and return the fitted model.
    """
    model.fit(X_train, y_train)
    return model

def compute_mse(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Compute Mean Squared Error on the test set.
    """
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def save_artifact(obj, path: str):
    """
    Save model or scaler to disk using joblib.
    """
    joblib.dump(obj, path)

