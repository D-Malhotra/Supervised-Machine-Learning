"""Loading and preparing the airfoil-noise and diabetes datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

AIRFOIL_TARGET = "Sound Pressure"

DIABETES_TARGET = "diabetes"
DIABETES_NEGATIVE_LABEL = "No diabetes"
DIABETES_ID_COLUMN = "patient_number"
# Stored with a comma as the decimal separator, so pandas reads them as strings.
DIABETES_DECIMAL_COMMA_COLUMNS = ("chol_hdl_ratio", "bmi", "waist_hip_ratio")


def load_airfoil_data():
    """Return the airfoil-noise training and test frames."""
    train = pd.read_csv(DATA_DIR / "airfoil" / "train.csv")
    test = pd.read_csv(DATA_DIR / "airfoil" / "test.csv")

    return train, test


def load_diabetes_data():
    """Return the diabetes training and test frames."""
    train = pd.read_csv(DATA_DIR / "diabetes" / "train.csv")
    test = pd.read_csv(DATA_DIR / "diabetes" / "test.csv")

    return train, test


def prepare_airfoil_xy(df):
    """Split an airfoil frame into a float feature matrix and target vector."""
    y = df[AIRFOIL_TARGET].to_numpy(dtype=float)
    X = df.drop(columns=AIRFOIL_TARGET).to_numpy(dtype=float)

    return X, y


def airfoil_feature_names(df):
    return [c for c in df.columns if c != AIRFOIL_TARGET]


def _to_float(column):
    """Coerce a column that may use ',' as its decimal separator."""
    if column.dtype == object:
        column = column.str.replace(",", ".", regex=False)

    return pd.to_numeric(column)


def prepare_diabetes_xy(df, negative_label=0):
    """Split a diabetes frame into a float feature matrix and label vector.

    `negative_label` is 0 for the tree-based models and -1 for the SVM, whose
    hinge loss needs labels in {-1, +1}.
    """
    features = df.drop(columns=[DIABETES_ID_COLUMN, DIABETES_TARGET])
    for name in DIABETES_DECIMAL_COMMA_COLUMNS:
        features[name] = _to_float(features[name])

    X = features.to_numpy(dtype=float)
    y = np.where(df[DIABETES_TARGET] == DIABETES_NEGATIVE_LABEL, negative_label, 1)

    return X, y.astype(float)


def diabetes_feature_names(df):
    return [c for c in df.columns if c not in (DIABETES_ID_COLUMN, DIABETES_TARGET)]
