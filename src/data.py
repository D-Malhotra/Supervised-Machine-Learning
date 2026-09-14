from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_airfoil_data():
    train = pd.read_csv(DATA_DIR / "airfoil" / "train.csv")
    test = pd.read_csv(DATA_DIR / "airfoil" / "test.csv")

    return train, test


def load_diabetes_data():
    train = pd.read_csv(DATA_DIR / "diabetes" / "train.csv")
    test = pd.read_csv(DATA_DIR / "diabetes" / "test.csv")

    return train, test