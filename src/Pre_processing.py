import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from collections import defaultdict


def standardise(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_std = (X - mu) / sigma 
    return X_std, mu, sigma