import classicml as cml
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, 
    load_breast_cancer, 
    fetch_california_housing,
    load_diabetes,
    load_wine,
    make_blobs
)
from sklearn.model_selection import train_test_split
# ========== 1. IRIS DATASET - Logistic Regression (Multiclass) ==========
print("\n### 1. IRIS DATASET - Logistic Regression (Multiclass) ###")
print("-" * 80)

iris = load_iris()
X = iris.data
y = iris.target

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

y_ohe = np.eye(3)[y]
print(y_ohe)
data = cml.Dataset(X, y)
data.Y = cml.one_hot_encoder(data.Y)
data.Y.print()