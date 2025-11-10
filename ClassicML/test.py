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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
csv_example = """
PassengerId,Survived,Pclass,Sex,Age,SibSp,Parch,Fare
1,0,3,1,22,1,0,7.25
2,1,1,0,38,1,0,71.28
3,1,3,0,26,0,0,7.92
4,1,1,0,35,1,0,53.10
5,0,3,1,35,0,0,8.05
6,0,3,1,27,0,0,8.46
7,0,1,1,54,0,0,51.86
8,0,3,1,2,3,1,21.08
9,1,3,0,27,0,2,11.13
10,1,2,0,14,1,0,30.07
"""

from io import StringIO
df = pd.read_csv(StringIO(csv_example))

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

X_titanic = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y_titanic = df['Survived'].values.astype(float)

print(f"Data: {X_titanic.shape[0]} samples, {X_titanic.shape[1]} features")


data_titanic = cml.Dataset(X_titanic, y_titanic)

scaler_titanic = cml.StandardScaler(data_titanic)
scaler_titanic.split(ratio=0.7, random=True)
scaler_titanic.standart_normalize()


knn_clf = cml.Knn(data_titanic, num_neighbors=3)
knn_clf.predict("evklid")
knn_clf.loss()
data_titanic.Y_test.print()
data_titanic.info()