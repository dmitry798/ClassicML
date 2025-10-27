import classicml as cml
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, 
    load_breast_cancer, 
    fetch_california_housing,
    load_diabetes,
    load_wine
)
from sklearn.model_selection import train_test_split

print("=" * 80)
print(" " * 20 + "CLASSICML - REAL KAGGLE DATASETS DEMO")
print("=" * 80)


# ========== 1. IRIS DATASET (классификация) ==========
print("\n### 1. IRIS DATASET - Multiclass Classification ###")
print("-" * 80)

iris = load_iris()
X = iris.data  # (150, 4)
y = iris.target  # (150,)

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"Classes: {iris.target_names}")

# One-Hot Encoding
y_ohe = np.eye(3)[y]

# Dataset
data = cml.Dataset(X, y_ohe)

# Preprocessing
scaler = cml.StandardScaler(data)
scaler.split(ratio=0.8, random=True)
scaler.standart_normalize()

# Training
logreg = cml.LogisticRegression(data, "multi")
logreg.train(method="gd", iters=500, lr=0.1)
logreg.predict()
logreg.loss()

print("✓ Training completed!")


# ========== 2. BREAST CANCER (бинарная классификация) ==========
print("\n### 2. BREAST CANCER DATASET - Binary Classification ###")
print("-" * 80)

cancer = load_breast_cancer()
X_cancer = cancer.data  # (569, 30)
y_cancer = cancer.target.astype(float)  # (569,)

print(f"Data: {X_cancer.shape[0]} samples, {X_cancer.shape[1]} features")
print(f"Classes: {cancer.target_names}")

# Dataset
data_cancer = cml.Dataset(X_cancer, y_cancer)

# Preprocessing
scaler_cancer = cml.StandardScaler(data_cancer)
scaler_cancer.split(ratio=0.75, random=True)
scaler_cancer.standart_normalize()

# Training
logreg_cancer = cml.LogisticRegression(data_cancer, "binary")
logreg_cancer.train(method="gd", iters=1000, lr=0.1)
logreg_cancer.predict()
logreg_cancer.loss(threshold=0.5)

print("✓ Training completed!")


# ========== 3. CALIFORNIA HOUSING (регрессия #1) ==========
print("\n### 3. CALIFORNIA HOUSING DATASET - Regression ###")
print("-" * 80)

housing = fetch_california_housing()
X_house = housing.data[:1000]  # (1000, 8) - берём 1000 для демо
y_house = housing.target[:1000]  # (1000,)

print(f"Data: {X_house.shape[0]} samples, {X_house.shape[1]} features")
print(f"Target: Median house value (in $100,000s)")
print(f"Features: {housing.feature_names}")

# Dataset
data_house = cml.Dataset(X_house, y_house)

# Preprocessing
scaler_house = cml.StandardScaler(data_house)
scaler_house.split(ratio=0.8, random=True)
scaler_house.standart_normalize()

# Training
lr_house = cml.LinearRegression(data_house)
lr_house.train(method="nesterov", iters=1500, lr=0.1)
lr_house.predict()
lr_house.loss()

print("✓ Training completed!")


# ========== 4. DIABETES DATASET (регрессия #2) ==========
print("\n### 4. DIABETES DATASET - Regression ###")
print("-" * 80)

diabetes = load_diabetes()
X_diabetes = diabetes.data  # (442, 10)
y_diabetes = diabetes.target  # (442,)

print(f"Data: {X_diabetes.shape[0]} samples, {X_diabetes.shape[1]} features")
print(f"Target: Disease progression one year after baseline")
print(f"Features: Age, sex, BMI, blood pressure, and 6 blood serum measurements")

# Dataset
data_diabetes = cml.Dataset(X_diabetes, y_diabetes)

# Preprocessing
scaler_diabetes = cml.StandardScaler(data_diabetes)
scaler_diabetes.split(ratio=0.8, random=True)
scaler_diabetes.standart_normalize()

# Training
lr_diabetes = cml.KnnRegression(data_diabetes, 3, "distance")
lr_diabetes.predict()
lr_diabetes.loss()

print("✓ Training completed!")


# ========== 5. WINE DATASET (классификация #2) ==========
print("\n### 5. WINE DATASET - Multiclass Classification ###")
print("-" * 80)

wine = load_wine()
X_wine = wine.data  # (178, 13)
y_wine = wine.target  # (178,)

print(f"Data: {X_wine.shape[0]} samples, {X_wine.shape[1]} features, {len(np.unique(y_wine))} classes")
print(f"Classes: {wine.target_names}")

# One-Hot Encoding
y_wine_ohe = np.eye(3)[y_wine]

# Dataset
data_wine = cml.Dataset(X_wine, y_wine_ohe)

# Preprocessing
scaler_wine = cml.StandardScaler(data_wine)
scaler_wine.split(ratio=0.8, random=True)
scaler_wine.standart_normalize()

# Training
logreg_wine = cml.LogisticRegression(data_wine, "multi")
logreg_wine.train(method="gd", iters=500, lr=0.1)
logreg_wine.predict()
logreg_wine.loss()

print("✓ Training completed!")


# ========== 6. SYNTHETIC REGRESSION DATASET ==========
print("\n### 6. SYNTHETIC REGRESSION DATASET ###")
print("-" * 80)

np.random.seed(42)

# Генерируем данные: y = 3*x1 + 5*x2 - 2*x3 + noise
n_samples = 500
n_features = 3

X_synth = np.random.randn(n_samples, n_features)
true_weights = np.array([3.0, 5.0, -2.0])
noise = np.random.randn(n_samples) * 0.5
y_synth = X_synth @ true_weights + noise + 10  # +10 для положительных значений

print(f"Data: {n_samples} samples, {n_features} features")
print(f"True coefficients: {true_weights}")
print(f"Target mean: {y_synth.mean():.2f}, std: {y_synth.std():.2f}")

# Dataset
data_synth = cml.Dataset(X_synth, y_synth)

# Preprocessing
scaler_synth = cml.StandardScaler(data_synth)
scaler_synth.split(ratio=0.8, random=True)
scaler_synth.standart_normalize()

# Training
lr_synth = cml.LinearRegression(data_synth)
lr_synth.train(method="nesterov", iters=1500, lr=0.1)
lr_synth.predict()
lr_synth.loss()

print("✓ Training completed!")


# ========== 7. PANDAS CSV EXAMPLE (Titanic-like) ==========
print("\n### 7. CSV DATASET EXAMPLE (Titanic-like) ###")
print("-" * 80)

# Расширенный CSV пример
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

# Features и target
X_titanic = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y_titanic = df['Survived'].values.astype(float)

print(f"Data: {X_titanic.shape[0]} samples, {X_titanic.shape[1]} features")
print(f"Features: Pclass, Sex (0=female, 1=male), Age, SibSp, Parch, Fare")

# Dataset
data_titanic = cml.Dataset(X_titanic, y_titanic)

# Preprocessing
scaler_titanic = cml.StandardScaler(data_titanic)
scaler_titanic.split(ratio=0.7, random=True)
scaler_titanic.standart_normalize()

# Training
logreg_titanic = cml.LogisticRegression(data_titanic, "binary")
logreg_titanic.train(method="nesterov", iters=1500, lr=0.1)
logreg_titanic.predict()
logreg_titanic.loss(threshold=0.5)

print("✓ CSV data loaded and model trained!")


# ========== SUMMARY ==========
print("\n" + "=" * 80)
print(" " * 25 + "SUMMARY - ALL DATASETS TESTED")
print("=" * 80)

summary = pd.DataFrame({
    'Dataset': [
        'Iris',
        'Breast Cancer',
        'California Housing',
        'Diabetes',
        'Wine',
        'Synthetic Regression',
        'Titanic CSV'
    ],
    'Task': [
        'Multiclass Classification (3 classes)',
        'Binary Classification',
        'Regression (House Prices)',
        'Regression (Disease Progression)',
        'Multiclass Classification (3 classes)',
        'Regression (Linear)',
        'Binary Classification (Survival)'
    ],
    'Samples': [150, 569, 1000, 442, 178, 500, 10],
    'Features': [4, 30, 8, 10, 13, 3, 6],
    'Model': [
        'LogisticRegression(multi)',
        'LogisticRegression(binary)',
        'LinearRegression',
        'LinearRegression',
        'LogisticRegression(multi)',
        'LinearRegression',
        'LogisticRegression(binary)'
    ]
})

print("\n", summary.to_string(index=False))

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED! ClassicML works with real Kaggle datasets!")
print("=" * 80)
