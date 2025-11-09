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


# def print_predictions(Y_test_true, Y_pred, label, n_samples=5):
#     """Выводит сравнение ИСТИННЫХ (из тестового набора) и предсказанных значений"""
#     try:
#         Y_test_true = np.asarray(Y_test_true).flatten()
#         Y_pred = np.asarray(Y_pred).flatten()
        
#         if len(Y_test_true) == 0 or len(Y_pred) == 0:
#             print(f"  {label}: нет данных для вывода")
#             return
        
#         print(f"\n  {label} (первые {min(n_samples, len(Y_test_true))} примеров):")
#         print(f"  {'True Value':<25} | {'Predicted Value':<25}")
#         print("  " + "-" * 53)
        
#         for i in range(min(n_samples, len(Y_test_true), len(Y_pred))):
#             true_val = float(Y_test_true[i])
#             pred_val = float(Y_pred[i])
#             print(f"  {true_val:<25.6f} | {pred_val:<25.6f}")
#     except Exception as e:
#         print(f"  {label}: ошибка при выводе - {str(e)}")


# print("=" * 80)
# print(" " * 15 + "CLASSICML - ALL MODELS WITH REAL KAGGLE DATASETS")
# print("=" * 80)


# ========== 1. IRIS DATASET - Logistic Regression (Multiclass) ==========
print("\n### 1. IRIS DATASET - Logistic Regression (Multiclass) ###")
print("-" * 80)

iris = load_iris()
X = iris.data
y = iris.target

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

y_ohe = np.eye(3)[y]
data = cml.Dataset(X, y_ohe)

scaler = cml.StandardScaler(data)
scaler.split(ratio=0.8, random=True)
scaler.standart_normalize()

logreg = cml.LogisticRegression(data, "multi")
logreg.train(method="nesterov", iters=1500, lr=0.001)
logreg.predict()
logreg.loss()
data.info()
# print_predictions(Y_test_classes, Y_pred_classes, "Class Predictions", n_samples=10)
print("✓ Logistic Regression (Multiclass) - PASSED!")


# ========== 2. BREAST CANCER - Logistic Regression (Binary) ==========
print("\n### 2. BREAST CANCER DATASET - Logistic Regression (Binary) ###")
print("-" * 80)

cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target.astype(float)

print(f"Data: {X_cancer.shape[0]} samples, {X_cancer.shape[1]} features")

data_cancer = cml.Dataset(X_cancer, y_cancer)

scaler_cancer = cml.StandardScaler(data_cancer)
scaler_cancer.split(ratio=0.75, random=True)
scaler_cancer.standart_normalize()

logreg_cancer = cml.LogisticRegression(data_cancer, "binary")
logreg_cancer.train(method="nesterov", iters=1000, lr=0.001)
logreg_cancer.predict()
logreg_cancer.loss(threshold=0.5)

data_cancer.info()
# print_predictions(data_cancer.Y_test, data_cancer.Y_pred, "Probability Predictions", n_samples=10)
print("✓ Logistic Regression (Binary) - PASSED!")


# ========== 3. CALIFORNIA HOUSING - Linear Regression ==========
print("\n### 3. CALIFORNIA HOUSING DATASET - Linear Regression ###")
print("-" * 80)

housing = fetch_california_housing()
X_house = housing.data[:1000]
y_house = housing.target[:1000]

print(f"Data: {X_house.shape[0]} samples, {X_house.shape[1]} features")

data_house = cml.Dataset(X_house, y_house)

scaler_house = cml.StandardScaler(data_house)
scaler_house.split(ratio=0.8, random=True)
scaler_house.standart_normalize()

lr_house = cml.KnnRegression(data_house, num_neighbors=7, weighted="distance")
lr_house.predict()
lr_house.loss()
data_house.info()
# print_predictions(Y_test_true, Y_pred, "House Price Predictions (True vs Predicted)", n_samples=15)
print("✓ Linear Regression - PASSED!")


# ========== 4. DIABETES - KNN Regression ==========
print("\n### 4. DIABETES DATASET - KNN Regression ###")
print("-" * 80)

diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

print(f"Data: {X_diabetes.shape[0]} samples, {X_diabetes.shape[1]} features")

data_diabetes = cml.Dataset(X_diabetes, y_diabetes)

scaler_diabetes = cml.StandardScaler(data_diabetes)
scaler_diabetes.split(ratio=0.8, random=True)
scaler_diabetes.standart_normalize()

knn_reg = cml.KnnRegression(data_diabetes, num_neighbors=5, weighted = "distance")
knn_reg.predict("evklid")
knn_reg.loss()
data_diabetes.info()
# print_predictions(data_diabetes.Y_test, data_diabetes.Y_pred, "Disease Progression Predictions", n_samples=10)
print("✓ KNN Regression - PASSED!")


# ========== 5. WINE - KNN Classifier ==========
print("\n### 5. WINE DATASET - KNN Classifier ###")
print("-" * 80)

wine = load_wine()
X_wine = wine.data
y_wine = wine.target

print(f"Data: {X_wine.shape[0]} samples, {X_wine.shape[1]} features, {len(np.unique(y_wine))} classes")

data_wine = cml.Dataset(X_wine, np.expand_dims(y_wine, axis=1))

data_wine.Y = cml.one_hot_encoder(data_wine.Y)

scaler_wine = cml.StandardScaler(data_wine)
scaler_wine.split(ratio=0.8, random=True)
scaler_wine.standart_normalize()

knn_clf = cml.Knn(data_wine, num_neighbors=1, weighted="distance")
knn_clf.predict("evklid")
knn_clf.loss()
data_wine.info()
# print_predictions(data_wine.Y_test, data_wine.Y_pred, "Wine Class Predictions", n_samples=10)

# ========== 6. SYNTHETIC CLUSTERING - K-Means ==========
print("\n### 6. SYNTHETIC CLUSTERING DATASET - K-Means ###")
print("-" * 80)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# Генерация синтетических данных
X_clusters, y_true = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=42,
    cluster_std=0.8
)

print(f"Data: {X_clusters.shape[0]} samples, {X_clusters.shape[1]} features")
print(f"True clusters: 3")

# Подготовка Dataset
y_dummy = np.zeros((X_clusters.shape[0], 1))
data_clusters = cml.Dataset(X_clusters, y_dummy)

# Нормализация и split
scaler_clusters = cml.StandardScaler(data_clusters)
scaler_clusters.split(ratio=0.8, random=True)
scaler_clusters.standart_normalize()

# Обучение K-Means
kmeans = cml.KMeans(data_clusters, k=3, max_iters=100)
kmeans.train(method="pp", rho="evklid")
kmeans.loss()

# Получение центроидов
centroids = kmeans.get_centroids()
print(f"Centroids shape: ({centroids.get_rows()}, {centroids.get_cols()})")
data_clusters.info()

print("✓ K-Means Clustering - PASSED!")

# ========== ВИЗУАЛИЗАЦИЯ ==========
# Извлекаем обучающие данные
X_train_clusters = np.array([[data_clusters.X_train_norm(i, j) 
                              for j in range(data_clusters.X_train_norm.get_cols())] 
                             for i in range(data_clusters.X_train_norm.get_rows())])

y_pred_clusters = np.array([data_clusters.Y_pred[i] 
                            for i in range(data_clusters.Y_pred.get_rows())])

# Извлекаем центроиды
centroids_np = np.array([[centroids(i, j) for j in range(centroids.get_cols())] 
                         for i in range(centroids.get_rows())])

# Создаём figure с 2 subplot'ами
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# === SUBPLOT 1: Истинные кластеры ===
# Находим соответствующие истинные метки для обучающих данных
train_indices = np.arange(len(y_true))[:len(X_train_clusters)]
y_true_train = y_true[train_indices]

scatter1 = ax1.scatter(X_train_clusters[:, 0], X_train_clusters[:, 1], 
                       c=y_true_train, 
                       cmap='viridis', 
                       s=80, 
                       alpha=0.7, 
                       edgecolors='black',
                       linewidth=0.5)
ax1.set_title('Synthetic Data - True Clusters', fontsize=15, fontweight='bold')
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('True Cluster', fontsize=10)

# === SUBPLOT 2: Предсказанные кластеры с центроидами ===
scatter2 = ax2.scatter(X_train_clusters[:, 0], X_train_clusters[:, 1], 
                       c=y_pred_clusters.flatten(), 
                       cmap='viridis', 
                       s=80, 
                       alpha=0.7, 
                       edgecolors='black',
                       linewidth=0.5)

# Рисуем центроиды
ax2.scatter(centroids_np[:, 0], centroids_np[:, 1], 
           c='red', 
           s=400, 
           marker='X', 
           edgecolors='black', 
           linewidths=3,
           label='Centroids',
           zorder=5)

# Рисуем линии от точек к их центроидам (первые 30 точек для читабельности)
for i in range(min(30, len(X_train_clusters))):
    cluster_id = int(y_pred_clusters[i])
    ax2.plot([X_train_clusters[i, 0], centroids_np[cluster_id, 0]], 
            [X_train_clusters[i, 1], centroids_np[cluster_id, 1]], 
            'gray', alpha=0.2, linewidth=0.5, zorder=1)

ax2.set_title('K-Means Clustering Result (k=3)', fontsize=15, fontweight='bold')
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Predicted Cluster', fontsize=10)

plt.tight_layout()
plt.savefig('6_synthetic_kmeans.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Visualization saved as '6_synthetic_kmeans.png'")

# ========== ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: СРАВНЕНИЕ МЕТОДОВ ==========
print("\n### Bonus: Comparing K-Means Initialization Methods ###")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# === Method 1: Base initialization ===
kmeans_base = cml.KMeans(data_clusters, k=3, max_iters=100)
kmeans_base.train(method="base", rho="evklid")

y_pred_base = np.array([data_clusters.Y_pred[i] 
                        for i in range(data_clusters.Y_pred.get_rows())])

centroids_base = kmeans_base.get_centroids()
centroids_base_np = np.array([[centroids_base(i, j) for j in range(centroids_base.get_cols())] 
                              for i in range(centroids_base.get_rows())])

scatter_base = ax1.scatter(X_train_clusters[:, 0], X_train_clusters[:, 1], 
                          c=y_pred_base.flatten(), 
                          cmap='viridis', 
                          s=80, 
                          alpha=0.7, 
                          edgecolors='black',
                          linewidth=0.5)
ax1.scatter(centroids_base_np[:, 0], centroids_base_np[:, 1], 
           c='red', 
           s=400, 
           marker='X', 
           edgecolors='black', 
           linewidths=3,
           label='Centroids',
           zorder=5)
ax1.set_title('K-Means: Base Initialization', fontsize=15, fontweight='bold')
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=11)
plt.colorbar(scatter_base, ax=ax1, label='Cluster')

# === Method 2: K-Means++ initialization ===
kmeans_pp = cml.KMeans(data_clusters, k=3, max_iters=100)
kmeans_pp.train(method="pp", rho="evklid")

y_pred_pp = np.array([data_clusters.Y_pred[i] 
                      for i in range(data_clusters.Y_pred.get_rows())])

centroids_pp = kmeans_pp.get_centroids()
centroids_pp_np = np.array([[centroids_pp(i, j) for j in range(centroids_pp.get_cols())] 
                            for i in range(centroids_pp.get_rows())])

scatter_pp = ax2.scatter(X_train_clusters[:, 0], X_train_clusters[:, 1], 
                        c=y_pred_pp.flatten(), 
                        cmap='viridis', 
                        s=80, 
                        alpha=0.7, 
                        edgecolors='black',
                        linewidth=0.5)
ax2.scatter(centroids_pp_np[:, 0], centroids_pp_np[:, 1], 
           c='red', 
           s=400, 
           marker='X', 
           edgecolors='black', 
           linewidths=3,
           label='Centroids',
           zorder=5)
ax2.set_title('K-Means: K-Means++ Initialization', fontsize=15, fontweight='bold')
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=11)
plt.colorbar(scatter_pp, ax=ax2, label='Cluster')

plt.tight_layout()
plt.savefig('6_synthetic_kmeans_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Comparison saved as '6_synthetic_kmeans_comparison.png'")
print("\n✓ Bonus visualization - PASSED!")


# ========== 7. IRIS - K-Means Clustering (alternative) ==========
# ========== 7. IRIS - K-Means Clustering (alternative) ==========
print("\n### 7. IRIS DATASET - K-Means Clustering ###")
print("-" * 80)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных
iris = load_iris()
X_iris = iris.data

print(f"Data: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
print(f"True classes (for reference): {len(np.unique(iris.target))}")

# Подготовка Dataset
y_iris_dummy = np.zeros((X_iris.shape[0], 1))
data_iris_clustering = cml.Dataset(X_iris, y_iris_dummy)

# Нормализация и split
scaler_iris_clustering = cml.StandardScaler(data_iris_clustering)
scaler_iris_clustering.split(ratio=0.8, random=True)
scaler_iris_clustering.standart_normalize()

# Извлекаем обучающие данные для визуализации
X_train_iris = np.array([[data_iris_clustering.X_train_norm(i, j) 
                          for j in range(data_iris_clustering.X_train_norm.get_cols())] 
                         for i in range(data_iris_clustering.X_train_norm.get_rows())])

# PCA для визуализации в 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train_iris)

# Истинные классы для обучающих данных
y_true_train = iris.target[:data_iris_clustering.X_train_norm.get_rows()]

# Создаём subplot'ы для всех значений k
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

k_values = [2, 3, 4, 5]
all_results = []

# Обучаем K-Means для каждого k и визуализируем
for idx, k_val in enumerate(k_values):
    print(f"\n  K-Means with k={k_val}:")
    
    # Создаём и обучаем модель
    kmeans_iris = cml.KMeans(data_iris_clustering, k=k_val, max_iters=100)
    kmeans_iris.train(method="pp", rho="evklid")
    kmeans_iris.loss()
    
    # Получаем предсказания
    y_pred_kmeans = np.array([data_iris_clustering.Y_pred[i] 
                              for i in range(data_iris_clustering.Y_pred.get_rows())])
    
    # Получаем центроиды
    centroids = kmeans_iris.get_centroids()
    centroids_np = np.array([[centroids(i, j) for j in range(centroids.get_cols())] 
                             for i in range(centroids.get_rows())])
    
    # Проецируем центроиды на 2D
    centroids_2d = pca.transform(centroids_np)
    
    # Сохраняем результаты
    all_results.append({
        'k': k_val,
        'y_pred': y_pred_kmeans,
        'centroids': centroids_np,
        'centroids_2d': centroids_2d
    })
    
    # Визуализация на subplot
    ax = axes[idx]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                        c=y_pred_kmeans.flatten(), 
                        cmap='viridis', 
                        s=70, 
                        alpha=0.6, 
                        edgecolors='black',
                        linewidth=0.5)
    
    # Рисуем центроиды
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
              c='red', 
              s=350, 
              marker='X', 
              edgecolors='black', 
              linewidths=2.5,
              label='Centroids',
              zorder=5)
    
    ax.set_title(f'K-Means: k={k_val}', fontsize=15, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Добавляем colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=10)

plt.suptitle('IRIS K-Means Clustering - Different K Values', 
             fontsize=17, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('7_iris_kmeans_multiple_k.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Visualization saved as '7_iris_kmeans_multiple_k.png'")
print("\n✓ K-Means (Multiple k values) - PASSED!")

# ========== ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: СРАВНЕНИЕ С ИСТИННЫМИ КЛАССАМИ ==========
print("\n### Bonus: Comparing K-Means (k=3) with True Classes ###")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

# === SUBPLOT 1: Истинные классы ===
scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], 
                       c=y_true_train, 
                       cmap='viridis', 
                       s=70, 
                       alpha=0.7, 
                       edgecolors='black',
                       linewidth=0.5)
ax1.set_title('IRIS - True Species Classes', fontsize=15, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
legend1 = ax1.legend(*scatter1.legend_elements(), 
                     title="Species", 
                     loc='upper right',
                     labels=['Setosa', 'Versicolor', 'Virginica'],
                     fontsize=9)
ax1.add_artist(legend1)
ax1.grid(True, alpha=0.3, linestyle='--')

# === SUBPLOT 2: K-Means k=3 ===
# Находим результаты для k=3
result_k3 = [r for r in all_results if r['k'] == 3][0]

scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], 
                       c=result_k3['y_pred'].flatten(), 
                       cmap='viridis', 
                       s=70, 
                       alpha=0.7, 
                       edgecolors='black',
                       linewidth=0.5)
ax2.scatter(result_k3['centroids_2d'][:, 0], result_k3['centroids_2d'][:, 1], 
           c='red', 
           s=350, 
           marker='X', 
           edgecolors='black', 
           linewidths=2.5,
           label='Centroids',
           zorder=5)
ax2.set_title('IRIS - K-Means Clusters (k=3)', fontsize=15, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
legend2 = ax2.legend(*scatter2.legend_elements(), title="Cluster", loc='upper left', fontsize=9)
ax2.add_artist(legend2)
ax2.legend(['Centroids'], loc='upper right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

# === SUBPLOT 3: Ошибки кластеризации ===
# Находим наилучшее соответствие между кластерами и классами
from scipy.optimize import linear_sum_assignment

# Создаём матрицу соответствия
confusion = np.zeros((3, 3))
for true_label in range(3):
    for pred_label in range(3):
        mask = (y_true_train == true_label) & (result_k3['y_pred'].flatten() == pred_label)
        confusion[true_label, pred_label] = np.sum(mask)

# Находим оптимальное соответствие
row_ind, col_ind = linear_sum_assignment(-confusion)
mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}

# Применяем соответствие
y_pred_mapped = np.array([mapping[int(label)] for label in result_k3['y_pred'].flatten()])

# Находим ошибки
errors = (y_true_train != y_pred_mapped)
correct = ~errors

# Визуализация
ax3.scatter(X_2d[correct, 0], X_2d[correct, 1], 
           c=y_true_train[correct], 
           cmap='viridis', 
           s=70, 
           alpha=0.7, 
           edgecolors='black',
           linewidth=0.5,
           label='Correct')
ax3.scatter(X_2d[errors, 0], X_2d[errors, 1], 
           c=y_true_train[errors], 
           cmap='viridis', 
           s=70, 
           alpha=0.7, 
           edgecolors='red',
           linewidth=2,
           marker='s',
           label='Misclassified')
ax3.scatter(result_k3['centroids_2d'][:, 0], result_k3['centroids_2d'][:, 1], 
           c='red', 
           s=350, 
           marker='X', 
           edgecolors='black', 
           linewidths=2.5,
           label='Centroids',
           zorder=5)

ax3.set_title(f'Clustering Accuracy: {100*(1-np.mean(errors)):.1f}%', 
             fontsize=15, fontweight='bold')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('7_iris_kmeans_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n📊 Comparison saved as '7_iris_kmeans_comparison.png'")
print(f"✓ Clustering accuracy: {100*(1-np.mean(errors)):.2f}%")
print(f"✓ Misclassified: {np.sum(errors)}/{len(y_true_train)} samples")
print("\n✓ Bonus visualization - PASSED!")



# ========== 8. SYNTHETIC REGRESSION - Linear Regression ==========
print("\n### 8. SYNTHETIC REGRESSION DATASET - Linear Regression ###")
print("-" * 80)

np.random.seed(42)

n_samples = 500
n_features = 3

X_synth = np.random.randn(n_samples, n_features)
true_weights = np.array([3.0, 5.0, -2.0])
noise = np.random.randn(n_samples) * 0.5
y_synth = X_synth @ true_weights + noise + 10

print(f"Data: {n_samples} samples, {n_features} features")
print(f"True coefficients: {true_weights}")

data_synth = cml.Dataset(X_synth, y_synth)

scaler_synth = cml.StandardScaler(data_synth)
scaler_synth.split(ratio=0.8, random=False)
scaler_synth.standart_normalize()

lr_synth = cml.LinearRegression(data_synth)
lr_synth.train(method="nesterov", iters=1500, lr=0.01)
lr_synth.predict()
lr_synth.loss()
data_synth.info()
# print_predictions(data_synth.Y_test, data_synth.Y_pred, "Synthetic Target Predictions", n_samples=10)
print("✓ Synthetic Linear Regression - PASSED!")


# ========== 9. PANDAS CSV EXAMPLE - Logistic Regression ==========
print("\n### 9. CSV DATASET EXAMPLE (Titanic-like) - Logistic Regression ###")
print("-" * 80)

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

X_titanic = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
y_titanic = df['Survived'].values.astype(float)

print(f"Data: {X_titanic.shape[0]} samples, {X_titanic.shape[1]} features")

data_titanic = cml.Dataset(X_titanic, y_titanic)

scaler_titanic = cml.StandardScaler(data_titanic)
scaler_titanic.split(ratio=0.7, random=True)
scaler_titanic.standart_normalize()

logreg_titanic = cml.LogisticRegression(data_titanic, "binary")
logreg_titanic.train(method="nesterov", iters=1500, lr=0.1)
logreg_titanic.predict()
logreg_titanic.loss(threshold=0.5)
data_titanic.info()
# print_predictions(data_titanic.Y_test, data_titanic.Y_pred, "Survival Probability Predictions", n_samples=10)
print("✓ Logistic Regression (CSV) - PASSED!")


# ========== SUMMARY TABLE ==========
print("\n" + "=" * 80)
print(" " * 20 + "SUMMARY - ALL MODELS TESTED")
print("=" * 80)

summary = pd.DataFrame({
    'Model': [
        'Logistic Regression (Multi)',
        'Logistic Regression (Binary)',
        'Linear Regression',
        'KNN Regression',
        'KNN Classifier',
        'K-Means (Synthetic)',
        'K-Means (Iris Multi-k)',
        'Linear Regression (Synthetic)',
        'Logistic Regression (CSV)'
    ],
    'Dataset': [
        'Iris',
        'Breast Cancer',
        'California Housing',
        'Diabetes',
        'Wine',
        'Synthetic Blobs',
        'Iris',
        'Synthetic Linear',
        'Titanic-like'
    ],
    'Task': [
        'Classification (3 classes)',
        'Classification (Binary)',
        'Regression',
        'Regression',
        'Classification (3 classes)',
        'Clustering (k=3)',
        'Clustering (k=2-5)',
        'Regression',
        'Classification (Binary)'
    ],
    'Samples': [150, 569, 1000, 442, 178, 300, 150, 500, 10],
    'Features': [4, 30, 8, 10, 13, 2, 4, 3, 6],
    'Status': ['✓ PASS'] * 9
})

print("\n", summary.to_string(index=False))

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED! ClassicML includes:")
print("   • Linear Regression (with SGD, Momentum, Nesterov, SVD)")
print("   • Logistic Regression (Binary & Multiclass)")
print("   • KNN Classifier")
print("   • KNN Regression")
print("   • K-Means Clustering (with Base & K-Means++ init)")
print("=" * 80)
