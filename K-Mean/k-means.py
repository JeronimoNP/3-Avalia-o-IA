import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ==============================
# ETAPA 1 - Carregar os dados
# ==============================

df = pd.read_csv('Mall_Customers.csv')
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# ETAPA 2 - K-Means SEM PCA
# ==============================

print("\nRodando K-Means SEM PCA...")
start = time.time()

kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
clusters_sem_pca = kmeans.fit_predict(X_scaled)

# Métricas SEM PCA
score_sem_pca = silhouette_score(X_scaled, clusters_sem_pca)
inertia_sem_pca = kmeans.inertia_
db_score_sem_pca = davies_bouldin_score(X_scaled, clusters_sem_pca)
ch_score_sem_pca = calinski_harabasz_score(X_scaled, clusters_sem_pca)

end = time.time()

print(f"Silhouette Score SEM PCA: {score_sem_pca:.4f}")
print(f"Inertia SEM PCA: {inertia_sem_pca:.2f}")
print(f"Davies-Bouldin SEM PCA: {db_score_sem_pca:.4f}")
print(f"Calinski-Harabasz SEM PCA: {ch_score_sem_pca:.2f}")
print("Tempo:", round(end - start, 4), "s")

# ==============================
# ETAPA 3 - PCA para 2 Componentes
# ==============================

print("\nAplicando PCA para 2 componentes...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ==============================
# ETAPA 4 - K-Means COM PCA
# ==============================

print("\nRodando K-Means COM PCA...")
start = time.time()

kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init='auto')
clusters_com_pca = kmeans_pca.fit_predict(X_pca)

# Métricas COM PCA
score_com_pca = silhouette_score(X_pca, clusters_com_pca)
inertia_com_pca = kmeans_pca.inertia_
db_score_com_pca = davies_bouldin_score(X_pca, clusters_com_pca)
ch_score_com_pca = calinski_harabasz_score(X_pca, clusters_com_pca)

end = time.time()

print(f"Silhouette Score COM PCA: {score_com_pca:.4f}")
print(f"Inertia COM PCA: {inertia_com_pca:.2f}")
print(f"Davies-Bouldin COM PCA: {db_score_com_pca:.4f}")
print(f"Calinski-Harabasz COM PCA: {ch_score_com_pca:.2f}")
print("Tempo:", round(end - start, 4), "s")

# ==============================
# ETAPA 5 - Visualização
# ==============================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=clusters_com_pca,
    palette='Set1',
    s=60
)
plt.title("Clusters formados após PCA (2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
