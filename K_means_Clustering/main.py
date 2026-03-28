from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_cluster = [[1, 2], [1.5, 1.8], [2, 2.5],  [8, 8], [8.5, 9], [9, 8.5]]

kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_cluster)

print(f"K-Means Labels: {cluster_labels}")

score = silhouette_score(X_cluster, cluster_labels)
print(f"Silhouette Score: {score:.2f}")