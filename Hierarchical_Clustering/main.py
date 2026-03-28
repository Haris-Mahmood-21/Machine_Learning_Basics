from sklearn.cluster import AgglomerativeClustering

X_cluster = [[1, 2], [1.5, 1.8], [2, 2.5],  [8, 8], [8.5, 9], [9, 8.5]]


hierarchical = AgglomerativeClustering(n_clusters=2)
labels = hierarchical.fit_predict(X_cluster)

print(f"Hierarchical Labels: {labels}")