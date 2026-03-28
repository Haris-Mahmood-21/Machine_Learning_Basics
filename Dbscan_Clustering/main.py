from sklearn.cluster import DBSCAN

X_density = [[1, 2], [1.5, 1.8], [2, 2], [8, 8], [8.5, 9], [9, 8], [50, 50]]

dbscan = DBSCAN(eps=2.0, min_samples=2)
labels = dbscan.fit_predict(X_density)

print(f"DBSCAN Labels: {labels}")