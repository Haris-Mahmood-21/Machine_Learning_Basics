from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X = np.array([[1, 2], [2, 4], [3, 4], [4, 6], [5, 8], [6, 8]])

y = np.array([0, 0, 0, 1, 1, 1])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

student_new = [[3, 6]]
prediction = knn.predict(student_new)

print(f"KNN Prediction: {prediction[0]}")