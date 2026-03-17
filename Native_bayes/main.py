from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])

y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

nb_classifier = GaussianNB()
nb_classifier.fit(X, y)

hours_new = [[4.5]]
prediction = nb_classifier.predict(hours_new)

print(f"Naive Bayes Prediction: {prediction[0]}")
