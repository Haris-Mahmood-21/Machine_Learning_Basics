from sklearn.svm import SVC
import numpy as np

X = np.array ( [[1, 2], [2, 3], [8, 9], [9, 10]] )
y = np.array ([0, 0, 1, 1])

svm_model = SVC(kernal = 'linear')
svm_model.fit(X,y)

new_data = [[7, 8]]

prediction = svm_model.predict(new_data)
print(f"SVM Prediction: {prediction[0]}")
