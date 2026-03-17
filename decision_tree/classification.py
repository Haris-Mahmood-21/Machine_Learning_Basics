from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array ([[0], [1], [2], [3], [4], [5], [6], [7]])

y = np.array ([0, 0, 0, 0, 1, 1, 1, 1])

clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X,y)

new_hours = [[3.6]]

prediction = clf_tree.predict(new_hours)

print(f"decision tree prediction for {new_hours}: {prediction[0]}" )