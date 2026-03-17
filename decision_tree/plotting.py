import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

X = np.array([
    [1, 2], [2, 4], [3, 4], [4, 6], [5, 8], [6, 8], [7, 6], [8, 9]
    ])

y = np.array(
    [0, 0, 0, 1, 1, 1, 1, 1]
    )

clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X, y)

plt.figure(figsize=(8,6))

display = DecisionBoundaryDisplay.from_estimator(
          clf_tree, X, response_method='predict', cmap = plt.cm.coolwarm, aplha = 0.6
    )

display.ax_.scatter(X[:, 0], X[:, 1] , c=y, cmap = plt.cm.coolwarm, edgecolors="black", s=100)

plt.title("Decision Tree Boundary: Pass(Red) vs Fail(Blue)")
plt.xlabel("Hours Studied")
plt.ylabel("Classes Attended")
plt.show()




