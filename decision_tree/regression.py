from sklearn.tree import DecisionTreeRegressor
import numpy as np

X = np.array([[0], [1], [2], [3], [4], [5], [6], [7]])

y = np.array([0, 11, 25, 40, 64, 75, 84, 91])

reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X,y)

new_hours = [[2.3]]

prediction = reg_tree.predict(new_hours)

print (f"dicision tree prediction for {new_hours}: {prediction[0]}")
