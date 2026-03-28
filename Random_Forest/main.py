from sklearn.ensemble import RandomForestClassifier

X = [[1, 2], [2, 3], [8, 9], [9, 10]]
y = [0, 0, 1, 1]

rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X, y)

prediction = rf_model.predict([[3, 4]])
print(f"Random Forest Prediction: {prediction[0]}")