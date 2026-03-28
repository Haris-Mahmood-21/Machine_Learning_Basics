from xgboost import XGBClassifier, XGBRegressor

X = [[1, 2], [2, 3], [8, 9], [9, 10]]

y_class = [0, 0, 1, 1]
xgb_clf = XGBClassifier()
xgb_clf.fit(X, y_class)
print(f"XGB Classifier: {xgb_clf.predict([[7, 8]])[0]}") 

y_reg = [30, 45, 85, 95] 
xgb_reg = XGBRegressor()
xgb_reg.fit(X, y_reg)
print(f"XGB Regressor: {xgb_reg.predict([[7, 8]])[0]:.2f}")