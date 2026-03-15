from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
import numpy as np

X = np.array ([
    [1], [2], [3], [4], [5]
    ])

y = np.array ([35, 45, 48, 52, 65])
z = np.array ([0, 0, 0, 1, 1])
    
Ridge_model = Ridge(alpha=1.0)
Ridge_model.fit(X,y)

linear_model = LinearRegression()
linear_model.fit(X,y)

Lasso_model = Lasso(alpha= 1.0)
Lasso_model.fit(X,y)

logistic_model = LogisticRegression()
logistic_model.fit(X,z)

new_student = [[6.5]]

print("Ridge Regression Prediction: ", Ridge_model.predict(new_student))
print("Lasso Regression Prediction: ", Lasso_model.predict(new_student))
print("Linear Regression Prediction: ", linear_model.predict(new_student))
print("Logistic Regression Prediction: ", logistic_model.predict(new_student))

print("Ridge Coefficients:", Ridge_model.coef_)
print("Lasso Coefficients:", Lasso_model.coef_)
print("linear Coefficients:", linear_model.coef_)
print("possiblity of Logistic Regression: ", logistic_model.predict_proba(new_student) )