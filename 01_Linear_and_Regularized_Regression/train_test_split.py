from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

X = np.array ([
    [1], [2], [3], [4], [5]
    ])

y = np.array ([35, 45, 48, 52, 65])
z = np.array ([0, 0, 0, 1, 1])


X_train, X_test, y_train, y_test, z_train, z_test = train_test_split (X, y, z, test_size = 0.2, random_state = 42)

lin_model = LinearRegression().fit(X_train, y_train)
ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
Lasso_model = Lasso(alpha=1.0).fit(X_train, y_train)
log_model = LogisticRegression().fit(X_train, z_train)

lin_pred = lin_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lasso_pred = Lasso_model.predict(X_test)
log_pred = log_model.predict(X_test)

mse_lin = mean_squared_error(y_test, lin_pred)
mse_ridge = mean_squared_error(y_test, ridge_pred)
mse_lasso = mean_squared_error(y_test, lasso_pred)

log_acc = accuracy_score(z_test, log_pred)

print(f"Linear MSE: {mse_lin:.2f}")
print(f"Ridge MSE: {mse_ridge:.2f}")
print(f"Lasso MSE: {mse_lasso:.2f}")
print(f"Logistic Accuracy: {log_acc:.0%}")

if mse_lin < mse_ridge and mse_lin < mse_lasso:
    print("Winner: Linear Regression")
elif mse_lasso < mse_ridge and mse_lasso < mse_lin:
    print("Winner: lasso Regression")
else:
    print("Winner: Ridge Regression")
