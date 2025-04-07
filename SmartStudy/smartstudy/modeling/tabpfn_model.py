from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_tabpfn(X_train, Y_train):
    reg = TabPFNRegressor(random_state=42)
    reg.fit(X_train, Y_train)
    return reg

def evaluate_tabpfn(reg, X_test, Y_test):
    Y_pred = reg.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    return mse, mae
