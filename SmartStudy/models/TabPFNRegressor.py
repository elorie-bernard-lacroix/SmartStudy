import pandas as pd
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

def load_data(path="dataset.csv"):
    data = pd.read_csv(path)
    X = data.drop(columns=["GPA"])
    y = data["GPA"]
    return X, y

def train_tabpfn_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = TabPFNRegressor(random_state=42)
    model.fit(X_scaled, y_train)

    return model, scaler

def predict_from_input(model, scaler, user_input: dict, columns):
    df = pd.DataFrame([user_input])[columns]
    df_scaled = scaler.transform(df)
    gpa = model.predict(df_scaled)[0]
    return round(gpa, 2)
