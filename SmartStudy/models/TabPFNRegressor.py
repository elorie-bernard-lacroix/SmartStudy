
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabpfn import TabPFNRegressor

# Load dataset
def load_dataset(path="dataset.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["GPA"])
    y = df["GPA"]
    return X, y

# Train TabPFN model and return with scaler
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = TabPFNRegressor(random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Predict GPA from user input
def predict_gpa(model, scaler, user_input, columns):
    df = pd.DataFrame([user_input])[columns]
    df_scaled = scaler.transform(df)
    return round(model.predict(df_scaled)[0], 2)

# Optimization (coordinate search over key fields)
def optimize_input(model, scaler, base_input, columns):
    best_input = base_input.copy()
    best_gpa = predict_gpa(model, scaler, best_input, columns)

    # Search over study time, tutoring, and attendance
    for study_time in range(5, 20, 2):  # try 5 to 20 hours
        for tutoring in [0, 1]:
            for absences in range(0, 10):  # fewer absences
                test_input = base_input.copy()
                test_input["StudyTimeWeekly"] = study_time
                test_input["Tutoring"] = tutoring
                test_input["Absences"] = absences

                predicted = predict_gpa(model, scaler, test_input, columns)
                if predicted > best_gpa:
                    best_input = test_input
                    best_input["PredictedGPA"] = predicted
                    best_gpa = predicted

    best_input["PredictedGPA"] = best_gpa
    return best_input
