import pandas as pd
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

# Load dataset
def load_dataset(path="data/processed/dataset.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["GPA"])  # Drop the target column
    y = df["GPA"]  # Target column
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
    df = pd.DataFrame([user_input])  # Convert user input to DataFrame
    
    # Ensure the DataFrame has the correct columns
    df = df[columns]
    
    # Scale the data and make a prediction
    df_scaled = scaler.transform(df)
    return round(model.predict(df_scaled)[0], 2)

# Coordinate Descent Optimization
def optimize_input(model, scaler, base_input, columns):
    user_df = pd.DataFrame([base_input])  # Convert base input to DataFrame

    # Ensure the DataFrame has the correct columns
    user_df = user_df[columns]

    best_grade = model.predict([scaler.transform(user_df)[0]])[0]
    best_params = pd.DataFrame([base_input])

    # Define parameters to search
    params_to_change = ['Absences', 'StudyTimeWeekly', 'Tutoring', 'Sports', 'Extracurricular', 'Music', 'Volunteering']
    values = {
        'Absences': [0, 5, 10, 20],
        'StudyTimeWeekly': [5, 10, 20, 30],
        'Tutoring': [0, 1],
        'Sports': [0, 1],
        'Extracurricular': [0, 1],
        'Music': [0, 1],
        'Volunteering': [0, 1]
    }

    # Try all combinations
    for param in params_to_change:
        for value in values[param]:
            test_params = best_params.copy()
            test_params[param] = value

            # Ensure the DataFrame has the correct columns
            test_params = test_params[columns]
            scaled = scaler.transform(test_params)
            pred = model.predict([scaled[0]])[0]

            if pred > best_grade:
                best_grade = pred
                best_params[param] = value

    # Finalize output
    best_output = best_params.iloc[0].to_dict()
    best_output["PredictedGPA"] = round(best_grade, 2)
    return best_output
