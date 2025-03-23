import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    df = pd.DataFrame([user_input])
    
    # Add engineered features
    df["ParentalInfluence"] = df["ParentalSupport"] + df["ParentalEducation"]
    df["TutoringEffect"] = df["Tutoring"] + df["StudyTimeWeekly"]

    # Keep only required columns
    df = df[columns]
    df_scaled = scaler.transform(df)
    return round(model.predict(df_scaled)[0], 2)

# Coordinate Descent Optimization
def optimize_input(model, scaler, base_input, columns):
    user_df = pd.DataFrame([base_input])

    # Add engineered features for initial evaluation
    user_df["ParentalInfluence"] = user_df["ParentalSupport"] + user_df["ParentalEducation"]
    user_df["TutoringEffect"] = user_df["Tutoring"] + user_df["StudyTimeWeekly"]
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

            # Add engineered features again
            test_params["ParentalInfluence"] = test_params["ParentalSupport"] + test_params["ParentalEducation"]
            test_params["TutoringEffect"] = test_params["Tutoring"] + test_params["StudyTimeWeekly"]

            test_params = test_params[columns]
            scaled = scaler.transform(test_params)
            pred = model.predict([scaled[0]])[0]

            if pred > best_grade:
                best_grade = pred
                best_params[param] = value

    # Finalize output
    best_params["ParentalInfluence"] = best_params["ParentalSupport"] + best_params["ParentalEducation"]
    best_params["TutoringEffect"] = best_params["Tutoring"] + best_params["StudyTimeWeekly"]

    best_output = best_params.iloc[0].to_dict()
    best_output["PredictedGPA"] = round(best_grade, 2)
    return best_output
