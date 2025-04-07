import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

input_path = Path("SmartStudy\\data\\raw")
output_dir = Path("SmartStudy\\data\\processed")
output_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(input_path)

X = data.drop(columns=["GPA"])
y = data["GPA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv(output_dir / "train_features.csv", index=False)
y_train.to_csv(output_dir / "train_labels.csv", index=False)
X_test.to_csv(output_dir / "test_features.csv", index=False)
y_test.to_csv(output_dir / "test_labels.csv", index=False)

print("saved to:", output_dir)
