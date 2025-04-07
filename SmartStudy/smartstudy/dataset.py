import pandas as pd
from pathlib import Path

input_path = Path("SmartStudy\\data\\raw\\databsae.csv")
output_path = Path("SmartStudy\\data\\processed\\processed_data.csv")

data = pd.read_csv(input_path)
print("Original shape:", data.shape)
data.info()

columns_to_drop = ['Ethnicity', 'StudentID', 'GradeClass']
data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')
print("After dropping columns:", data_cleaned.shape)

data_cleaned.to_csv(output_path, index=False)
print(f"Processed dataset saved to: {output_path}")
