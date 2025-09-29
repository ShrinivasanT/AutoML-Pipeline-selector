import joblib
import pandas as pd

# Load dictionary
obj = joblib.load("Linear Regression AutoML Pipeline.joblib")

# Extract actual pipeline/model
pipeline = obj["model"]   # adjust key if different

# Load dataset
data = pd.read_csv("insurance.csv")

# Drop target column (charges)
X = data.drop("charges", axis=1)

# Take a single sample
sample = X.iloc[[0]]

# Predict
prediction = pipeline.predict(sample)

print("Sample input:")
print(sample)
print("\nPredicted charges:", prediction[0])
