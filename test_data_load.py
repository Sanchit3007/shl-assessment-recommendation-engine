import pandas as pd

# Load the SHL assessment catalogue CSV
df = pd.read_csv("data/shl_catalogue.csv")

# Print first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Print column names
print("\nColumn names:")
print(df.columns)
