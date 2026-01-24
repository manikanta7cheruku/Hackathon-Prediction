import pandas as pd

# Load Data
df = pd.read_csv('hackathon_data.csv')

print("--- 1. Data Shape (Rows, Columns) ---")
print(df.shape)

print("\n--- 2. Missing Values Check ---")
print(df.isnull().sum())

print("\n--- 3. Statistics Summary ---")
print(df.describe())

print("\n--- 4. Correlation with Target (Win) ---")
# We assume Tech_Stack_Diversity is still text here, so we skip it for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
print(numeric_df.corr()['Target'].sort_values(ascending=False))