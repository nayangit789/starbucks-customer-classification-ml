import pandas as pd
df = pd.read_csv('starbucks_customer_ordering_patterns.csv')
print(df.head())
print("\nColumns:\n", df.columns)
print("\nInfo:\n")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())