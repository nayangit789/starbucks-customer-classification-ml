import pandas as pd
def load_data(path):
  return pd.read_csv(path)
def preprocess(df):
  df = df.dropna()
  # Create target variable from total_spend (high_spend if >= median)
  median_spend = df['total_spend'].median()
  df['high_spend'] = (df['total_spend'] >= median_spend).astype(int)
  # Drop non-numeric and ID columns before encoding
  df = df.drop(['customer_id', 'order_id', 'order_date', 'order_time', 'total_spend'], axis=1)
  # Separate target before one-hot encoding
  target = 'high_spend'
  y = df[target].copy()
  x = df.drop(target, axis=1)
  # Apply one-hot encoding only to x
  x = pd.get_dummies(x, drop_first=True)
  return x, y 

