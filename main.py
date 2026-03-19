from preprocessing import load_data, preprocess
import importlib
train_module = importlib.import_module("train model")
train_model = train_module.train_model

# Load dataset
df = load_data("starbucks_customer_ordering_patterns.csv")

# Preprocess
x, y = preprocess(df)

# Train
model = train_model(x, y)