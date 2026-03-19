from preprocessing import load_data, preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = load_data('starbucks_customer_ordering_patterns.csv')
x, y = preprocess(df)

print('x shape:', x.shape, 'y shape:', y.shape)

# Manually replicate train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f'x_train: {x_train.shape}, y_train: {y_train.shape}')
print(f'y_train type: {type(y_train)}, y_train shape: {y_train.shape}')

model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)
print("Model trained successfully!")
