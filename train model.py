from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
def train_model(x,y):
     x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,
                                                     random_state=42)
     model = RandomForestClassifier(n_estimators=100)
     model.fit(x_train, y_train)
     y_pred = model.predict(x_test)
     print("Accuracy:", accuracy_score(y_test, y_pred))
     print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
     return model
