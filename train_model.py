import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

os.makedirs("models", exist_ok=True)

data = pd.read_csv('data/your_fraud_data.csv')

X = data.drop('fraud_label', axis=1)
y = data['fraud_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open('models/fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved at models/fraud_model.pkl")
