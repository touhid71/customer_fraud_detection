import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('data/sample_data.csv')

df = df[(df['feature1'] > 0) & (df['feature5'] >= 0)].copy()

df['amount_to_balance_ratio'] = df['feature1'] / (df['feature5'] + 1)
df['age_times_txn'] = df['feature2'] * df['feature3']
df['feature1_log'] = df['feature1'].apply(lambda x: np.log1p(x))
df['feature5_sqrt'] = df['feature5'].apply(lambda x: np.sqrt(x))
df['txn_per_age'] = df['feature3'] / (df['feature2'] + 1)

X = df.drop('fraud_label', axis=1)
y = df['fraud_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

with open('models/fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ model.pkl saved successfully!")
