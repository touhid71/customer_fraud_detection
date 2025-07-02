import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000  # sample size

# Features simulate করা হচ্ছে
feature1 = np.random.normal(loc=5000, scale=1500, size=n)           # Amount
feature2 = np.random.randint(18, 70, size=n)                        # Age
feature3 = np.random.randint(1, 50, size=n)                         # Number of Transactions
feature4 = np.random.choice([0, 1], size=n, p=[0.8, 0.2])           # Premium Customer (20% yes)
feature5 = np.random.normal(loc=3000, scale=1000, size=n)           # Account Balance

# প্রথমে সব 0 (no fraud) দিয়ে লেবেল বানাও
labels = np.zeros(n, dtype=int)

# 10% fraud randomly assign করো
num_fraud = int(0.1 * n)
fraud_indices = np.random.choice(n, num_fraud, replace=False)
labels[fraud_indices] = 1

# DataFrame বানাও
df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'feature3': feature3,
    'feature4': feature4,
    'feature5': feature5,
    'fraud_label': labels
})

df.to_csv('data/sample_data.csv', index=False)
print("Imbalanced dataset with 10% fraud saved as 'data/sample_data.csv'")
