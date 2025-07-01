import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)

data = pd.DataFrame({
    'feature1': np.random.uniform(100, 10000, 200),     # Transaction Amount
    'feature2': np.random.randint(18, 70, 200),         # Customer Age
    'feature3': np.random.randint(1, 50, 200),          # Number of Transactions
    'feature4': np.random.randint(0, 2, 200),           # Is Premium Customer (0 or 1)
    'feature5': np.random.normal(5000, 1500, 200),      # Account Balance
    'fraud_label': np.random.randint(0, 2, 200)         # Fraud Label (0 or 1)
})

data['feature5'] = data['feature5'].apply(lambda x: max(x, 0))  # Account balance can't be negative

data.to_csv('data/your_fraud_data.csv', index=False)
print("✅ Dataset created at data/your_fraud_data.csv")
