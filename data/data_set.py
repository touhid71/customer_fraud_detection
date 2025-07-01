# create_dataset.py
import pandas as pd
import numpy as np

# ১০০টি রো, ৫টি ফিচার (random সংখ্যা)
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.randint(0, 100, 100),
    'feature4': np.random.randint(0, 2, 100),
    'feature5': np.random.normal(0, 1, 100),
    'fraud_label': np.random.randint(0, 2, 100)  # target column (0 or 1)
})

# ফাইল সেভ করো
data.to_csv('data/your_fraud_data.csv', index=False)
print("✅ Dummy dataset created at: data/your_fraud_data.csv")
