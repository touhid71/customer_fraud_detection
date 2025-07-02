import pandas as pd
import numpy as np

# ডেটা লোড
df = pd.read_csv('data/sample_input.csv')

# লেবেল কলাম বানাও — ২০% ফ্রড হিসেবে র্যান্ডম সিলেকশন
np.random.seed(42)

n = len(df)
fraud_ratio = 0.2
num_fraud = int(n * fraud_ratio)

labels = np.zeros(n, dtype=int)
fraud_indices = np.random.choice(n, num_fraud, replace=False)
labels[fraud_indices] = 1

df['fraud_label'] = labels

# নতুন CSV ফাইল সেভ করো
df.to_csv('data/sample_input_with_label.csv', index=False)

print("New labeled CSV file saved as 'data/sample_input_with_label.csv'")
