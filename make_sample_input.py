import pandas as pd

df = pd.read_csv("data/your_fraud_data.csv")

df_features = df.drop('fraud_label', axis=1)

df_features.to_csv("data/sample_input.csv", index=False)
print("Sample input CSV ready: data/sample_input.csv")
