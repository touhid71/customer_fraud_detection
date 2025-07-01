import pandas as pd

# Sample data
data = [
    ["C001", "01712345678", "O001", "delivered", "Dhaka", "2025-06-01", ""],
    ["C002", "01812345679", "O002", "canceled", "Chittagong", "2025-06-02", "wrong address"],
    ["C001", "01712345678", "O003", "canceled", "Dhaka", "2025-06-04", "refused to accept"],
    ["C003", "01998765432", "O004", "failed", "Barisal", "2025-06-05", "phone switched off"],
    ["C001", "01712345678", "O005", "canceled", "Dhaka", "2025-06-06", "wrong address"],
    ["C004", "01611112222", "O006", "delivered", "Rajshahi", "2025-06-07", ""],
    ["C002", "01812345679", "O007", "failed", "Chittagong", "2025-06-08", "refused to accept"],
]

# Column names
columns = ["customer_id", "phone", "order_id", "status", "address", "date", "cancel_reason"]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("data/orders.csv", index=False)

print("✅ orders.csv created successfully!")
