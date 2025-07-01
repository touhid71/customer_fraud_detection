import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"ডেটা লোড সম্পন্ন: {len(data)} রেকর্ড পাওয়া গেছে।")
        return data
    except Exception as e:
        print(f"ডেটা লোড করতে সমস্যা হয়েছে: {e}")
        return None

def fraud_detection_summary(data):
    # canceled বা failed অর্ডারগুলো ফিল্টার করা
    suspicious = data[data['status'].isin(['canceled', 'failed'])]

    # প্রতিটি কাস্টমারের (customer_id) কতগুলো suspicious অর্ডার আছে
    suspicious_counts = suspicious.groupby('customer_id').size().reset_index(name='suspicious_orders')

    # প্রতিটি কাস্টমারের canceled অর্ডার গুলো দেখার জন্য
    canceled_orders = suspicious[suspicious['status'] == 'canceled']

    # canceled অর্ডারে cancel_reason এর সংখ্যা
    cancel_reason_counts = canceled_orders.groupby('customer_id')['cancel_reason'].apply(lambda x: x.notna().sum()).reset_index(name='cancel_reasons')

    # suspicious অর্ডার + cancel_reason যোগ করে fraud_score হিসেব করা যেতে পারে
    fraud_summary = pd.merge(suspicious_counts, cancel_reason_counts, on='customer_id', how='left')
    fraud_summary['cancel_reasons'] = fraud_summary['cancel_reasons'].fillna(0).astype(int)
    fraud_summary['fraud_score'] = fraud_summary['suspicious_orders'] + fraud_summary['cancel_reasons']

    # fraud_score অনুযায়ী সাজানো
    fraud_summary = fraud_summary.sort_values(by='fraud_score', ascending=False)

    print("\n--- সন্দেহভাজন কাস্টমারের তালিকা (Fraud Score সহ) ---")
    print(fraud_summary)

def main():
    file_path = '../data/orders.csv'

    data = load_data(file_path)

    if data is not None:
        print("\nডেটার হেডার দেখুন:")
        print(data.head())

        fraud_detection_summary(data)

if __name__ == "__main__":
    main()
