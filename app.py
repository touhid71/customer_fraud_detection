import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import requests

# ✅ মডেল লোড
@st.cache_resource
def load_model():
    with open('models/fraud_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ✅ Streamlit Page Config
st.set_page_config(page_title="Fraud Customer Checker", layout="centered")
st.title("🚨 Fraud Customer Checker")

# ✅ Column Rename Dict
rename_dict = {
    'feature1': 'Transaction Amount',
    'feature2': 'Customer Age',
    'feature3': 'Number of Transactions',
    'feature4': 'Is Premium Customer',
    'feature5': 'Account Balance',
    'amount_to_balance_ratio': 'Amount to Balance Ratio',
    'age_times_txn': 'Age x Transactions',
    'feature1_log': 'Log(Transaction Amount)',
    'feature5_sqrt': 'Sqrt(Account Balance)',
    'txn_per_age': 'Transactions per Age',
}

# ✅ Instruction
st.markdown("""
**📂 Upload a CSV file**  
Your file **must** have these 5 columns:  
`feature1`, `feature2`, `feature3`, `feature4`, `feature5`

App will auto calculate extra features (engineering).
""")

# ✅ Sample CSV Download
sample_url = "https://raw.githubusercontent.com/touhid71/customer_fraud_detection/main/data/sample_data.csv"
response = requests.get(sample_url)
sample_csv = response.text

st.download_button(
    label="📥 Download Sample CSV",
    data=sample_csv,
    file_name='sample_input.csv',
    mime='text/csv'
)

# ✅ File Upload
uploaded_file = st.file_uploader("Upload your CSV", type=['csv'])

required_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # ✅ Check if all required columns exist
        if not all(col in data.columns for col in required_columns):
            st.error(f"❌ Uploaded file is missing one or more required columns: {required_columns}")
        else:
            # ✅ Feature Engineering
            data['amount_to_balance_ratio'] = data['feature1'] / (data['feature5'] + 1)
            data['age_times_txn'] = data['feature2'] * data['feature3']
            data['feature1_log'] = data['feature1'].apply(lambda x: np.log1p(x))
            data['feature5_sqrt'] = data['feature5'].apply(lambda x: np.sqrt(x))
            data['txn_per_age'] = data['feature3'] / (data['feature2'] + 1)

            data_display = data.rename(columns=rename_dict)

            if 'Is Premium Customer' in data_display.columns:
                data_display['Is Premium Customer'] = data_display['Is Premium Customer'].map({0: 'No', 1: 'Yes'})

            st.subheader("🔎 Uploaded Data Preview:")
            st.dataframe(data_display, use_container_width=True)

            if st.button("🔮 Predict Fraud"):
                # Prepare for prediction
                data_for_pred = data.copy()
                features = [
                    'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
                    'amount_to_balance_ratio', 'age_times_txn',
                    'feature1_log', 'feature5_sqrt', 'txn_per_age'
                ]
                data_for_pred = data_for_pred[features]

                prediction = model.predict(data_for_pred)
                prediction_labels = ['Fraud' if p == 1 else 'No Fraud' for p in prediction]

                result_df = data_display.copy()
                result_df['Fraud Prediction'] = prediction_labels

                st.subheader("✅ Prediction Result:")
                st.dataframe(result_df, use_container_width=True)

                # ✅ Pie Chart
                counts = result_df['Fraud Prediction'].value_counts()
                fig1, ax1 = plt.subplots(figsize=(3, 3), dpi=70)
                ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
                        colors=['#66b3ff', '#ff6666'], textprops={'fontsize': 10})
                ax1.axis('equal')
                st.pyplot(fig1)

                # ✅ CSV Download
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions", data=csv,
                                   file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
