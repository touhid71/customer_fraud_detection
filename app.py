import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# মডেল ক্যাশ করে লোড করো
@st.cache_resource
def load_model():
    with open('models/fraud_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="Fraud Checker App", layout="centered")
st.title("Fraud Customer Checker🚨")

rename_dict = {
    'feature1': 'Transaction Amount',
    'feature2': 'Customer Age',
    'feature3': 'Number of Transactions',
    'feature4': 'Is Premium Customer',
    'feature5': 'Account Balance',
}

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data_display = data.rename(columns=rename_dict)

    if 'Is Premium Customer' in data_display.columns:
        data_display['Is Premium Customer'] = data_display['Is Premium Customer'].map({0: 'No', 1: 'Yes'})

    st.subheader("🔎 Uploaded Data Preview:")
    st.dataframe(data_display, use_container_width=True)

    if st.button("🔮 Predict Fraud"):
        data_for_pred = data_display.rename(columns={v: k for k, v in rename_dict.items()})

        if 'feature4' in data_for_pred.columns:
            data_for_pred['feature4'] = data_for_pred['feature4'].map({'No': 0, 'Yes': 1})

        if 'fraud_label' in data_for_pred.columns:
            data_for_pred = data_for_pred.drop('fraud_label', axis=1)

        prediction = model.predict(data_for_pred)
        prediction_labels = ['Fraud' if pred == 1 else 'No Fraud' for pred in prediction]

        result_df = data_display.copy()
        result_df['Fraud Prediction'] = prediction_labels

        # ✅ প্রথমে টেবিল দেখাও
        st.subheader("✅ Prediction Result:")
        st.dataframe(result_df, use_container_width=True)

        # ✅ এরপর pie chart
        plt.close('all')
        st.subheader("📊 Fraud Prediction Summary:")
        counts = result_df['Fraud Prediction'].value_counts()

        fig1, ax1 = plt.subplots(figsize=(3, 3), dpi=70)
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
                colors=['#66b3ff', '#ff6666'], textprops={'fontsize': 10})
        ax1.axis('equal')
        st.pyplot(fig1)

        # ✅ Download CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", data=csv,
                           file_name="predictions.csv", mime="text/csv")
