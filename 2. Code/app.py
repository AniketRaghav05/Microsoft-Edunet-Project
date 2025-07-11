import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# 💡 App Title and Intro
st.set_page_config(page_title="IntelliSecure Fraud Detection", layout="wide")
st.title("🛡️ IntelliSecure: AI-Powered Financial Fraud Detection System")
st.markdown(
    "Welcome to **IntelliSecure**, a smart system built with machine learning to detect and prevent fraudulent financial transactions in real time. "
)

# Sidebar Mode Selection
mode = st.sidebar.radio("Select Input Mode", ["🧾 Upload CSV", "✍️ Manual Entry"])

# Required feature columns (same as training)
feature_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# 📂 Mode 1: CSV Upload
if mode == "🧾 Upload CSV":
    st.subheader("📄 Upload Transaction Data")
    uploaded_file = st.file_uploader("Choose a CSV file with correct 30 columns", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            # Validate columns
            if list(data.columns) != feature_columns:
                st.error("❌ Column mismatch. Your CSV must have 30 columns: Time, V1–V28, Amount.")
            else:
                # Predict
                predictions = model.predict(data)
                data["Prediction"] = predictions

                st.success("✅ Prediction completed!")
                st.dataframe(data)

                frauds = sum(predictions == 1)
                genuines = sum(predictions == 0)

                st.info(f"🧾 Total Transactions: {len(predictions)}")
                st.success(f"✅ Genuine: {genuines}")
                st.error(f"🚨 Fraudulent: {frauds}")

                # Download results
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv,
                    file_name="intellisecure_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"⚠️ Error while processing the file: {e}")

# ✍️ Mode 2: Manual Entry
else:
    st.subheader("✍️ Manually Enter Transaction Details")
    user_inputs = []

    for col in feature_columns:
        val = st.number_input(col, format="%.6f")
        user_inputs.append(val)

    if st.button("🔍 Predict Transaction"):
        input_df = pd.DataFrame([user_inputs], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        prediction_label = "🚨 FRAUDULENT" if prediction == 1 else "✅ GENUINE"

        if prediction == 1:
            st.error(f"⚠️ This transaction is predicted to be **{prediction_label}**.")
        else:
            st.success(f"🎉 This transaction is predicted to be **{prediction_label}**.")

# Footer
st.markdown("---")
st.markdown("🔐 *Project: IntelliSecure – AI-Based Financial Fraud Detection System*  \n👨‍💻 *Developed by Aniket Raghav*  \n📅 *2025*")
