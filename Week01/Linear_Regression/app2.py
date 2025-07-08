import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and preprocessing objects
model = joblib.load('boston_model2.pkl')
poly = joblib.load('poly_features.pkl')
scaler = joblib.load('scaler_boston1.pkl')

# Load dataset (for visualization)
@st.cache_data
def load_data():
    # Assuming you have the original data saved as CSV
    return pd.read_csv("boston.csv")

data = load_data()

st.title("🏡 Boston Housing Price Predictor + Insights")
def safe_float_input(label, default):
    val = st.text_input(label, value=str(default))
    try:
        float_val = float(val)
        if float_val < 0:
            st.error(f"❌ {label} must be a non-negative number.")
            st.stop()
        return float_val
    except ValueError:
        st.error(f"❌ Please enter a valid number for {label}.")
        st.stop()


# User inputs
lstat = safe_float_input("LSTAT", 5.0)
rm = safe_float_input("RM", 6.0)
ptratio = safe_float_input("PTRATIO", 15.0)
indus = safe_float_input("INDUS", 5.0)
tax = safe_float_input("TAX", 300.0)

input_data = np.array([[lstat, rm, ptratio, indus, tax]])

# Predict and show visualizations
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)
    prediction = model.predict(input_poly)[0]

    st.success(f"🏠 Predicted Median House Price: ${prediction:.2f}")


    st.subheader("📦 Boxplot of Features")
    selected_features = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX']
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=data[selected_features], ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("🔍 Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)
