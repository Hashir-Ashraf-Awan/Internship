import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load('iris_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler_iris.pkl')

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

@st.cache_data
def load_test_data():
    df = pd.read_csv("Iris.csv")
    X_test = df.drop(['Species','Id'], axis=1).values
    y_test = label_encoder.transform(df['Species'])
    return X_test, y_test

st.title("🌸 Iris Species Predictor + Model Evaluation")

# Input fields
sepal_length = safe_float_input("Sepal Length (cm)", 5.1)
sepal_width = safe_float_input("Sepal Width (cm)", 3.5)
petal_length = safe_float_input("Petal Length (cm)", 1.4)
petal_width = safe_float_input("Petal Width (cm)", 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    scaled_data = scaler.fit_transform(input_data)
    prediction = model.predict(scaled_data)[0]
    species = label_encoder.inverse_transform([prediction])[0]
    st.success(f"🌼 Predicted Species: {species}")

st.subheader("📊 Model Evaluation on Test Set")
X_test, y_test = load_test_data()
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)

report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df.style.format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}))