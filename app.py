import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------
# Streamlit Page Settings
# ----------------------------
st.set_page_config(
    page_title="Future Sales Predictor",
    page_icon="📊",
    layout="centered"
)

# ----------------------------
# Load Data Function
# ----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv"
    return pd.read_csv(url)

# ----------------------------
# Train Model Function
# ----------------------------
@st.cache_resource
def train_model(data):
    x = np.array(data.drop(["Sales"], axis=1))
    y = np.array(data["Sales"])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    accuracy = model.score(xtest, ytest)
    return model, accuracy, xtest, ytest

# ----------------------------
# UI Layout
# ----------------------------
st.title("📈 Future Sales Prediction App")
st.markdown("Predict future product sales based on advertising costs.")

data = load_data()
model, accuracy, xtest, ytest = train_model(data)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Enter Advertising Costs:")
tv = st.sidebar.number_input("TV Advertising ($)", value=100.0, step=10.0)
radio = st.sidebar.number_input("Radio Advertising ($)", value=50.0, step=5.0)
newspaper = st.sidebar.number_input("Newspaper Advertising ($)", value=20.0, step=5.0)

# ----------------------------
# Prediction Output
# ----------------------------
if st.sidebar.button("Predict Sales"):
    features = np.array([[tv, radio, newspaper]])
    prediction = model.predict(features)[0]
    
    st.success(f"✅ Predicted Sales: **{prediction:.2f}** units")
    st.info(f"📊 Model Accuracy: **{accuracy*100:.2f}%**")

    # 📊 Bar chart comparing with average sales
    avg_sales = data["Sales"].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(["Your Prediction", "Average Sales"], [prediction, avg_sales], color=['green', 'gray'])
    ax2.set_ylabel("Sales")
    ax2.set_title("Predicted vs Average Sales")
    st.pyplot(fig2)

# ----------------------------
# Dataset Preview
# ----------------------------
st.markdown("### 📂 Sample Dataset")
st.dataframe(data.head())

# ----------------------------
# Actual vs Predicted Chart
# ----------------------------
st.subheader("📉 Actual vs Predicted (on Test Set)")
st.caption("🔹 This chart shows model accuracy on test data (not your input).")

y_pred = model.predict(xtest)

fig, ax = plt.subplots()
ax.scatter(ytest, y_pred, color='blue', alpha=0.6, label='Model Predictions')
ax.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], color='red', linestyle='--', label='Perfect Prediction')
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Test Data: Actual vs Predicted Sales")
ax.legend()
st.pyplot(fig)
