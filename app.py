
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Future Sales Predictor",   # Title in browser tab
    page_icon="📊",                        # Tab icon (emoji or image)
    layout="centered"                      # You can also use 'wide'
)

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv"
    return pd.read_csv(url)

# Train model
@st.cache_resource
def train_model(data):
    x = np.array(data.drop(["Sales"], axis=1))
    y = np.array(data["Sales"])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    return model, model.score(xtest, ytest)

# UI
st.title("📈 Future Sales Prediction App")
st.markdown("Predict future product sales based on advertising costs.")

data = load_data()
model, accuracy = train_model(data)

st.sidebar.header("Enter Advertising Costs:")
tv = st.sidebar.number_input("TV Advertising ($)", value=100.0, step=10.0)
radio = st.sidebar.number_input("Radio Advertising ($)", value=50.0, step=5.0)
newspaper = st.sidebar.number_input("Newspaper Advertising ($)", value=20.0, step=5.0)

if st.sidebar.button("Predict Sales"):
    features = np.array([[tv, radio, newspaper]])
    prediction = model.predict(features)[0]
    
    st.success(f"✅ Predicted Sales: **{prediction:.2f}** units")
    st.info(f"🔍 Model Accuracy: **{accuracy*100:.2f}%**")

st.markdown("### Sample Dataset")
st.dataframe(data.head())
