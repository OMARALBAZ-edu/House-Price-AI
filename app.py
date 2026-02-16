import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import joblib

# -------------------------------
# Load data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
target = pd.Series(housing.target, name="PRICE")

# -------------------------------
# Train the model once and save it
try:
    model = joblib.load("house_model.pkl")
except:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(data, target)
    joblib.dump(model, "house_model.pkl")

# -------------------------------
# Streamlit UI
st.title("🏠 House Price Predictor")
st.write("Enter the house features below to predict the price.")

# User inputs
user_input = {}
for feature in data.columns:
    user_input[feature] = st.number_input(
        feature,
        float(data[feature].min()),
        float(data[feature].max()),
        float(data[feature].mean())  # default value
    )

# Predict button
if st.button("Predict Price"):
    input_array = np.array([list(user_input.values())])
    prediction = model.predict(input_array)
    st.success(f"💰 Predicted House Price: ${prediction[0]*100000:.2f}")