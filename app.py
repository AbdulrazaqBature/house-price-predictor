import streamlit as st
import numpy as np
import joblib
import gdown
import os

# Google Drive file ID for the model
MODEL_FILE_ID = "1zVfoW_vuMVwOBQ1pL4axJtMj2RiRmFrg"  # replace with your actual file ID
MODEL_PATH = "house_price_model.pkl"

# Download the model only if it doesn't already exist
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait"):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = joblib.load(MODEL_PATH)

# Load the scaler (this one is small, so it's fine to upload directly to GitHub)
scaler = joblib.load('scaler.pkl')

# App title
st.title("🏠 California House Price Predictor")
st.write("Enter the details below to get an estimated house price.")

# Input fields for the user
MedInc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
HouseAge = st.slider("House Age (years)", 1, 52, 20)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Block Population", 100, 5000, 1200)
AveOccup = st.slider("Average Occupants per House", 1.0, 6.0, 3.0)
Latitude = st.slider("Latitude", 32.5, 42.0, 36.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -119.0)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_price = prediction[0] * 100000
    st.success(f"🏡 Estimated House Price: **${predicted_price:,.0f}**")
