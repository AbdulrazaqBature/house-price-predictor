import os
import numpy as np
import joblib
import gdown
import streamlit as st

# ── Model & encoder file IDs from Google Drive ──────────────────────────
MODEL_FILE_ID   = "1m4Z1naXDe2AhPXjJGHA-ROwlzgfvA7AZ"       # nigeria_house_price_model.pkl
SCALER_FILE_ID  = 'scaler(1).pkl'      # scaler.pkl
TITLE_FILE_ID   = 'le_tittle.pkl'   # le_title.pkl
TOWN_FILE_ID    = 'le_town.pkl'    # le_town.pkl
STATE_FILE_ID   = 'le_state.pkl'    # le_state.pkl

# ── Auto-download files from Google Drive if not present ────────────────
files_to_download = {
    "nigeria_house_price_model.pkl": MODEL_FILE_ID,
    "scaler.pkl":                    SCALER_FILE_ID,
    "le_title.pkl":                  TITLE_FILE_ID,
    "le_town.pkl":                   TOWN_FILE_ID,
    "le_state.pkl":                  STATE_FILE_ID,
}

for filename, file_id in files_to_download.items():
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                filename,
                quiet=False
            )

# ── Load model and encoders ─────────────────────────────────────────────
model    = joblib.load("nigeria_house_price_model.pkl")
scaler   = joblib.load("scaler.pkl")
le_title = joblib.load("le_title.pkl")
le_town  = joblib.load("le_town.pkl")
le_state = joblib.load("le_state.pkl")

# ── Property options ─────────────────────────────────────────────────────
property_types = [
    'Block of Flats', 'Detached Bungalow', 'Detached Duplex',
    'Semi Detached Bungalow', 'Semi Detached Duplex',
    'Terraced Bungalow', 'Terraced Duplexes'
]

states = [
    'Abia', 'Abuja', 'Akwa Ibom', 'Anambara', 'Bayelsa', 'Borno',
    'Cross River', 'Delta', 'Edo', 'Ekiti', 'Enugu', 'Imo', 'Kaduna',
    'Kano', 'Katsina', 'Kogi', 'Kwara', 'Lagos', 'Nasarawa', 'Niger',
    'Ogun', 'Osun', 'Oyo', 'Plateau', 'Rivers'
]

towns = sorted(le_town.classes_.tolist())

# ── App UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nigeria House Price Predictor", page_icon="🏡")

st.title("🏡 Nigeria House Price Predictor")
st.write("Fill in the property details below to get an estimated price in Naira.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    title       = st.selectbox("Property Type", property_types)
    state       = st.selectbox("State", states)
    town        = st.selectbox("Town", towns)

with col2:
    bedrooms      = st.slider("Bedrooms",      1, 10, 3)
    bathrooms     = st.slider("Bathrooms",     1, 10, 2)
    toilets       = st.slider("Toilets",       1, 10, 2)
    parking_space = st.slider("Parking Space", 0, 10, 1)

st.divider()

if st.button("🔍 Predict Price", use_container_width=True):
    try:
        # Encode text inputs
        title_encoded = le_title.transform([title])[0]
        town_encoded  = le_town.transform([town])[0]
        state_encoded = le_state.transform([state])[0]

        # Arrange input in correct column order
        input_data = np.array([[
            bedrooms, bathrooms, toilets, parking_space,
            title_encoded, town_encoded, state_encoded
        ]])

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict (model was trained on log price)
        log_prediction = model.predict(input_scaled)

        # Convert back from log to actual Naira price
        predicted_price = np.expm1(log_prediction)[0]

        # Display result
        st.success(f"### 🏠 Estimated House Price: ₦{predicted_price:,.0f}")
        st.caption(f"Approximately ₦{predicted_price/1_000_000:.1f} million")

    except ValueError as e:
        st.error(f"Input error: {e}. Please check your selections and try again.")
