import streamlit as st
import pandas as pd
import joblib

model = joblib.load('car_model.pkl')
preprocessor = joblib.load('car_preprocessor.pkl')

st.title("Bilprisprediktion 🚗")
st.write("Mata in bilens uppgifter nedan för att få en prisuppskattning.")

brand = st.selectbox("Märke", ["Toyota", "Volvo", "BMW", "Audi", "Volkswagen", "Kia", "Mercedes", "Chevrolet"])
year = st.number_input("Tillverkningsår", min_value=1990, max_value=2026, value=2020)
engine_size = st.number_input("Motorstorlek (liter)", min_value=0.5, max_value=8.0, value=2.0)
mileage = st.number_input("Miltal (km)", min_value=0, max_value=500000, value=50000)
fuel_type = st.selectbox("Bränsletyp", ["Petrol", "Diesel", "Hybrid", "Electric"])
transmission = st.selectbox("Växellåda", ["Manual", "Automatic", "Semi-Automatic"])
owner_count = st.number_input("Antal tidigare ägare", min_value=1, max_value=10, value=1)

if st.button("Beräkna pris"):
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Model': ['Unknown'],
        'Year': [year],
        'Engine_Size': [engine_size],
        'Mileage': [mileage],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Owner_Count': [owner_count]
    })
    
    processed_input = preprocessor.transform(input_data)
    
    predicted_price = model.predict(processed_input)[0]
    
    st.success(f"Det uppskattade priset är ungefär: {predicted_price:,.0f} $")