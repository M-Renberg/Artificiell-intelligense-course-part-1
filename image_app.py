import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('mnistmodelone.pkl')
    scaler = joblib.load('mnist_scaler.pkl')
    return model, scaler

st.title("Siffer scanner 3000")
st.write("Har du en siffra skriven med dålig handstil? Låt våran model försöka tyda den")

uploaded_file = st.file_uploader("Välj en bild...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uppladdad originalbild", width=250)
    
    st.write("Processar bild...")

    img_gray = image.convert('L')
    img_inverted = ImageOps.invert(img_gray)
    img_resized = img_inverted.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img_resized)
    img_vector = img_array.flatten().reshape(1, -1)
    
    st.image(img_resized, caption="Förprocessad bild (28x28)", width=100)

    if st.button("Prediktera siffra"):
        model, scaler = load_model_and_scaler()
        
        img_vector_scaled = scaler.transform(img_vector) 
        
        prediction = model.predict(img_vector_scaled) 
        
        st.success(f"Modellen gissar att siffran är: {prediction[0]}")
#Run by using "streamlit run image_app.py" in the console