import streamlit as st
import requests

# URL de tu endpoint FastAPI
API_URL = "http://localhost:8000/predict"   # <-- cámbiala si tu API está en la nube

st.title(" Predicción de Precio – Streamlit + FastAPI")

st.write("Completa los valores y presiona Predecir para obtener el resultado del modelo.")

# Formulario
with st.form("input_form"):
    category = st.text_input("Categoría (str)")
    bathrooms = st.number_input("Baños (float)", step=0.1)
    bedrooms = st.number_input("Habitaciones (int)", step=1)
    has_photo = st.selectbox("¿Tiene foto? (str)", ["yes", "no"])
    pets_allowed = st.selectbox("¿Mascotas permitidas? (float)", [0.0, 1.0])
    square_feet = st.number_input("Tamaño en pies cuadrados (int)", step=1)
    cityname = st.text_input("Cityname (str)")   # tu modelo lo tiene como int pero quizá es str
    state = st.text_input("State (str)")
    latitude = st.number_input("Latitud (float)", step=0.0001)
    longitude = st.number_input("Longitud (float)", step=0.0001)
    amenities_count = st.number_input("Cantidad de amenities (int)", step=1)

    submit = st.form_submit_button("Predecir")

if submit:
    payload = {
        "category": category,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "has_photo": has_photo,
        "pets_allowed": pets_allowed,
        "square_feet": square_feet,
        "cityname": cityname,
        "state": state,
        "latitude": latitude,
        "longitude": longitude,
        "amenities_count": amenities_count
    }

    with st.spinner("Calculando predicción..."):
        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                pred = response.json()["prediction"]
                st.success(f"🎯 *Predicción del modelo: {pred}*")
            else:
                st.error(f"Error en el servidor: {response.text}")

        except Exception as e:
            st.error(f"Error al conectar con la API: {e}")