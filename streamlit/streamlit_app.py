import streamlit as st
import requests

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg",".dcm"])

if uploaded_file:
    try:
        response = requests.post("http://127.0.0.1:8000/predict", files={"file": uploaded_file})
        st.write("Status code:", response.status_code)
        st.write("Raw response text:", response.text)
        # Attempt to parse JSON safely
        data = response.json()  
    except Exception as e:
        st.error(f"Error: {e}")
